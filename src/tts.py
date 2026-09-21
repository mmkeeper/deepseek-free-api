"""DeepSeek voice synthesis (TTS) over its proprietary WebSocket API.

DeepSeek's web UI voices only real chat messages. The flow is:

  POST /api/v0/auth/ticket {"scope":"tts"}  -> ticket (Bearer, TTL 600 s)
  wss://chat.deepseek.com/api/v0/chat/tts/?chat_session_id=...&message_id=...
      &ticket=...&mode=manual&format=opus
      -> JSON events (ready / ack / finish) + binary Opus audio frames

This module downloads the stream, packaging raw Opus packets into a valid
Ogg Opus file (RFC 7845): a passthrough is used when the server already
delivers whole Ogg pages, otherwise raw packets are muxed locally using the
Opus TOC duration table (RFC 6716 Table 2) to compute granule positions.

The caller is responsible for resolving which (chat_session_id, message_id)
pair may be voiced — only messages that were actually produced in the DeepSeek
chat can be synthesized.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import aiohttp

from .client import DeepSeekClient
from .config import BASE_URL, TTS_CACHE_DIR, TTS_PATH, get_socks5_proxy

log = logging.getLogger("ds")

_OGG_SERIAL = 0x64656570  # "deep"
_OGG_CRC_POLY = 0x04C11DB7
_OGG_BOS = 0x02
_OGG_EOS = 0x04

# DeepSeek prefixes every WS audio frame with a 4-byte big-endian counter
# before the actual Opus packet — drop it before muxing.
_WS_COUNTER_LEN = 4

# RFC 6716 Table 2 — Opus TOC configuration number -> per-frame duration (ms).
#    0..11 SILK-only (NB/MB/WB): 10, 20, 40, 60 ms by config % 4
#   12..15 hybrid         (SWB/FB): 10, 20 ms by config % 2
#   16..31 CELT-only (NB/WB/SWB/FB): 2.5, 5, 10, 20 ms by (config - 16) % 4
_OPUS_SAMPLE_RATE = 48000


def strip_ws_counter(payload: bytes) -> bytes:
    """Drop the 4-byte counter prefix from a DeepSeek WS audio frame."""
    if len(payload) <= _WS_COUNTER_LEN:
        return b""
    return payload[_WS_COUNTER_LEN:]


def opus_toc_config_ms(config: int) -> float:
    if config <= 11:
        return (10, 20, 40, 60)[config % 4]
    if config <= 15:
        return (10, 20)[config % 2]
    return (2.5, 5, 10, 20)[(config - 16) % 4]


def opus_packet_samples(packet: bytes) -> int:
    """Number of 48 kHz samples carried by one raw Opus packet."""
    if not packet:
        return 0
    toc = packet[0]
    config = toc >> 3
    code = toc & 0x03
    per_frame = opus_toc_config_ms(config)

    if code == 0:
        frames = 1
    elif code == 3:
        if len(packet) >= 2:
            frames = packet[1] & 0x3F or 1
        else:
            frames = 1
    else:
        frames = 2
    return int(per_frame * _OPUS_SAMPLE_RATE) * frames // 1000


def _ogg_crc(data: bytes) -> int:
    """Ogg CRC-32: non-reflected, init 0, polynomial 0x04C11DB7."""
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ _OGG_CRC_POLY) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def _build_page(payload: bytes, granule: int, header_type: int, seq: int) -> bytes:
    """Build one Ogg page (27-byte header + lacing table + payload)."""
    lacing: list[int] = []
    remaining = len(payload)
    if remaining > 0:
        while remaining >= 255:
            lacing.append(255)
            remaining -= 255
        lacing.append(remaining)  # trailing 0 when payload is an exact multiple of 255
    nseg = len(lacing)

    head = bytearray(27 + nseg)
    head[0:4] = b"OggS"
    head[4] = 0
    head[5] = header_type
    head[6:14] = struct.pack("<q", granule)
    head[14:18] = struct.pack("<I", _OGG_SERIAL)
    head[18:22] = struct.pack("<I", seq)
    head[26] = nseg
    head[27:] = bytes(lacing)

    crc = _ogg_crc(bytes(head) + payload)
    head[22:26] = struct.pack("<I", crc)
    return bytes(head) + payload


def _opus_head(channels: int = 1) -> bytes:
    return (
        b"OpusHead"
        + bytes([1, channels])
        + struct.pack("<H", 0)   # pre-skip
        + struct.pack("<I", _OPUS_SAMPLE_RATE)
        + struct.pack("<H", 0)   # output gain
        + bytes([0])             # mapping family
    )


def _opus_tags() -> bytes:
    vendor = b"deepseek-free-api"
    return b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)


class OggOpusMuxer:
    """Wrap raw Opus packets into a valid Ogg Opus stream.

    Header pages (OpusHead + OpusTags) are emitted together with the first
    audio page. Granule positions accumulate at 48 kHz. finish() appends an
    empty EOS page so players see a clean end of stream.
    """

    def __init__(self, channels: int = 1):
        self._channels = channels
        self._seq = 0
        self._headered = False
        self.granule = 0
        self.packets = 0
        # Last audio page is held back so finish() can mark it EOS instead of
        # appending an (empty) trailer page — an EOS page without packets must
        # not carry a segment table.
        self._last: tuple[bytes, int, int] | None = None  # payload, granule, seq

    def feed(self, payload: bytes) -> bytes:
        pages = []
        if not self._headered:
            pages.append(_build_page(_opus_head(self._channels), 0, _OGG_BOS, self._seq))
            self._seq += 1
            pages.append(_build_page(_opus_tags(), 0, 0, self._seq))
            self._seq += 1
            self._headered = True
        if self._last is not None:
            lp, lg, ls = self._last
            pages.append(_build_page(lp, lg, 0, ls))
        self.granule += opus_packet_samples(payload)
        self._last = (payload, self.granule, self._seq)
        self._seq += 1
        self.packets += 1
        return b"".join(pages)

    def finish(self) -> bytes:
        pages = []
        if self._last is not None:
            lp, lg, ls = self._last
            pages.append(_build_page(lp, lg, _OGG_EOS, ls))
            self._last = None
        return b"".join(pages)


def tts_cache_dir() -> Path:
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TTS_CACHE_DIR


async def download_tts(
    client: DeepSeekClient,
    chat_session_id: str,
    message_id: int,
    voice: str = "mira",
    on_page: Callable[[bytes], None] | None = None,
    on_ready: Callable[[dict], None] | None = None,
    req_id: str = "",
    debug: bool = False,
) -> dict[str, Any]:
    """Synthesize audio for one chat message and stream Ogg pages to on_page.

    on_page receives the exact bytes that make up the cache file (and the
    client response): whole Ogg pages in passthrough mode, muxed pages in raw
    mode. Raises RuntimeError on DeepSeek-side failure (e.g. the message is
    not voiceable).
    """
    ticket = await client.get_tts_ticket(req_id=req_id)
    params = {
        "chat_session_id": chat_session_id,
        "message_id": str(message_id),
        "ticket": ticket,
        "mode": "manual",
        "format": "opus",
    }
    ws_base = "wss://" + BASE_URL.split("://", 1)[1]
    url = f"{ws_base}{TTS_PATH}" + "?" + urlencode(params)

    headers = client._build_headers()
    headers["x-client-bundle-id"] = "com.deepseek.chat"
    headers["x-device-id"] = getattr(client, "device_id", "") or uuid.uuid4().hex
    headers["x-device-model"] = ""

    if debug:
        log.debug(f"[REQ-{req_id}] TTS WS {url} voice={voice}")
    if get_socks5_proxy():
        log.warning(
            "[REQ-%s] TTS: aiohttp ws_connect does not support SOCKS5, "
            "connecting directly",
            req_id,
        )

    meta: dict[str, Any] = {
        "mode": None,
        "packets": 0,
        "bytes": 0,
        "audio_id": None,
        "voice_id": None,
        "trace_id": None,
    }
    muxer = OggOpusMuxer()
    seq = 0

    async with aiohttp.ClientSession() as session:
        ws = await session.ws_connect(url, headers=headers, timeout=30.0)
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    data = bytes(msg.data)
                    if not data:
                        continue
                    if meta["mode"] is None:
                        meta["mode"] = "ogg" if data[:4] == b"OggS" else "raw"
                    seq += 1

                    if meta["mode"] == "raw":
                        packet = strip_ws_counter(data)
                        if not packet:
                            continue
                        chunk = muxer.feed(packet)
                    else:
                        chunk = data

                    if chunk:
                        if on_page:
                            on_page(chunk)
                        meta["packets"] += 1
                        meta["bytes"] += len(chunk)

                    if seq % 8 == 0:
                        try:
                            await ws.send_str(
                                json.dumps(
                                    {
                                        "event": "ack",
                                        "received_seq": seq,
                                        "played_seq": seq,
                                    }
                                )
                            )
                        except Exception:
                            pass
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(msg.data)
                    ev = payload.get("event")
                    if ev == "ready":
                        meta["audio_id"] = payload.get("audio_id")
                        meta["voice_id"] = payload.get("voice_id")
                        meta["trace_id"] = payload.get("trace_id")
                        if on_ready:
                            on_ready(payload)
                    elif ev == "finish":
                        if payload.get("code", 0) != 0:
                            raise RuntimeError(
                                f"DeepSeek TTS error: "
                                f"{payload.get('msg') or payload.get('error') or payload}"
                            )
                        break
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
        finally:
            if meta["mode"] == "raw" and muxer.packets:
                chunk = muxer.finish()
                if chunk:
                    if on_page:
                        on_page(chunk)
                    meta["packets"] += 1
                    meta["bytes"] += len(chunk)
            if not ws.closed:
                await ws.close()

    if meta["packets"] == 0 and (meta["mode"] == "raw" and not muxer.packets):
        raise RuntimeError("DeepSeek TTS returned no audio data")
    return meta