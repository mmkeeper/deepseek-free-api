"""Tests for the DeepSeek TTS Ogg Opus packaging (src/tts.py)."""

import struct

from src.tts import (
    OggOpusMuxer,
    _build_page,
    _ogg_crc,
    opus_packet_samples,
    opus_toc_config_ms,
    strip_ws_counter,
    tts_ws_url,
)


def test_opus_toc_config_ms_rfc6716_table2():
    assert opus_toc_config_ms(0) == 10
    assert opus_toc_config_ms(1) == 20
    assert opus_toc_config_ms(2) == 40
    assert opus_toc_config_ms(3) == 60
    assert opus_toc_config_ms(7) == 60
    assert opus_toc_config_ms(8) == 10
    assert opus_toc_config_ms(11) == 60
    assert opus_toc_config_ms(12) == 10
    assert opus_toc_config_ms(13) == 20
    assert opus_toc_config_ms(14) == 10
    assert opus_toc_config_ms(15) == 20
    assert opus_toc_config_ms(16) == 2.5
    assert opus_toc_config_ms(17) == 5
    assert opus_toc_config_ms(18) == 10
    assert opus_toc_config_ms(19) == 20
    assert opus_toc_config_ms(24) == 2.5
    assert opus_toc_config_ms(28) == 2.5
    assert opus_toc_config_ms(31) == 20


def test_opus_packet_samples_single_frame():
    # config 0 (SILK NB 10 ms), code 0 → 10 * 48 = 480 samples
    assert opus_packet_samples(bytes([0x00]) + b"\x01") == 480
    # config 9 (SILK WB 20 ms), code 0 → 20 * 48 = 960
    assert opus_packet_samples(bytes([0x48]) + b"\x01") == 960
    # config 13 (hybrid FB 20 ms), code 0 → 960
    assert opus_packet_samples(bytes([0x68]) + b"\x01") == 960
    # config 17 (CELT WB 5 ms), code 0 → 240
    assert opus_packet_samples(bytes([0x88]) + b"\x01") == 240


def test_opus_packet_samples_multiple_frames():
    # config 2 (40 ms/frame), code 1 (2 frames) → 2 * 40 * 48 = 3840
    assert opus_packet_samples(bytes([0x12])) == 3840
    # config 5 (20 ms/frame), code 2 (2 frames) → 2 * 20 * 48 = 1920
    assert opus_packet_samples(bytes([0x2A])) == 1920
    # config 19 (20 ms/frame), code 1 → 1920
    assert opus_packet_samples(bytes([0x99])) == 1920


def test_opus_packet_samples_code3_frame_count():
    # config 28 (2.5 ms/frame), code 3, M=3 → 3 * 2.5 * 48 = 360
    packet = bytes([(28 << 3) | 0x03, 0x03])
    assert opus_packet_samples(packet) == 360
    # M=48 (max allowed) of 2.5 ms frames → 48 * 120 = 5760
    packet = bytes([(28 << 3) | 0x03, 0x30])
    assert opus_packet_samples(packet) == 5760
    # malformed code 3 without frame-count byte → treat as single frame
    assert opus_packet_samples(bytes([0xE3])) == 120


def test_opus_packet_samples_0x6b():
    # observed DeepSeek frame: TOC 0x6b = config 13 (hybrid FB 20 ms), code 3,
    # M=1 → 960 samples
    assert opus_packet_samples(bytes([0x6B, 0x01])) == 960


def test_strip_ws_counter():
    assert strip_ws_counter(b"\x00\x00\x00\x01\x6b\x01") == b"\x6b\x01"
    assert strip_ws_counter(b"\x00\x00\x00\x00") == b""
    assert strip_ws_counter(b"\x00\x00") == b""


def test_muxer_with_counter_prefixed_frames():
    # WS delivers [4-byte BE counter][opus packet]; granule must be computed
    # from the real TOC after stripping, not from the counter bytes.
    muxer = OggOpusMuxer()
    out = bytearray()
    for c in (1, 2):
        out += muxer.feed(strip_ws_counter(bytes([0, 0, 0, c, 0x6B, 0x01])))
    out += muxer.finish()
    eos = None
    for p in _split_pages(out):
        if p[5] & 0x04:
            eos = p
    assert eos is not None
    # two packets of 960 samples each
    assert struct.unpack("<q", eos[6:14])[0] == 1920


def test_ogg_page_crc_valid():
    page = _build_page(b"payload-bytes", 480, 0x02, 0)
    assert page[:4] == b"OggS"
    nseg = page[26]
    header_len = 27 + nseg
    header = page[:header_len]
    body = page[header_len:]
    crc_field = struct.unpack("<I", header[22:26])[0]
    # Recompute CRC over header with zeroed CRC field + body
    zeroed = header[:22] + b"\x00\x00\x00\x00" + header[26:]
    assert crc_field == _ogg_crc(zeroed + body)


def test_ogg_page_lacing_multiple_of_255():
    page = _build_page(b"\x00" * 255, 0, 0, 1)
    # lacing table must carry a trailing 0 to terminate a 255-length segment
    assert page[26] == 2
    assert page[27] == 255
    assert page[28] == 0


def test_ogg_page_empty_has_no_segments():
    # An empty page (e.g. EOS) must not carry a segment table at all.
    page = _build_page(b"", 5, 0x04, 3)
    assert page[26] == 0
    assert len(page) == 27
    # checksum stays valid
    crc_field = struct.unpack("<I", page[22:26])[0]
    zeroed = page[:22] + b"\x00\x00\x00\x00" + page[26:]
    assert crc_field == _ogg_crc(zeroed)


def test_muxer_produces_valid_stream():
    muxer = OggOpusMuxer()
    out = bytearray()
    for _ in range(2):
        out += muxer.feed(bytes([0x48]) + b"\x01")  # 960 samples each
    out += muxer.finish()

    # Should contain OpusHead + OpusTags header pages and audio pages
    assert b"OpusHead" in out
    assert b"OpusTags" in out

    pages = _split_pages(out)
    # head + tags + 2 audio, EOS on the LAST audio page (no empty trailer)
    assert len(pages) == 4

    # First page is BOS with version byte 0, granule 0
    assert pages[0][5] == 0x02
    assert struct.unpack("<q", pages[0][6:14])[0] == 0
    # Last audio page carries EOS with the total granule = 2 * 960
    last = pages[-1]
    assert last[5] & 0x04
    assert struct.unpack("<q", last[6:14])[0] == 1920
    # ...and carries actual audio (no empty trailer page)
    nseg = last[26]
    assert nseg > 0
    assert len(last) > 27 + nseg
    # Page sequence numbers are contiguous
    seqs = [struct.unpack("<I", p[18:22])[0] for p in pages]
    assert seqs == list(range(len(pages)))
    # Every page checksums
    for p in pages:
        nseg = p[26]
        header = p[: 27 + nseg]
        body = p[27 + nseg :]
        zeroed = header[:22] + b"\x00\x00\x00\x00" + header[26:]
        assert struct.unpack("<I", header[22:26])[0] == _ogg_crc(zeroed + body)


def test_muxer_single_packet_marks_last_page_eos():
    muxer = OggOpusMuxer()
    out = bytearray(muxer.feed(bytes([0x6B, 0x01])))  # 960 samples
    out += muxer.finish()
    pages = _split_pages(out)
    # head + tags + audio page (EOS on it, no trailer)
    assert len(pages) == 3
    last = pages[-1]
    assert last[5] & 0x04
    assert struct.unpack("<q", last[6:14])[0] == 960
    nseg = last[26]
    assert nseg > 0


def test_tts_ws_url_no_voice_param():
    from urllib.parse import parse_qs, urlparse

    url = tts_ws_url("sess-42", 7, "tok123")
    parts = urlparse(url)
    assert parts.scheme == "wss"
    assert parts.path.endswith("/api/v0/chat/tts")
    qs = parse_qs(parts.query)
    assert qs["chat_session_id"] == ["sess-42"]
    assert qs["message_id"] == ["7"]
    assert qs["ticket"] == ["tok123"]
    assert qs["mode"] == ["manual"]
    assert qs["format"] == ["opus"]
    assert "voice" not in qs and "voice_id" not in qs


def _split_pages(data: bytes) -> list[bytes]:
    pages = []
    pos = 0
    while pos < len(data):
        assert data[pos : pos + 4] == b"OggS"
        nseg = data[pos + 26]
        header = 27 + nseg
        body_len = sum(data[pos + 27 : pos + header])
        end = pos + header + body_len
        pages.append(data[pos:end])
        pos = end
    assert pos == len(data)
    return pages