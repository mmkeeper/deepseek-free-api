"""Tests for the TTS voice catalogue: shaping, validation and the
GET /v1/audio/voices endpoint (served from the in-memory cache, no network)."""

import asyncio
import threading

import pytest
from aiohttp import ClientSession
from aiohttp import web

import server
from server import (
    TtsResolveError,
    _TTS_VOICES,
    _validate_voice,
    _voice_entries,
    build_app,
    _resolve_tts_target,
)

_loop = asyncio.new_event_loop()
_state = {}

UPSTREAM_VOICES = [
    {
        "voice_id": "mira",
        "name_i18n": {"zh": "贝壳", "en": "Mira"},
        "description_i18n": {"en": "Versatile & Playful", "zh": "百变活泼"},
        "gender": "female",
        "languages": ["en", "ru", "zh"],
        "demo_urls": {"en": "https://cdn.deepseek.com/chat/tts/voice-demos/mira_en.mp3"},
        "color_palette": {"c1": "#E9F7FF"},
        "is_default": True,
    },
    {
        "voice_id": "echo",
        "name_i18n": {"en": "Echo"},
        "gender": "male",
        "languages": ["en", "ru"],
        "is_default": False,
    },
    {
        "voice_id": "stella",
        "gender": "female",
        "languages": ["en", "zh"],
        "is_default": False,
    },
]


def _call(coro):
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result()


@pytest.fixture(scope="module", autouse=True)
def _server_with_cached_voices():
    def run_loop():
        _loop.run_forever()

    threading.Thread(target=run_loop, daemon=True).start()

    server._TTS_VOICES = {
        "token": server.auth["token"],
        "voices": UPSTREAM_VOICES,
        "default_voice_id": "mira",
        "current_voice_id": "mira",
    }

    async def start():
        app = build_app(port=0)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        _state["runner"] = runner
        return f"http://127.0.0.1:{port}"

    _state["base_url"] = _call(start())
    yield
    server._TTS_VOICES = None

    runner = _state["runner"]

    async def stop():
        await runner.cleanup()

    _call(stop())
    _loop.call_soon_threadsafe(_loop.stop)


@pytest.fixture()
def base_url(_server_with_cached_voices):
    return _state["base_url"]


def _request(method: str, url: str, **kwargs) -> tuple[int, dict, object]:
    async def go():
        async with ClientSession() as cs:
            async with cs.request(method, url, **kwargs) as resp:
                ctype = resp.headers.get("Content-Type", "")
                body = await resp.json() if "json" in ctype else None
                return resp.status, dict(resp.headers), body
    return _call(go())


def test_voice_entries_shapes_upstream():
    entries = _voice_entries(UPSTREAM_VOICES)
    mira = entries[0]
    assert mira["voice_id"] == "mira"
    assert mira["name"] == "Mira"
    assert mira["description"] == "Versatile & Playful"
    assert mira["gender"] == "female"
    assert mira["languages"] == ["en", "ru", "zh"]
    assert mira["is_default"] is True
    # unknown name/description fall back to voice_id / ""
    assert entries[2]["name"] == "stella"
    assert entries[2]["description"] == ""


def test_validate_voice_ok():
    _validate_voice("mira", UPSTREAM_VOICES)
    _validate_voice("echo", UPSTREAM_VOICES)


def test_validate_voice_unknown():
    with pytest.raises(TtsResolveError) as ei:
        _validate_voice("alloy", UPSTREAM_VOICES)
    assert ei.value.error_code == "invalid_voice"
    assert ei.value.param == "voice"


def test_voices_endpoint_serves_cached_catalogue(base_url):
    status, _, body = _request("GET", base_url + "/v1/audio/voices")
    assert status == 200
    assert body["object"] == "list"
    assert body["default_voice_id"] == "mira"
    assert {v["voice_id"] for v in body["data"]} == {"mira", "echo", "stella"}


def test_voices_endpoint_wrong_method_405(base_url):
    status, headers, _ = _request("POST", base_url + "/v1/audio/voices")
    assert status == 405
    assert "GET" in headers["Allow"]


def test_tts_unknown_voice_rejected(base_url):
    status, _, body = _request(
        "POST",
        base_url + "/v1/audio/tts",
        json={"chat_session_id": "sx", "message_index": 0, "voice": "alloy"},
    )
    assert status == 400
    assert body["error"] == "invalid_voice"
    assert body["param"] == "voice"


def test_tts_known_voice_passes_validation(base_url):
    # voice "echo" is allowed; resolution then fails on the unknown session
    # (message_not_voiceable), proving validation ran and passed.
    status, _, body = _request(
        "POST",
        base_url + "/v1/audio/tts",
        json={"chat_session_id": "sx", "message_index": 0, "voice": "echo"},
    )
    assert status == 400
    assert body["error"] == "message_not_voiceable"
    assert body["param"] == "chat_session_id"


def test_default_voice_mira_always_accepted(base_url):
    status, _, body = _request(
        "POST",
        base_url + "/v1/audio/tts",
        json={"chat_session_id": "sx"},
    )
    assert status == 400
    assert body["error"] == "message_not_voiceable"


class _FakeVoiceClient:
    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    async def set_tts_voice(self, voice_id: str, req_id: str = "") -> None:
        self.calls.append(voice_id)
        if self.fail:
            raise RuntimeError("upstream down")


def test_ensure_tts_voice_sets_once(monkeypatch):
    fake = _FakeVoiceClient()
    monkeypatch.setattr(server, "create_client", lambda: fake)
    monkeypatch.setattr(server, "_TTS_CURRENT_VOICE", None)
    _call(server._ensure_tts_voice("echo"))
    _call(server._ensure_tts_voice("echo"))
    assert fake.calls == ["echo"]
    assert server._TTS_CURRENT_VOICE == "echo"
    assert server._TTS_VOICES["current_voice_id"] == "echo"


def test_ensure_tts_voice_reapplies_on_change(monkeypatch):
    fake = _FakeVoiceClient()
    monkeypatch.setattr(server, "create_client", lambda: fake)
    monkeypatch.setattr(server, "_TTS_CURRENT_VOICE", "echo")
    _call(server._ensure_tts_voice("stella"))
    assert fake.calls == ["stella"]
    assert server._TTS_CURRENT_VOICE == "stella"


def test_ensure_tts_voice_fails_open(monkeypatch):
    fake = _FakeVoiceClient(fail=True)
    monkeypatch.setattr(server, "create_client", lambda: fake)
    monkeypatch.setattr(server, "_TTS_CURRENT_VOICE", None)
    _call(server._ensure_tts_voice("echo"))  # must not raise
    # failed set is not remembered, so the next request retries
    assert server._TTS_CURRENT_VOICE is None