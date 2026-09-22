"""Tests for TTS download / voice-switch serialization (server.py).

The account has ONE current voice, so upstream syntheses must never overlap:
a concurrent /v1/audio/tts with a different voice must wait for the in-flight
one and switch the voice only right before its own download. The first request
always establishes the requested voice; a repeat of the current voice does not
re-set it (only the first request of each new voice calls the upstream switch).
"""

import asyncio
import threading
import uuid

import pytest
from aiohttp import ClientSession
from aiohttp import web

import server
from server import build_app, _record_turn

_loop = asyncio.new_event_loop()
_state = {}


def _call(coro, timeout=15):
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result(timeout)


@pytest.fixture(scope="module", autouse=True)
def _server():
    def run_loop():
        _loop.run_forever()

    threading.Thread(target=run_loop, daemon=True).start()

    server._TTS_VOICES = {
        "token": server.auth["token"],
        "voices": [
            {"voice_id": "mira", "is_default": True},
            {"voice_id": "echo", "is_default": False},
            {"voice_id": "stella", "is_default": False},
        ],
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
    _state["started"].set()
    release = _state.get("release")
    if release is not None:
        release.set()  # don't let a stuck download hang the cleanup
    _state["release"] = None
    runner = _state["runner"]

    async def stop():
        await runner.cleanup()

    _call(stop())
    _loop.call_soon_threadsafe(_loop.stop)


@pytest.fixture()
def base_url(_server):
    return _state["base_url"]


@pytest.fixture()
def _reset_state():
    server._TTS_CURRENT_VOICE = None
    server._session_frames.clear()
    server._session_store.clear()
    server._tts_pending.clear()
    server._tts_inflight.clear()
    cache = server.tts.tts_cache_dir()
    for f in list(cache.glob("*.ogg")) + list(cache.glob("*.part")):
        f.unlink(missing_ok=True)
    _state["started"] = asyncio.Event()
    _state["release"] = asyncio.Event()
    _state["actual_voice"] = None


class _FakeClient:
    """Minimal client satisfying _ensure_tts_voice + the fake download."""
    device_id = "test-device"

    def __init__(self, events):
        self.events = events

    async def set_tts_voice(self, voice_id, req_id: str = ""):
        self.events.append(("set", voice_id))

    async def get_tts_ticket(self, req_id=""):
        return "tok"

    def _build_headers(self):
        return {}


@pytest.fixture()
def _fake_down(monkeypatch):
    events: list = []

    async def fake_download(client, session_id, message_id, voice="mira",
                            on_page=None, on_ready=None, req_id="", debug=False):
        # The first download pauses until released; assert it never overlaps.
        _state["active"].append(voice)
        assert len(_state["active"]) == 1, "overlapping TTS downloads"
        events.append(("dl_start", voice))
        release = _state["release"]
        if release is not None:
            _state["started"].set()
            await release.wait()
        if on_page:
            on_page(b"OggS fake-page")
        events.append(("dl_end", voice))
        _state["active"].pop()
        return {"mode": "ogg", "packets": 1, "bytes": 1,
                "audio_id": None, "voice_id": _state.get("actual_voice"),
                "trace_id": None}

    _state["active"] = []
    monkeypatch.setattr(server, "create_client", lambda: _FakeClient(events))
    monkeypatch.setattr(server.tts, "download_tts", fake_download)
    return events


async def _post(url, body):
    async with ClientSession() as cs:
        async with cs.post(url, json=body) as resp:
            data = await resp.read()
            return resp.status, len(data)


def test_concurrent_different_voices_are_serialized(base_url, _fake_down, _reset_state):
    a_sess, b_sess = f"sA-{uuid.uuid4().hex[:8]}", f"sB-{uuid.uuid4().hex[:8]}"
    _record_turn(a_sess, "kA", None, 101, "hi")
    _record_turn(b_sess, "kB", None, 201, "hi")
    events = _fake_down

    async def go():
        url = base_url + "/v1/audio/tts"
        a = asyncio.create_task(_post(
            url, {"chat_session_id": a_sess, "message_index": 0, "voice": "echo"}))
        await _state["started"].wait()  # download A started and holds the lock
        b = asyncio.create_task(_post(
            url, {"chat_session_id": b_sess, "message_index": 0, "voice": "stella"}))
        await asyncio.sleep(0.3)        # give B time to queue on the lock
        # B cannot have started while A's download is still running
        assert ("dl_start", "stella") not in events
        _state["release"].set()
        sa, _ = await a
        sb, _ = await b
        return sa, sb

    sa, sb = _call(go())
    assert sa == 200 and sb == 200
    # voice switch happens right before each download, strictly in request order
    assert events == [
        ("set", "echo"),
        ("dl_start", "echo"),
        ("dl_end", "echo"),
        ("set", "stella"),
        ("dl_start", "stella"),
        ("dl_end", "stella"),
    ]


def test_repeat_voice_is_not_reset(base_url, _fake_down, _reset_state):
    a_sess, b_sess = f"sC-{uuid.uuid4().hex[:8]}", f"sD-{uuid.uuid4().hex[:8]}"
    _record_turn(a_sess, "kC", None, 301, "hi")
    _record_turn(b_sess, "kD", None, 401, "hi")
    events = _fake_down
    _state["release"] = None  # no pausing in this test

    async def go():
        url = base_url + "/v1/audio/tts"
        await _post(url, {"chat_session_id": a_sess, "message_index": 0, "voice": "echo"})
        await _post(url, {"chat_session_id": b_sess, "message_index": 0, "voice": "echo"})

    _call(go())
    set_calls = [v for k, v in events if k == "set"]
    # set exactly once — the second request reused the current voice
    assert set_calls == ["echo"]
    assert events.count(("dl_start", "echo")) == 2


def test_mismatch_ready_voice_triggers_reset(base_url, _fake_down, _reset_state):
    """DeepSeek ignores the voice switch (ready event reports another voice);
    the proxy must remember the ACTUAL voice and re-set on the next request."""
    a_sess = f"sE-{uuid.uuid4().hex[:8]}"
    _record_turn(a_sess, "kE", None, 501, "hi")
    _record_turn(a_sess, "kF", None, 502, "hi")
    events = _fake_down
    _state["release"] = None
    _state["actual_voice"] = "mira"  # every synthesis reports this actual voice

    async def go():
        url = base_url + "/v1/audio/tts"
        await _post(url, {"chat_session_id": a_sess, "message_index": 0, "voice": "echo"})
        await _post(url, {"chat_session_id": a_sess, "message_index": 1, "voice": "echo"})

    _call(go())
    # backend truth is the ready event, not our POST
    assert server._TTS_CURRENT_VOICE == "mira"
    set_calls = [v for k, v in events if k == "set"]
    # first request (account unknown) sets; the mismatch is detected and the
    # second request retries the switch instead of assuming it stuck
    assert set_calls == ["echo", "echo"]


def test_tts_by_messages_voices_reply(base_url, _fake_down, _reset_state):
    """messages (like /v1/chat/completions) resolves the reply to voice without
    chat_session_id — the client need not know the DeepSeek session id."""
    msgs = [{"role": "user", "content": "hi"}]
    nkey = server._hash_messages(server._user_messages(msgs))
    a_sess = f"sM-{uuid.uuid4().hex[:8]}"
    server._session_store[nkey] = (a_sess, 601, False, None)
    server._record_turn(a_sess, nkey, None, 601, "hi")
    events = _fake_down
    _state["release"] = None

    async def go():
        url = base_url + "/v1/audio/tts"
        return await _post(url, {"messages": msgs, "voice": "echo"})

    status, _ = _call(go())
    assert status == 200
    assert events == [("set", "echo"), ("dl_start", "echo"), ("dl_end", "echo")]


async def _post_json(url, body):
    async with ClientSession() as cs:
        async with cs.post(url, json=body) as resp:
            data = await resp.json()
            return resp.status, data


def test_thinking_only_reply_is_clean_400(base_url, _fake_down, _reset_state, monkeypatch):
    """DeepSeek refuses to synthesize a message with no voiceable text
    (think-only reply → finish no_content). The proxy must answer a clean
    contract 400 instead of crashing with a 502 / unretrieved future."""
    a_sess = f"sN-{uuid.uuid4().hex[:8]}"
    _record_turn(a_sess, "kN", None, 701, "hi")

    async def synth_error(client, session_id, message_id, voice="mira",
                          on_page=None, on_ready=None, req_id="", debug=False):
        raise server.tts.TtsSynthesisError(
            "no_content", "Сообщение не содержит текста для озвучки")

    monkeypatch.setattr(server.tts, "download_tts", synth_error)
    _state["release"] = None

    async def go():
        url = base_url + "/v1/audio/tts"
        return await _post_json(url, {"chat_session_id": a_sess,
                                      "message_index": 0, "voice": "mira"})

    status, body = _call(go())
    assert status == 400
    assert body["error"] == "no_content"
    assert server._tts_inflight == {}


def test_generic_tts_failure_is_502(base_url, _fake_down, _reset_state, monkeypatch):
    """A transport/upstream synthesis failure stays a 502 tts_failed and must
    not leak an unretrieved future exception."""
    a_sess = f"sO-{uuid.uuid4().hex[:8]}"
    _record_turn(a_sess, "kO", None, 702, "hi")

    async def synth_boom(client, session_id, message_id, voice="mira",
                         on_page=None, on_ready=None, req_id="", debug=False):
        raise RuntimeError("DeepSeek TTS error: upstream exploded")

    monkeypatch.setattr(server.tts, "download_tts", synth_boom)
    _state["release"] = None

    async def go():
        url = base_url + "/v1/audio/tts"
        return await _post_json(url, {"chat_session_id": a_sess,
                                      "message_index": 0, "voice": "mira"})

    status, body = _call(go())
    assert status == 502
    assert body["error"] == "tts_failed"
    assert server._tts_inflight == {}