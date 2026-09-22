"""Tests for HTTP routing: method filtering (405 + Allow), trailing-slash
normalization and CORS preflight headers.

The server and the clients must share ONE event loop (Windows Proactor), so
the app lives on a dedicated loop in a background thread."""

import asyncio
import threading

import pytest
from aiohttp import ClientSession
from aiohttp import web

import server
from server import build_app

_loop = asyncio.new_event_loop()
_thread: threading.Thread | None = None
_state = {}


def _call(coro):
    fut = asyncio.run_coroutine_threadsafe(coro, _loop)
    return fut.result()


@pytest.fixture(scope="module", autouse=True)
def _server():
    global _thread
    _thread = threading.Thread(target=_loop.run_forever, daemon=True)
    _thread.start()

    # Voice catalogue pre-seeded so TTS requests never hit the network.
    server._TTS_VOICES = {
        "token": server.auth["token"],
        "voices": [{"voice_id": "mira"}],
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
    _thread.join(timeout=5)


@pytest.fixture()
def base_url(_server):
    return _state["base_url"]


def _request(method: str, url: str, **kwargs) -> tuple[int, dict, dict]:
    async def go():
        async with ClientSession() as cs:
            async with cs.request(method, url, **kwargs) as resp:
                body = await resp.json()
                return resp.status, dict(resp.headers), body
    return _call(go())


def test_wrong_method_tts_gives_405_with_allow(base_url):
    status, headers, body = _request("GET", base_url + "/v1/audio/tts")
    assert status == 405
    assert headers["Allow"].split(", ") == ["POST"]
    assert body["error"] == "method_not_allowed"


def test_wrong_method_chat_gives_405_with_allow(base_url):
    status, headers, _ = _request("DELETE", base_url + "/v1/chat/completions")
    assert status == 405
    assert "POST" in headers["Allow"]


def test_wrong_method_models_gives_405_with_allow(base_url):
    status, headers, _ = _request("POST", base_url + "/v1/models")
    assert status == 405
    assert headers["Allow"].split(", ") == ["GET"]


def test_unknown_path_gives_404(base_url):
    status, _, body = _request("GET", base_url + "/v1/tts/nope")
    assert status == 404
    assert body["error"] == "not_found"


def test_trailing_slash_is_normalized(base_url):
    # POST /v1/audio/tts/ should reach the real handler (empty body → 400),
    # not a 404 from the catch-all.
    status, _, _ = _request("POST", base_url + "/v1/audio/tts/", json={})
    assert status == 400


def test_trailing_slash_wrong_method_still_405(base_url):
    status, headers, _ = _request("GET", base_url + "/v1/audio/tts/")
    assert status == 405
    assert "POST" in headers["Allow"]


def test_preflight_returns_per_path_methods(base_url):
    async def go():
        async with ClientSession() as cs:
            async with cs.options(
                base_url + "/v1/audio/tts",
                headers={
                    "Origin": "http://localhost",
                    "Access-Control-Request-Method": "POST",
                },
            ) as resp:
                return resp.status, dict(resp.headers)
    status, headers = _call(go())
    assert status == 204
    assert "POST" in headers["Access-Control-Allow-Methods"]
    assert "GET" not in headers["Access-Control-Allow-Methods"]
    assert headers["Access-Control-Max-Age"] == "86400"
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_health_has_version(base_url):
    status, _, body = _request("GET", base_url + "/health")
    assert status == 200
    assert body["version"] == "2.0.0"