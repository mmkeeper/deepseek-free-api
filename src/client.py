from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import BASE_URL, COMPLETION_PATH, STOP_STREAM_PATH, TICKET_PATH, VOICE_PATH, VOICES_PATH
from .headers import base_headers
from .pow import solve_pow
from .proxy import get_http_client
from .sse import DeepSeekError, stream_sse

log = logging.getLogger("ds")

# Backoff (seconds) between retries. DeepSeek answers with a retryable error
# (rate_limit_reached — "Слишком частые сообщения", expert_busy_use_default —
# "Сервер перегружен. Попробуйте позже или используйте быстрый режим", and
# generation_timeout — "Сервер занят, пожалуйста, попробуйте позже." /
# "Server busy, please try again later.") when requests come in too often or
# the server is busy. Retry with doubling backoff up to ~17 min per wait —
# 11 delays, 12 attempts, ~34 minutes total — so a peak-hour rate limit that
# lasts half an hour is survived. Delay values are logged so the real needed
# spacing can be tuned later.
_RETRYABLE_FINISH_REASONS = {
    "rate_limit_reached",
    "expert_busy_use_default",
    "generation_timeout",
}
_RATE_LIMIT_BACKOFF = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Polling backoff (seconds) between fetch_files attempts while a file is being
# processed on DeepSeek's side (status PENDING → SUCCESS). Doubling delays give
# fast feedback for small files (first poll after 0.5s) yet patience for large
# PDFs/xlsx that take tens of seconds to index.
_FILE_READY_BACKOFF = [0.5, 1, 2, 4, 8, 16, 32, 64]


class AuthError(Exception):
    def __init__(self, context: str):
        super().__init__(f"Auth required during {context}")
        self.context = context


class DeepSeekClient:
    def __init__(self, cookie_header: str, token: str, debug: bool = False):
        self.cookie_header = cookie_header
        self.token = token
        self.debug = debug
        self._model_settings: dict[str, dict] | None = None
        self.device_id = uuid.uuid4().hex

    def _build_headers(self) -> dict:
        return base_headers(self.cookie_header, self.token)

    async def get_tts_ticket(self, req_id: str = "") -> str:
        """Fetch a short-lived (600 s) TTS ticket for the WebSocket handshake.

        The ticket is minted per-scope ({"scope":"tts"}) with the same Bearer
        token used for chat sessions, so it must be fetched right before each
        synthesis and never cached.
        """
        client = get_http_client()
        url = f"{BASE_URL}{TICKET_PATH}"
        headers = self._build_headers()
        headers["x-client-bundle-id"] = "com.deepseek.chat"
        headers["x-device-id"] = self.device_id
        headers["x-device-model"] = ""

        if req_id:
            log.debug(f"[REQ-{req_id}] DEEPSEEK POST {TICKET_PATH}")
        resp = await client.post(url, headers=headers, content='{"scope":"tts"}')

        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, ValueError):
            if resp.status_code in (401, 403):
                raise AuthError("tts ticket")
            raise RuntimeError(
                f"TTS ticket: expected JSON, got HTTP {resp.status_code}: "
                f"{resp.text[:180]}"
            )

        if resp.status_code in (401, 403) or data.get("code") in (40002, 40003):
            raise AuthError("tts ticket")
        if resp.is_error or (data.get("code") is not None and data["code"] != 0):
            raise RuntimeError(
                f"DeepSeek API error at {TICKET_PATH}: HTTP {resp.status_code}, "
                f"code {data.get('code')}, msg {data.get('msg', '')}"
            )

        ticket = data.get("data", {}).get("biz_data", {}).get("ticket")
        if not ticket:
            raise RuntimeError(
                f"Cannot read TTS ticket: {json.dumps(data)[:300]}"
            )
        if req_id:
            log.debug(f"[REQ-{req_id}] TTS ticket ok (len={len(ticket)})")
        return ticket

    async def get_tts_voices(self) -> dict:
        """Fetch the TTS voice catalogue once per session.

        Returns {"voices": [...], "default_voice_id": str|None,
        "current_voice_id": str|None} — the same Bearer token as chat is
        used, so the caller caches it keyed by the token.
        """
        client = get_http_client()
        url = f"{BASE_URL}{VOICES_PATH}"
        headers = self._build_headers()
        headers["x-client-bundle-id"] = "com.deepseek.chat"
        headers["x-device-id"] = self.device_id
        headers["x-device-model"] = ""

        resp = await client.get(url, headers=headers)
        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, ValueError):
            if resp.status_code in (401, 403):
                raise AuthError("tts voices")
            raise RuntimeError(
                f"TTS voices: expected JSON, got HTTP {resp.status_code}: "
                f"{resp.text[:180]}"
            )

        if resp.status_code in (401, 403) or data.get("code") in (40002, 40003):
            raise AuthError("tts voices")
        if resp.is_error or (data.get("code") is not None and data["code"] != 0):
            raise RuntimeError(
                f"DeepSeek API error at {VOICES_PATH}: HTTP {resp.status_code}, "
                f"code {data.get('code')}, msg {data.get('msg', '')}"
            )

        biz = data.get("data", {}).get("biz_data", {})
        return {
            "voices": biz.get("voices", []),
            "default_voice_id": biz.get("default_voice_id"),
            "current_voice_id": biz.get("current_voice_id"),
        }

    async def set_tts_voice(self, voice_id: str, req_id: str = "") -> None:
        """Set the account's current TTS voice.

        DeepSeek's TTS WebSocket carries no voice — the server synthesizes
        with the account's current voice, changed via this endpoint. The
        caller is expected to remember the last voice it set and only call
        this on change.
        """
        client = get_http_client()
        url = f"{BASE_URL}{VOICE_PATH}"
        headers = self._build_headers()
        headers["x-client-bundle-id"] = "com.deepseek.chat"
        headers["x-device-id"] = self.device_id
        headers["x-device-model"] = ""

        if req_id:
            log.debug(f"[REQ-{req_id}] DEEPSEEK POST {VOICE_PATH} body={{\"voice_id\": \"{voice_id}\"}}")
        resp = await client.post(url, headers=headers, json={"voice_id": voice_id})
        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, ValueError):
            if resp.status_code in (401, 403):
                raise AuthError("tts voice")
            raise RuntimeError(
                f"TTS voice: expected JSON, got HTTP {resp.status_code}: "
                f"{resp.text[:180]}"
            )

        if resp.status_code in (401, 403) or data.get("code") in (40002, 40003):
            raise AuthError("tts voice")
        if resp.is_error or (data.get("code") is not None and data["code"] != 0):
            raise RuntimeError(
                f"DeepSeek API error at {VOICE_PATH}: HTTP {resp.status_code}, "
                f"code {data.get('code')}, msg {data.get('msg', '')}"
            )
        if req_id:
            log.debug(f"[REQ-{req_id}] TTS voice set: code={data.get('code')} "
                      f"voice_id={voice_id!r}")

    async def fetch_model_settings(self) -> dict[str, dict]:
        """Fetch model settings from DeepSeek and cache them."""
        if self._model_settings is not None:
            return self._model_settings

        did = uuid.uuid4().hex[:32]
        data = await self._request(
            f"/api/v0/client/settings?did={did}&scope=model"
        )
        configs = (
            data.get("data", {})
            .get("biz_data", {})
            .get("settings", {})
            .get("model_configs", {})
            .get("value", [])
        )
        self._model_settings = {}
        for cfg in configs:
            mt = cfg.get("model_type", "")
            self._model_settings[mt] = cfg

        return self._model_settings

    async def get_file_limits(self, model_type: str) -> dict | None:
        """Get file limits for a model type."""
        settings = await self.fetch_model_settings()
        cfg = settings.get(model_type, {})
        return cfg.get("file_feature")

    async def validate_upload(self, filename: str, data: bytes, model_type: str) -> None:
        """Validate file against model limits before upload."""
        limits = await self.get_file_limits(model_type)
        if limits is None:
            raise RuntimeError(
                f"Model '{model_type}' does not support file uploads"
            )

        max_size = limits.get("max_upload_file_size", 0)
        if max_size and len(data) > max_size:
            raise RuntimeError(
                f"File too large: {len(data)} bytes (max {max_size})"
            )

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        allowed = limits.get("support_file_exts", [])
        if allowed and ext and ext not in allowed:
            raise RuntimeError(
                f"File type '.{ext}' not allowed for model '{model_type}'"
            )

    async def _request(self, path: str, method: str = "GET", body: dict | None = None) -> Any:
        client = get_http_client()
        url = f"{BASE_URL}{path}"
        headers = self._build_headers()
        content = json.dumps(body) if body is not None else None

        resp = await client.request(method, url, headers=headers, content=content)
        text = resp.text

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            if resp.status_code in (401, 403):
                raise AuthError(f"HTTP {resp.status_code}")
            raise RuntimeError(
                f"Expected JSON from {path}, got HTTP {resp.status_code}: {text[:180]}"
            )

        if resp.status_code in (401, 403) or data.get("code") in (40002, 40003):
            raise AuthError(f"code {data.get('code', '')}")

        if resp.is_error or (data.get("code") is not None and data["code"] != 0):
            raise RuntimeError(
                f"DeepSeek API error at {path}: HTTP {resp.status_code}, "
                f"code {data.get('code')}, msg {data.get('msg', '')}"
            )

        return data

    async def upload_file(
        self,
        filename: str,
        data: bytes,
        model_type: str = "vision",
        thinking_enabled: bool = True,
        req_id: str = "",
    ) -> str:
        """Upload a file to DeepSeek and return the file_id."""
        import aiohttp

        await self.validate_upload(filename, data, model_type)

        url = f"{BASE_URL}/api/v0/file/upload_file"
        headers = self._build_headers()
        headers.pop("Content-Type", None)

        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        pow_header = await self.create_pow_header("/api/v0/file/upload_file")
        headers["x-ds-pow-response"] = pow_header
        headers["x-file-size"] = str(len(data))
        headers["x-model-type"] = model_type
        headers["x-thinking-enabled"] = "1" if thinking_enabled else "0"
        headers["x-client-bundle-id"] = "com.deepseek.chat"

        form = aiohttp.FormData()
        form.add_field("file", data, filename=filename, content_type=content_type)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form) as resp:
                result = await resp.json()

        if req_id:
            log.debug(f"[REQ-{req_id}] upload_file response: {json.dumps(result)[:2000]}")

        if result.get("code") != 0:
            raise RuntimeError(f"File upload failed: {result.get('msg', 'unknown')}")

        file_id = result["data"]["biz_data"]["id"]
        return file_id

    async def upload_and_confirm(
        self,
        filename: str,
        data: bytes,
        model_type: str = "vision",
        thinking_enabled: bool = True,
        req_id: str = "",
    ) -> str:
        """Upload a file and wait until it is ready, returning its canonical id.

        Files go through an async processing pipeline (PENDING → SUCCESS/FAILED)
        and the completion endpoint rejects not-yet-ready files with
        "invalid ref file id", so confirm via fetch_files before using the id.
        """
        file_id = await self.upload_file(filename, data, model_type=model_type,
                                         thinking_enabled=thinking_enabled, req_id=req_id)
        if req_id:
            log.debug(f"[REQ-{req_id}] uploaded {filename} -> {file_id}")
        files = await self.fetch_files([file_id])
        if req_id:
            log.debug(f"[REQ-{req_id}] fetch_files response: {json.dumps(files)[:2000]}")
        for f in files:
            if f.get("status") == "SUCCESS":
                return f.get("id") or file_id
        raise RuntimeError(
            f"File {file_id} did not become ready "
            f"(statuses={[f.get('status') for f in files]})"
        )

    async def fetch_files(self, file_ids: list[str]) -> list[dict]:
        """Poll file status until all are SUCCESS, doubling wait between attempts."""
        import asyncio

        client = get_http_client()
        url = f"{BASE_URL}/api/v0/file/fetch_files"
        headers = self._build_headers()
        ids_param = ",".join(file_ids)

        for attempt in range(len(_FILE_READY_BACKOFF) + 1):
            resp = await client.get(
                url, headers=headers, params={"file_ids": ids_param}
            )
            data = resp.json()
            files = (
                data.get("data", {}).get("biz_data", {}).get("files", [])
            )
            all_ready = True
            for f in files:
                if f.get("status") not in ("SUCCESS", "FAILED"):
                    all_ready = False
                    break
            if all_ready:
                return files
            if attempt < len(_FILE_READY_BACKOFF):
                await asyncio.sleep(_FILE_READY_BACKOFF[attempt])

        return files

    async def create_session(self) -> str:
        data = await self._request("/api/v0/chat_session/create", "POST", {})
        biz_data = data.get("data", {}).get("biz_data", {})
        # New API: session id is directly in biz_data.id
        session_id = biz_data.get("id")
        if not session_id:
            # Old API: nested in chat_session
            session = biz_data.get("chat_session", {})
            session_id = session.get("id")
        if not session_id:
            raise RuntimeError(
                f"Cannot read chat session id: {json.dumps(data)[:300]}"
            )
        return session_id

    async def create_pow_header(self, target_path: str) -> str:
        data = await self._request(
            "/api/v0/chat/create_pow_challenge",
            "POST",
            {"target_path": target_path},
        )
        challenge = data.get("data", {}).get("biz_data", {}).get("challenge")
        if not challenge:
            raise RuntimeError(
                f"Cannot read PoW challenge: {json.dumps(data)[:300]}"
            )

        answer = await solve_pow(challenge)
        payload = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": target_path,
        }
        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    async def stop_stream(self, session_id: str, message_id: int | None = None) -> bool:
        """Ask DeepSeek to stop the currently generating response in a session.

        The web client calls POST /api/v0/chat/stop_stream when the user hits
        stop — the server stops generation and does NOT commit the partial
        message, keeping the session parent pointer at the previous message.
        """
        body: dict = {"chat_session_id": session_id}
        if message_id is not None:
            body["message_id"] = message_id
        try:
            data = await self._request(STOP_STREAM_PATH, "POST", body)
            return data.get("code") == 0
        except Exception as e:
            log.debug(f"stop_stream failed: {e}")
            return False

    async def _complete_once(
        self,
        session_id: str,
        prompt: str,
        model_type: str | None = None,
        parent_message_id: Any = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        ref_file_ids: list[str] | None = None,
        req_id: str = "",
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_message_id: Callable[[int], None] | None = None,
    ) -> dict:
        pow_header = await self.create_pow_header(COMPLETION_PATH)
        body = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "model_type": model_type,
            "preempt": False,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
        }

        client = get_http_client()
        url = f"{BASE_URL}{COMPLETION_PATH}"
        headers = {**self._build_headers(), "X-DS-PoW-Response": pow_header}
        content = json.dumps(body)

        if req_id:
            log.debug(f"[REQ-{req_id}] DEEPSEEK API POST {COMPLETION_PATH}")
            log.debug(f"[REQ-{req_id}] Request payload ({len(content)} chars): {content[:3000]}")

        async with client.stream("POST", url, headers=headers, content=content) as resp:
            content_type = resp.headers.get("content-type", "")
            if req_id:
                log.debug(f"[REQ-{req_id}] DeepSeek HTTP {resp.status_code} content-type={content_type}")

            if resp.status_code >= 400 or "text/event-stream" not in content_type:
                text = await resp.aread()
                text = text.decode("utf-8", errors="replace")
                if req_id:
                    log.debug(f"[REQ-{req_id}] DeepSeek error response: {text[:1000]}")
                if resp.status_code in (401, 403):
                    raise AuthError("completion")
                try:
                    parsed = json.loads(text)
                    if parsed.get("code") in (40002, 40003):
                        raise AuthError("completion")
                except AuthError:
                    raise
                except (json.JSONDecodeError, ValueError):
                    pass
                raise RuntimeError(f"Completion failed: HTTP {resp.status_code}: {text[:1000]}")

            return await stream_sse(resp, on_text=on_text, on_thinking=on_thinking,
                                    on_message_id=on_message_id, debug=self.debug,
                                    req_id=req_id)

    async def complete(
        self,
        session_id: str,
        prompt: str,
        model_type: str | None = None,
        parent_message_id: Any = None,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        ref_file_ids: list[str] | None = None,
        req_id: str = "",
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        on_message_id: Callable[[int], None] | None = None,
    ) -> dict:
        """Call _complete_once, retrying on retryable errors.

        DeepSeek returns an SSE hint with finish_reason=rate_limit_reached
        ("Слишком частые сообщения") when we hit the per-user rate limit,
        finish_reason=expert_busy_use_default ("Сервер перегружен...") when
        the server is busy, and finish_reason=generation_timeout
        ("Сервер занят, пожалуйста, попробуйте позже.") when generation was
        discarded because the server was too busy to start it. Retry with
        doubling backoff 1..1024 s (11 delays, ~34 minutes total). If the
        request still fails after all retries the last error is propagated to
        the caller.
        Retries happen only when the failure arrived before any content was
        emitted — replaying an already-partially-streamed response would
        duplicate output for the client.
        """
        emitted = {"text": False, "thinking": False}

        def _wrap_text(fn):
            if fn is None:
                return None
            def wrapped(t):
                if t:
                    emitted["text"] = True
                fn(t)
            return wrapped

        def _wrap_thinking(fn):
            if fn is None:
                return None
            def wrapped(t):
                if t:
                    emitted["thinking"] = True
                fn(t)
            return wrapped

        for attempt in range(len(_RATE_LIMIT_BACKOFF) + 1):
            try:
                return await self._complete_once(
                    session_id=session_id,
                    prompt=prompt,
                    model_type=model_type,
                    parent_message_id=parent_message_id,
                    thinking_enabled=thinking_enabled,
                    search_enabled=search_enabled,
                    ref_file_ids=ref_file_ids,
                    req_id=req_id,
                    on_text=_wrap_text(on_text),
                    on_thinking=_wrap_thinking(on_thinking),
                    on_message_id=on_message_id,
                )
            except DeepSeekError as e:
                if e.finish_reason not in _RETRYABLE_FINISH_REASONS:
                    raise
                if any(emitted.values()):
                    log.warning(
                        f"[REQ-{req_id}] {e.finish_reason} after partial output "
                        f"(text={emitted['text']} thinking={emitted['thinking']}) "
                        f"— not retrying, propagating: {e.message}"
                    )
                    raise
                if attempt >= len(_RATE_LIMIT_BACKOFF):
                    log.warning(
                        f"[REQ-{req_id}] {e.finish_reason} — all {len(_RATE_LIMIT_BACKOFF)} "
                        f"retries exhausted (delays={_RATE_LIMIT_BACKOFF}s), propagating error: {e.message}"
                    )
                    raise
                delay = _RATE_LIMIT_BACKOFF[attempt]
                log.warning(
                    f"[REQ-{req_id}] {e.finish_reason} (attempt {attempt + 1}/"
                    f"{len(_RATE_LIMIT_BACKOFF) + 1}) — retry in {delay}s: {e.message}"
                )
                await asyncio.sleep(delay)
        # Unreachable; keep linters happy.
        raise DeepSeekError("rate_limit_reached after all retries", "rate_limit_reached")
