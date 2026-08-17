from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import BASE_URL, COMPLETION_PATH, STOP_STREAM_PATH
from .headers import base_headers
from .pow import solve_pow
from .proxy import get_http_client
from .sse import DeepSeekError, stream_sse

log = logging.getLogger("ds")

# Backoff (seconds) between retries. DeepSeek answers with a retryable error
# (rate_limit_reached — "Слишком частые сообщения", expert_busy_use_default —
# "Сервер перегружен. Попробуйте позже или используйте быстрый режим") when
# requests come in too often or the server is busy. Retry with 1, 2, 4, 8, 16,
# 32, 64 s — 7 attempts total before the error is propagated to the client.
# Delay values are logged so the real needed spacing can be tuned later.
_RETRYABLE_FINISH_REASONS = {"rate_limit_reached", "expert_busy_use_default"}
_RATE_LIMIT_BACKOFF = [1, 2, 4, 8, 16, 32, 64]


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

    def _build_headers(self) -> dict:
        return base_headers(self.cookie_header, self.token)

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

        if result.get("code") != 0:
            raise RuntimeError(f"File upload failed: {result.get('msg', 'unknown')}")

        file_id = result["data"]["biz_data"]["id"]
        return file_id

    async def fetch_files(self, file_ids: list[str]) -> list[dict]:
        """Poll file status until all are SUCCESS."""
        import asyncio

        client = get_http_client()
        url = f"{BASE_URL}/api/v0/file/fetch_files"
        headers = self._build_headers()
        ids_param = ",".join(file_ids)

        for _ in range(20):
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
            await asyncio.sleep(0.5)

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
        ("Слишком частые сообщения") when we hit the per-user rate limit, or
        finish_reason=expert_busy_use_default ("Сервер перегружен...") when
        the server is busy. Retry with backoff 1, 2, 4, 8, 16, 32, 64 s. If
        the request still fails after all retries the last error is propagated
        to the caller.
        Retries happen only when the failure arrived before any content was
        emitted — replaying an already-partially-streamed response would
        duplicate output for the client.
        """
        emitted = {"text": False, "thinking": False, "message_id": False}

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

        def _wrap_message_id(fn):
            if fn is None:
                return None
            def wrapped(m):
                emitted["message_id"] = True
                fn(m)
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
                    on_message_id=_wrap_message_id(on_message_id),
                )
            except DeepSeekError as e:
                if e.finish_reason not in _RETRYABLE_FINISH_REASONS:
                    raise
                if any(emitted.values()):
                    log.warning(
                        f"[REQ-{req_id}] {e.finish_reason} after partial output "
                        f"(text={emitted['text']} thinking={emitted['thinking']} "
                        f"msg={emitted['message_id']}) — not retrying, propagating: {e.message}"
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
