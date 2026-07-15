from __future__ import annotations

import base64
import json
import logging
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Callable

from .config import BASE_URL, COMPLETION_PATH
from .headers import base_headers
from .pow import solve_pow
from .proxy import get_http_client
from .sse import stream_sse

log = logging.getLogger("ds")


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

            return await stream_sse(resp, on_text=on_text, on_thinking=on_thinking, debug=self.debug, req_id=req_id)
