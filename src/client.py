from __future__ import annotations

import base64
import json
from typing import Any, Callable

from .config import BASE_URL, COMPLETION_PATH
from .headers import base_headers
from .pow import solve_pow
from .proxy import get_http_client
from .sse import stream_sse


class AuthError(Exception):
    def __init__(self, context: str):
        super().__init__(f"Auth required during {context}")
        self.context = context


class DeepSeekClient:
    def __init__(self, cookie_header: str, token: str, debug: bool = False):
        self.cookie_header = cookie_header
        self.token = token
        self.debug = debug

    def _build_headers(self) -> dict:
        return base_headers(self.cookie_header, self.token)

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
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
        }

        client = get_http_client()
        url = f"{BASE_URL}{COMPLETION_PATH}"
        headers = {**self._build_headers(), "X-DS-PoW-Response": pow_header}
        content = json.dumps(body)

        async with client.stream("POST", url, headers=headers, content=content) as resp:
            content_type = resp.headers.get("content-type", "")
            if resp.status_code >= 400 or "text/event-stream" not in content_type:
                text = await resp.aread()
                text = text.decode("utf-8", errors="replace")
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

            return await stream_sse(resp, on_text=on_text, on_thinking=on_thinking, debug=self.debug)
