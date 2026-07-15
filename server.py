#!/usr/bin/env python3
"""
DeepSeek Free -> OpenAI-совместимый прокси.

Использует браузерную сессию DeepSeek (бесплатно) и предоставляет
OpenAI-совместимый REST API для любых клиентов.

Запуск:          python server.py
Первый вход:     python server.py --login
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

DEBUG = False  # Переключи в True для отладки или используй --debug

from aiohttp import web

from src.auth import (
    connect_to_running_chrome,
    import_cookies,
    login_and_save_auth,
    print_manual_instructions,
    read_saved_auth,
)
from src.client import AuthError, DeepSeekClient
from src.config import BASE_URL
from src.proxy import get_proxy_info


# ─── Config ───────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeepSeek Free -> OpenAI Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python server.py                          Запуск сервера
  python server.py --login                  Войти через Playwright
  python server.py --connect                Забрать сессию из Chrome
  python server.py --proxy 127.0.0.1:1080   Через SOCKS5 прокси
  python server.py --no-thinking            Выключить мышление по умолчанию
  python server.py --no-search              Выключить поиск по умолчанию
""",
    )
    p.add_argument("--port", type=int, default=None, help="Listen port (default: 18632)")
    p.add_argument("--host", default=None, help="Listen host (default: 0.0.0.0)")
    p.add_argument("--proxy", default=None, help="SOCKS5 proxy (socks5://host:port)")
    p.add_argument("--api-key", default=None, help="API key for client auth")
    p.add_argument("--no-thinking", action="store_true", help="Disable thinking by default")
    p.add_argument("--no-search", action="store_true", help="Disable search by default")
    p.add_argument("--login", action="store_true", help="Логин через Playwright")
    p.add_argument("--connect", nargs="?", const=9222, type=int, metavar="PORT",
                   help="Подключиться к Chrome через CDP")
    p.add_argument("--import", nargs=2, metavar=("COOKIES", "TOKEN"), dest="import_cookies",
                   help="Импорт cookies.json + userToken")
    p.add_argument("--manual", action="store_true",
                   help="Показать инструкцию по ручному экспорту")
    p.add_argument("--debug", action="store_true",
                   help="Включить отладочное логирование в файл")
    return p.parse_args()


# ─── Auth state ───────────────────────────────────────────

auth = {"cookieHeader": "", "token": ""}

# ─── Feature defaults (overridden by CLI args) ─────────────

default_thinking = True
default_search = True


async def init_auth(force_login: bool = False):
    global auth

    if force_login:
        try:
            from src.auth import _launch_persistent_context
            from playwright.async_api import async_playwright
            async with async_playwright() as pw:
                ctx = await _launch_persistent_context(pw.chromium, True)
                try:
                    await ctx.clear_cookies()
                finally:
                    await ctx.close()
        except Exception:
            pass

        result = await login_and_save_auth()
        auth["cookieHeader"] = result["cookieHeader"]
        auth["token"] = result["token"]
        print("[auth] Новый вход выполнен успешно")
        return

    saved = read_saved_auth()
    if saved:
        auth["cookieHeader"] = saved["cookieHeader"]
        auth["token"] = saved["token"]
        print("[auth] Загружена сохранённая авторизация, проверяю...")
        try:
            client = create_client()
            await client.create_session()
            print("[auth] Токен валиден")
            return
        except AuthError:
            print("[auth] Токен истёк, открываю окно логина...")
        except Exception as e:
            print(f"[auth] Ошибка проверки: {e}, открываю окно логина...")

    result = await login_and_save_auth()
    auth["cookieHeader"] = result["cookieHeader"]
    auth["token"] = result["token"]
    print("[auth] Авторизация получена")


def create_client() -> DeepSeekClient:
    return DeepSeekClient(
        cookie_header=auth["cookieHeader"],
        token=auth["token"],
        debug=DEBUG or logging.getLogger().isEnabledFor(logging.DEBUG),
    )


# ─── Session store (reuse DeepSeek sessions within one server run) ───

_session_store: dict[str, tuple[str, int | None, bool]] = {}

# ─── Logging ────────────────────────────────────────────────

def _log_setup(enabled: bool, log_dir: str | None = None):
    """Configure logging — structured format with timestamps, writes to
    logs/ds_session.log (or a custom directory) + stderr when --debug is on."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if enabled else logging.CRITICAL)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%f",
    )
    fmt.converter = time.gmtime  # UTC timestamps

    if enabled:
        # File handler — always on when debug is enabled
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, "ds_session.log"),
            mode="a", encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    else:
        # Even when disabled, keep a console handler for CRITICAL
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.CRITICAL)
        ch.setFormatter(fmt)
        root.addHandler(ch)


log = logging.getLogger("ds")


# ─── Request correlation ID ─────────────────────────────────

_req_counter = 0


def _req_id() -> str:
    global _req_counter
    _req_counter += 1
    return f"{uuid.uuid4().hex[:6]}{_req_counter:04x}"


def rlog(req_id: str, msg: str):
    """Log with request correlation id prefix."""
    log.debug(f"[REQ-{req_id}] {msg}")


# ─── XML tag stripping — keep content clean from tool markup ─

_TOOL_TAG_RE = re.compile(r'</?(?:tool_calls|tool_call|invoke|parameter)[^>]*>')


def _strip_tool_tags(text: str) -> str:
    return _TOOL_TAG_RE.sub("", text)

PREFIX = "dsf-"

def strip_prefix(model: str) -> str:
    return model[len(PREFIX):] if model.startswith(PREFIX) else model

def _hash_messages(msgs: list[dict]) -> str:
    raw = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _user_messages(msgs: list[dict]) -> list[dict]:
    """Extract only user + system messages — stable across turns."""
    return [{"role": m["role"], "content": m.get("content", "")}
            for m in msgs if m.get("role") in ("user", "system")]


def _prefix_key(messages: list[dict]) -> str:
    """Hash of user/system messages in prefix (all except last user turn)."""
    prefix = messages[:-1] if len(messages) >= 1 else []
    return _hash_messages(_user_messages(prefix))


# ─── OpenAI -> DeepSeek conversion ─────────────────────────

def messages_to_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    parts = []
    if tools:
        tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
        tool_descs = []
        for t in tools:
            func = t.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            param_props = params.get("properties", {})
            param_names = list(param_props.keys())
            tool_descs.append(f"  - {name}: {desc} (params: {param_names})")
        tools_text = chr(10).join(tool_descs)
        tool_names_str = ", ".join(tool_names)
        lt = chr(60)
        gt = chr(62)
        tc_open = lt + "tool_calls" + gt
        tc_close = lt + "/tool_calls" + gt
        inv_open = lt + "invoke" + gt
        inv_close = lt + "/invoke" + gt
        param_open = lt + "parameter" + gt
        param_close = lt + "/parameter" + gt
        tool_header = "You have access to the following tools. To call a tool, respond with:" + chr(10)
        tool_header += tc_open + chr(10)
        tool_header += "  " + inv_open + chr(32) + "name=" + chr(34) + "TOOL_NAME" + chr(34) + chr(32) + inv_close + chr(10)
        tool_header += "    " + param_open + chr(32) + "name=" + chr(34) + "PARAM_NAME" + chr(34) + chr(32) + param_close + " VALUE " + lt + "/parameter" + gt + chr(10)
        tool_header += "  " + lt + "/invoke" + gt + chr(10)
        tool_header += tc_close + chr(10)
        tool_header += "Available tools: " + tool_names_str + chr(10)
        tool_header += tools_text + chr(10)
        tool_header += "Only call tools when the user explicitly asks. Otherwise respond normally." + chr(10)
        tool_header += chr(10)
    for m in messages:
        role_map = {"assistant": "Assistant", "system": "System"}
        role = role_map.get(m.get("role"), "User")
        content = ""
        c = m.get("content")
        if isinstance(c, str):
            content = c
        elif isinstance(c, list):
            # Extract text and check for images
            texts = []
            has_images = False
            for item in c:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    has_images = True
            content = "\n".join(texts)
            if has_images:
                log.debug(f"WARNING: Image content in messages_to_prompt - images not supported")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts) + "\n\nAssistant:"


def openai_chunk(chunk_id: str, created: int, model: str, content: str, finish_reason: str | None = None, reasoning_content: str | None = None) -> str:
    delta: dict = {}
    if content:
        delta["content"] = content
        delta["role"] = "assistant"
    if reasoning_content:
        delta["reasoning_content"] = reasoning_content
        if "role" not in delta:
            delta["role"] = "assistant"
    return (
        f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': finish_reason}]})}\n\n"
    )


def openai_done() -> str:
    return "data: [DONE]\n\n"


def openai_tool_calls_chunk(chunk_id, created, model, tool_calls):
    """Generate OpenAI chunk with tool_calls in delta."""
    formatted_calls = []
    for i, tc in enumerate(tool_calls):
        formatted_calls.append({
            "index": i,
            "id": f"call_{hash(tc['name']) % 100000:05d}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"]
            }
        })
    return json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": formatted_calls},
            "logprobs": None,
            "finish_reason": "tool_calls"
        }]
    }) + "\n\n"


def openai_tool_calls_response(chunk_id, created, model, tool_calls):
    """Generate full OpenAI response with tool_calls."""
    formatted_calls = []
    for i, tc in enumerate(tool_calls):
        formatted_calls.append({
            "index": i,
            "id": f"call_{hash(tc['name']) % 100000:05d}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"]
            }
        })
    return json.dumps({
        "id": chunk_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": formatted_calls
            },
            "logprobs": None,
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })


def parse_tool_calls(text, available_tools=None):
    """Parse tool calls from LLM text output."""
    import re, json
    tool_calls = []
    available_names = set()
    if available_tools:
        for t in available_tools:
            if t.get("type") == "function":
                fn = t.get("function", {})
                name = fn.get("name", "")
                if name:
                    available_names.add(name)

    skip = {"thinking", "think", "tool_calls"}

    def _valid(name):
        if not name or name.lower() in skip:
            return False
        if available_names and name not in available_names:
            return False
        return True

    def _clean(s):
        return s.strip().strip(chr(34)).strip(chr(39)).strip(",")

    def _parse_props(txt):
        result = {}
        param_pat = '<parameter\\s+name="([^"]+)"[^>]*>(.*?)</parameter>'
        param_pat2 = '<param\\s+name="([^"]+)"[^>]*>(.*?)</param>'
        for m in re.finditer(param_pat, txt, re.DOTALL):
            result[m.group(1)] = _clean(m.group(2))
        if not result:
            for m in re.finditer(param_pat2, txt, re.DOTALL):
                result[m.group(1)] = _clean(m.group(2))
        if not result:
            try:
                c = txt.strip()
                if c.startswith("{"):
                    result = json.loads(c)
            except (json.JSONDecodeError, ValueError):
                pass
        if not result:
            attr_pat = r"(\w+)\s*=\s*" + chr(34) + '([^"]*)' + chr(34)
            for m in re.finditer(attr_pat, txt):
                result[m.group(1)] = m.group(2)
        return result

    # Format 1: invoke with parameter tags
    for m in re.finditer('<invoke name="([^"]+)">(.*?)</invoke>', text, re.DOTALL):
        name, props = m.group(1), m.group(2)
        if _valid(name):
            args = _parse_props(props)
            if args:
                tool_calls.append({"name": name, "arguments": json.dumps(args)})

    # Format 2: tool_calls wrapper with invoke
    if not tool_calls:
        tc = re.search('<tool_calls>(.*?)</tool_calls>', text, re.DOTALL)
        if tc:
            for m in re.finditer('<invoke name="([^"]+)">(.*?)</invoke>', tc.group(1), re.DOTALL):
                name, props = m.group(1), m.group(2)
                if _valid(name):
                    args = _parse_props(props)
                    if args:
                        tool_calls.append({"name": name, "arguments": json.dumps(args)})

    # Format 3: tool_calls with JSON
    if not tool_calls:
        tc = re.search('<tool_calls>(.*?)</tool_calls>', text, re.DOTALL)
        if tc:
            try:
                p = json.loads(tc.group(1).strip())
                if isinstance(p, dict) and "name" in p:
                    if _valid(p['name']):
                        args = p.get('arguments', {})
                        if isinstance(args, dict): args = json.dumps(args)
                        tool_calls.append({"name": p["name"], "arguments": args})
                elif isinstance(p, list):
                    for item in p:
                        if isinstance(item, dict) and "name" in item:
                            if _valid(item['name']):
                                args = item.get('arguments', {})
                                if isinstance(args, dict): args = json.dumps(args)
                                tool_calls.append({"name": item["name"], "arguments": args})
            except (json.JSONDecodeError, ValueError): pass

    # Format 4: self-closing XML in tool_calls
    if not tool_calls:
        tc = re.search('<tool_calls>(.*?)</tool_calls>', text, re.DOTALL)
        if tc:
            for m in re.finditer('<(\\w+)\\s+([^>]*?)/>>', tc.group(1)):
                name, attrs = m.group(1), m.group(2)
                if _valid(name):
                    args = {}
                    for am in re.finditer(r"(\w+)=" + chr(34) + '([^"]*)' + chr(34), attrs):
                        args[am.group(1)] = _clean(am.group(2))
                    if args:
                        tool_calls.append({"name": name, "arguments": json.dumps(args)})

    # Format 6: tag with JSON content
    if not tool_calls:
        for m in re.finditer('<(\\w+)>\\s*(\\{.*?\\})\\s*</\\1>', text, re.DOTALL):
            name = m.group(1)
            try:
                p = json.loads(m.group(2))
                if isinstance(p, dict) and _valid(name):
                    if "name" in p and "arguments" in p:
                        tool_calls.append(p)
                    else:
                        tool_calls.append({"name": name, "arguments": json.dumps(p)})
            except (json.JSONDecodeError, ValueError): pass

    # Format 8: colon-separated tag
    if not tool_calls:
        for m in re.finditer('<(\\w+):(\\w+)>(.*?)</\\1:\\2>', text):
            tool_name, param_name, value = m.group(1), m.group(2), _clean(m.group(3))
            if value and _valid(tool_name):
                tool_calls.append({"name": tool_name, "arguments": json.dumps({param_name: value})})

    # Format 9: Hermes-style <tool_call name="..."> with parameter tags
    if not tool_calls:
        for m in re.finditer('<tool_call\\s+name="([^"]+)"[^>]*>(.*?)</tool_call>', text, re.DOTALL):
            name, props = m.group(1), m.group(2)
            if _valid(name):
                args = _parse_props(props)
                if args:
                    tool_calls.append({"name": name, "arguments": json.dumps(args)})

    # Bare JSON fallback
    if not tool_calls and available_names:
        try:
            p = json.loads(text.strip())
            if isinstance(p, dict) and "name" in p:
                if _valid(p['name']):
                    args = p.get('arguments', {})
                    if isinstance(args, dict): args = json.dumps(args)
                    tool_calls.append({"name": p["name"], "arguments": args})
            elif isinstance(p, list):
                for item in p:
                    if isinstance(item, dict) and "name" in item:
                        if _valid(item['name']):
                            args = item.get('arguments', {})
                            if isinstance(args, dict): args = json.dumps(args)
                            tool_calls.append({"name": item["name"], "arguments": args})
        except (json.JSONDecodeError, ValueError): pass

    return tool_calls


def openai_full(chunk_id: str, created: int, model: str, content: str) -> str:
    return json.dumps({
        "id": chunk_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ─── Completion handler ────────────────────────────────────

async def handle_completion(body: dict, req_id: str) -> dict:
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = strip_prefix(body.get("model", "deepseek-chat"))
    tools = body.get("tools")

    rlog(req_id, f"=" * 60)
    rlog(req_id, f"→ HERMES → PROXY  model={body.get('model')} stream={stream} tools={len(tools) if tools else 0}")
    rlog(req_id, f"Raw body: {json.dumps(body, ensure_ascii=False)[:5000]}")

    model_lower = (model or "").lower()
    model_type = "default"
    if "reasoner" in model_lower or "r1" in model_lower:
        model_type = "expert"
    elif "vision" in model_lower:
        model_type = "vision"

    thinking_enabled = body.get("thinking_enabled", default_thinking)
    search_enabled = body.get("search_enabled", default_search)

    client = create_client()

    # ── Session reuse logic ──────────────────────────────────
    last_msg = messages[-1] if messages else {}
    last_content = ""
    c = last_msg.get("content")
    if isinstance(c, str):
        last_content = c
    elif isinstance(c, list):
        texts = []
        has_images = False
        for item in c:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") in ("image_url", "image"):
                has_images = True
        last_content = "\n".join(texts)

    session_id: str
    prompt: str
    parent_message_id: int | None = None

    # ── Detect tool result ───────────────────────────────────
    last_tool_call_id = None
    is_tool_result = False

    if last_msg.get("role") == "tool":
        last_tool_call_id = last_msg.get("tool_call_id")
        is_tool_result = True
        rlog(req_id, f"DETECT: role=tool → tool_result (tool_call_id={last_tool_call_id})")
    elif last_msg.get("role") == "user" and len(messages) >= 2:
        prev = messages[-2]
        if prev.get("role") == "assistant":
            prev_text = prev.get("content", "")
            if isinstance(prev_text, list):
                prev_text = "\n".join(item.get("text", "") for item in prev_text if item.get("type") == "text")
            if re.search(r'<tool_call\s+name=', prev_text) or re.search(r'<invoke\s+name=', prev_text):
                is_tool_result = True
                rlog(req_id, f"DETECT: prev assistant has tool_call XML → tool_result")

    # ── Session lookup / create ──────────────────────────────
    if last_msg.get("role") in ("user", "tool"):
        pkey = _prefix_key(messages)
        existing = _session_store.get(pkey)
        if existing:
            session_id, parent_message_id, had_tool_call = existing
            rlog(req_id, f"SESSION: REUSE {session_id} (parent={parent_message_id} had_tool_call={had_tool_call})")
        else:
            session_id = await client.create_session()
            had_tool_call = False
            rlog(req_id, f"SESSION: NEW {session_id} key={pkey}")
    else:
        session_id = await client.create_session()
        had_tool_call = False
        rlog(req_id, f"SESSION: NEW {session_id} (no user/tool role)")

    # ── Build prompt ─────────────────────────────────────────
    if is_tool_result:
        lt = chr(60)
        gt = chr(62)
        tc_id = last_tool_call_id or "unknown"
        prompt = f"User: {lt}tool_result id={chr(34)}{tc_id}{chr(34)}{gt}{last_content}{lt}/tool_result{gt}\n\nAssistant:"
        rlog(req_id, f"ACTION: wrap tool_result → DeepSeek")
        rlog(req_id, f"Tool result content ({len(last_content)} chars): {last_content[:500]}")
    elif last_msg.get("role") in ("user", "tool"):
        sys_parts = []
        for m in messages:
            if m.get("role") == "system":
                sc = m.get("content", "")
                if isinstance(sc, list):
                    sc = "\n".join(item.get("text", "") for item in sc if item.get("type") == "text")
                sys_parts.append("System: " + sc)
        sys_text = "\n\n".join(sys_parts) if sys_parts else ""
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            tool_descs = []
            for t in tools:
                func = t.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                params = func.get("parameters", {}).get("properties", {})
                param_names = list(params.keys())
                tool_descs.append(f"  - {name}: {desc} (params: {param_names})")
            tools_text = chr(10).join(tool_descs)
            names_str = ", ".join(tool_names)
            lt = chr(60)
            gt = chr(62)
            tc_open = lt + "tool_calls" + gt
            tc_close = lt + "/tool_calls" + gt
            inv_open = lt + "invoke" + gt
            inv_close = lt + "/invoke" + gt
            param_open = lt + "parameter" + gt
            param_close = lt + "/parameter" + gt
            tool_ctx = "System: You have access to the following tools. To call a tool, respond with:" + chr(10)
            tool_ctx += tc_open + chr(10)
            tool_ctx += "  " + inv_open + chr(32) + "name=" + chr(34) + "TOOL_NAME" + chr(34) + chr(32) + inv_close + chr(10)
            tool_ctx += "    " + param_open + chr(32) + "name=" + chr(34) + "PARAM_NAME" + chr(34) + chr(32) + param_close + " VALUE " + lt + "/parameter" + gt + chr(10)
            tool_ctx += "  " + lt + "/invoke" + gt + chr(10)
            tool_ctx += tc_close + chr(10)
            tool_ctx += "Available tools: " + names_str + chr(10)
            tool_ctx += tools_text + chr(10)
            tool_ctx += "Only call tools when the user explicitly asks. Otherwise respond normally." + chr(10)
        else:
            tool_ctx = ""
        prompt = (sys_text + "\n\n" + tool_ctx + "\n\nUser: " + last_content + "\n\nAssistant:").strip()
        rlog(req_id, f"ACTION: reuse session → build prompt with system+tools+user")
    else:
        prompt = messages_to_prompt(messages, tools)
        rlog(req_id, f"ACTION: new session → messages_to_prompt")

    # ── LOG: Outgoing to DeepSeek ────────────────────────────
    rlog(req_id, f"← PROXY → DEEPSEEK  session={session_id} parent={parent_message_id}")
    rlog(req_id, f"PROMPT ({len(prompt)} chars):\n{prompt}")

    # ── Streaming ────────────────────────────────────────────
    if stream:
        async def run_stream(on_chunk, on_done, on_error):
            session_cleaned = False
            try:
                chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
                created = int(time.time())
                on_chunk(openai_chunk(chunk_id, created, model, "", None))

                thinking_opened = False
                full_text = ""
                text_buf = ""          # text to send as content
                in_tool_call = False   # true once we detect a tool call starting
                tool_text_buf = ""     # accumulated tool call XML

                def on_thinking_chunk(text: str):
                    nonlocal thinking_opened
                    if not thinking_opened:
                        on_chunk(openai_chunk(chunk_id, created, model, "<think>", None))
                        thinking_opened = True
                    on_chunk(openai_chunk(chunk_id, created, model, "", None, reasoning_content=text))

                def on_text_chunk(text: str):
                    nonlocal thinking_opened, full_text, text_buf, in_tool_call, tool_text_buf
                    full_text += text
                    if thinking_opened:
                        on_chunk(openai_chunk(chunk_id, created, model, "</think>", None))
                        thinking_opened = False

                    if in_tool_call:
                        tool_text_buf += text
                        return

                    # Check if this chunk transitions into a tool call
                    m = re.search(r'<(?:invoke|tool_call)\s', text)
                    if m:
                        # Flush text before the tool call marker
                        before = _strip_tool_tags(text[:m.start()])
                        if before:
                            on_chunk(openai_chunk(chunk_id, created, model, before, None))
                        # Start buffering tool call XML
                        tool_text_buf = text[m.start():]
                        in_tool_call = True
                    else:
                        text_buf += text

                result = await client.complete(
                    session_id=session_id,
                    prompt=prompt,
                    model_type=model_type,
                    parent_message_id=parent_message_id,
                    thinking_enabled=thinking_enabled,
                    search_enabled=search_enabled,
                    req_id=req_id,
                    on_text=on_text_chunk,
                    on_thinking=on_thinking_chunk,
                )

                if thinking_opened:
                    on_chunk(openai_chunk(chunk_id, created, model, "</think>", None))

                # ── LOG: DeepSeek raw response
                rlog(req_id, f"DEEPSEEK RESPONSE ({len(full_text)} chars):\n{full_text}")

                # ── At the end: decide whether we have tool calls ──
                had_tool_call = False
                if in_tool_call:
                    # Parse tool calls from full_text (includes any lead-in text)
                    tool_calls = parse_tool_calls(full_text, tools)
                    if tool_calls:
                        had_tool_call = True
                        rlog(req_id, f"TOOL CALLS detected ({len(tool_calls)}): {json.dumps(tool_calls, ensure_ascii=False)}")
                        # Send tool_calls chunk (replaces the buffered XML)
                        on_chunk(openai_tool_calls_chunk(chunk_id, created, model, tool_calls))
                    else:
                        # Fallback: flush buffered tool text as filtered content
                        rlog(req_id, f"TOOL CALL PARSE FAILED — sending as filtered text")
                        remaining = _strip_tool_tags(tool_text_buf)
                        if remaining:
                            on_chunk(openai_chunk(chunk_id, created, model, remaining, None))
                else:
                    # No tool calls at all — flush remaining text_buf
                    remaining = _strip_tool_tags(text_buf)
                    if remaining:
                        on_chunk(openai_chunk(chunk_id, created, model, remaining, None))

                # ── Store session for reuse ──
                if result and result.get("lastAssistantMessageId"):
                    nkey = _hash_messages(_user_messages(messages))
                    pkey = _prefix_key(messages)
                    _session_store[nkey] = (session_id, result["lastAssistantMessageId"], had_tool_call)
                    _session_store[pkey] = (session_id, result["lastAssistantMessageId"], had_tool_call)
                    rlog(req_id, f"STORE session key={nkey} pkey={pkey} had_tool_call={had_tool_call}")

                finish_reason = "tool_calls" if had_tool_call else "stop"
                on_chunk(openai_chunk(chunk_id, created, model, "", finish_reason))

                rlog(req_id, f"→ PROXY → HERMES  finish_reason={finish_reason} had_tool_call={had_tool_call}")
                on_done()
            except Exception as e:
                rlog(req_id, f"STREAM ERROR: {e}")
                if not session_cleaned:
                    session_cleaned = True
                    for k, v in list(_session_store.items()):
                        if v[0] == session_id:
                            rlog(req_id, f"Removing broken session {session_id} key={k} from store")
                            del _session_store[k]
                on_error(e)

        return {"type": "stream", "run": run_stream}

    # ── Non-streaming ────────────────────────────────────────
    full_text = ""
    full_thinking = ""

    def on_text(text: str):
        nonlocal full_text
        full_text += text

    def on_thinking(text: str):
        nonlocal full_thinking
        full_thinking += text

    result = await client.complete(
        session_id=session_id,
        prompt=prompt,
        model_type=model_type,
        parent_message_id=parent_message_id,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
        req_id=req_id,
        on_text=on_text,
        on_thinking=on_thinking,
    )

    rlog(req_id, f"DEEPSEEK RESPONSE ({len(full_text)} chars):\n{full_text}")

    # Store session for reuse — key is hash of user messages including this turn
    had_tool_call = bool(parse_tool_calls(full_text, tools))
    if result and result.get("lastAssistantMessageId"):
        nkey = _hash_messages(_user_messages(messages))
        pkey = _prefix_key(messages)
        _session_store[nkey] = (session_id, result["lastAssistantMessageId"], had_tool_call)
        _session_store[pkey] = (session_id, result["lastAssistantMessageId"], had_tool_call)
        rlog(req_id, f"STORE session key={nkey} pkey={pkey} had_tool_call={had_tool_call}")

    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    tool_calls = parse_tool_calls(full_text, tools)

    if tool_calls:
        rlog(req_id, f"TOOL CALLS detected ({len(tool_calls)}): {json.dumps(tool_calls, ensure_ascii=False)}")
        response_body = json.loads(openai_tool_calls_response(chunk_id, created, model, tool_calls))
    else:
        response_body = json.loads(openai_full(chunk_id, created, model, full_text))

    if full_thinking:
        response_body["thinking"] = full_thinking
        response_body["choices"][0]["message"]["reasoning_content"] = full_thinking

    rlog(req_id, f"→ PROXY → HERMES  response ({len(json.dumps(response_body))} chars)")
    return {"type": "json", "body": json.dumps(response_body)}


# ─── HTTP routes ──────────────────────────────────────────

async def handle_options(request: web.Request) -> web.Response:
    return web.Response(status=204)


async def handle_models(request: web.Request) -> web.Response:
    now = int(time.time() * 1000)
    models = [
        {"id": f"{PREFIX}deepseek-chat", "object": "model", "created": now, "owned_by": "deepseek"},
        {"id": f"{PREFIX}deepseek-reasoner", "object": "model", "created": now, "owned_by": "deepseek"},
        {"id": f"{PREFIX}deepseek-vision", "object": "model", "created": now, "owned_by": "deepseek"},
    ]
    return web.json_response({"object": "list", "data": models})


async def handle_health(request: web.Request) -> web.Response:
    import os
    return web.json_response({
        "status": "ok",
        "auth_loaded": bool(auth["token"]),
        "port": request.app["port"],
        "deepseek_url": BASE_URL,
    })


async def handle_chat(request: web.Request) -> web.StreamResponse:
    req_id = _req_id()
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    messages = body.get("messages", [])

    try:
        result = await handle_completion(body, req_id)
    except AuthError as e:
        return web.json_response({"error": "auth_required", "message": str(e)}, status=401)
    except Exception as e:
        return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    if result["type"] == "json":
        rlog(req_id, f"Response: {result['body'][:2000]}")
        return web.Response(text=result["body"], content_type="application/json")

    # Streaming response
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    write_queue = asyncio.Queue()
    closed = False

    async def writer():
        nonlocal closed
        while True:
            data = await write_queue.get()
            if data is None:
                break
            try:
                await response.write(data)
            except Exception:
                closed = True
                break
        if not closed:
            try:
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except Exception:
                pass

    writer_task = asyncio.create_task(writer())

    def on_chunk(chunk: str):
        if not closed:
            write_queue.put_nowait(chunk.encode("utf-8"))

    def on_done():
        write_queue.put_nowait(None)

    def on_error(error: Exception):
        nonlocal closed
        print(f"[stream] {error}", file=sys.stderr)
        if not closed:
            err_data = json.dumps({"error": str(error)})
            write_queue.put_nowait(f"data: {err_data}\n\n".encode("utf-8"))
            write_queue.put_nowait(None)

    asyncio.create_task(result["run"](on_chunk, on_done, on_error))

    await writer_task
    return response


async def handle_not_found(request: web.Request) -> web.Response:
    return web.json_response(
        {"error": "not_found", "message": f"Path {request.path} not found"},
        status=404,
    )


# ─── CORS middleware ───────────────────────────────────────

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as ex:
            resp = ex
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


# ─── Main ─────────────────────────────────────────────────

async def run_server(port: int, host: str):
    app = web.Application(middlewares=[cors_middleware])
    app["port"] = port

    app.router.add_route("*", "/v1/models", handle_models)
    app.router.add_route("*", "/health", handle_health)
    app.router.add_route("*", "/", handle_health)
    app.router.add_route("*", "/v1/chat/completions", handle_chat)
    app.router.add_route("*", "/{path:.*}", handle_not_found)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    proxy_info = get_proxy_info()
    proxy_line = (
        f"SOCKS5:  {proxy_info['host']}:{proxy_info['port']}"
        + (" (auth)" if proxy_info["hasAuth"] else "")
        if proxy_info
        else "SOCKS5:  off"
    )

    try:
        print(f"""
╔══════════════════════════════════════════════════╗
║     DeepSeek Free -> OpenAI Proxy                ║
║══════════════════════════════════════════════════║
║  Port:    {str(port):<39}║
║  Host:    {host:<39}║
║  {proxy_line:<47}║
║══════════════════════════════════════════════════║
║  POST http://localhost:{port}/v1/chat/completions    ║
║  GET  http://localhost:{port}/v1/models               ║
║  GET  http://localhost:{port}/health                   ║
╚══════════════════════════════════════════════════╝
        """)
    except UnicodeEncodeError:
        safe_line = proxy_line.encode("ascii", "replace").decode()
        print(f"""
+----------------------------------------------+
|     DeepSeek Free -> OpenAI Proxy            |
+----------------------------------------------+
|  Port:    {str(port):<39}|
|  Host:    {host:<39}|
|  {safe_line:<47}|
+----------------------------------------------+
|  POST http://localhost:{port}/v1/chat/completions  |
|  GET  http://localhost:{port}/v1/models             |
|  GET  http://localhost:{port}/health                 |
+----------------------------------------------+
        """)

    try:
        await init_auth()
        print("\nСервер готов к работе!\n")
    except Exception as e:
        print(f"\nАвторизация не загружена: {e}")
        print("   Выполни: python server.py --login\n")

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runner.cleanup()


async def validate_and_login():
    saved = read_saved_auth()
    if saved:
        auth["cookieHeader"] = saved["cookieHeader"]
        auth["token"] = saved["token"]
        print("[auth] Загружена сохранённая авторизация, проверяю...")
        try:
            client = create_client()
            await client.create_session()
            print("[auth] Токен валиден")
            return
        except AuthError:
            print("[auth] Токен истёк, открываю окно логина...")
        except Exception as e:
            print(f"[auth] Ошибка проверки: {e}, открываю окно логина...")

    result = await login_and_save_auth()
    auth["cookieHeader"] = result["cookieHeader"]
    auth["token"] = result["token"]
    print("[auth] Авторизация получена")


def main():
    global default_thinking, default_search
    args = parse_args()

    _log_setup(DEBUG or args.debug)

    default_thinking = not args.no_thinking
    default_search = not args.no_search

    if args.proxy:
        os.environ["SOCKS5_PROXY"] = args.proxy

    if args.manual:
        print_manual_instructions()
        return

    if args.import_cookies:
        cookies_file, token_str = args.import_cookies
        try:
            import_cookies(cookies_file, token_str)
            print("Импорт готов. Запускай: python server.py")
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.connect is not None:
        try:
            asyncio.run(connect_to_running_chrome(args.connect))
            print("Подключение готово. Запускай: python server.py")
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.login:
        asyncio.run(init_auth(force_login=True))
        return

    port = args.port or int(__import__("os").environ.get("PORT", "18632"))
    host = args.host or __import__("os").environ.get("HOST", "0.0.0.0")
    asyncio.run(run_server(port, host))


if __name__ == "__main__":
    main()
