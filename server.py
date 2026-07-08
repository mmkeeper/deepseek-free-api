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
import sys
import time
from pathlib import Path

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
  python server.py --proxy socks5://user:pass@10.0.0.1:1080
""",
    )
    p.add_argument("--port", type=int, default=None, help="Listen port (default: 18632)")
    p.add_argument("--host", default=None, help="Listen host (default: 0.0.0.0)")
    p.add_argument("--proxy", default=None, help="SOCKS5 proxy (socks5://host:port)")
    p.add_argument("--api-key", default=None, help="API key for client auth")
    p.add_argument("--login", action="store_true", help="Логин через Playwright")
    p.add_argument("--connect", nargs="?", const=9222, type=int, metavar="PORT",
                   help="Подключиться к Chrome через CDP")
    p.add_argument("--import", nargs=2, metavar=("COOKIES", "TOKEN"), dest="import_cookies",
                   help="Импорт cookies.json + userToken")
    p.add_argument("--manual", action="store_true",
                   help="Показать инструкцию по ручному экспорту")
    return p.parse_args()


# ─── Auth state ───────────────────────────────────────────

auth = {"cookieHeader": "", "token": ""}


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
        debug=False,
    )


# ─── Session store (reuse DeepSeek sessions within one server run) ───

_session_store: dict[str, tuple[str, int | None]] = {}

_log_file = None


def _log(msg: str):
    global _log_file
    if _log_file is None:
        import tempfile, os
        _log_file = open(os.path.join(tempfile.gettempdir(), "ds_session.log"), "a", encoding="utf-8")
    _log_file.write(msg + "\n")
    _log_file.flush()


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

def messages_to_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = "Assistant" if m.get("role") == "assistant" else "User"
        content = ""
        c = m.get("content")
        if isinstance(c, str):
            content = c
        elif isinstance(c, list):
            content = "\n".join(
                item["text"] for item in c if item.get("type") == "text"
            )
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts) + "\n\nAssistant:"


def openai_chunk(chunk_id: str, created: int, model: str, content: str, finish_reason: str | None = None) -> str:
    delta = {"content": content, "role": "assistant"} if content else {}
    return (
        f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': finish_reason}]})}\n\n"
    )


def openai_done() -> str:
    return "data: [DONE]\n\n"


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

async def handle_completion(body: dict) -> dict:
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", "deepseek-chat")

    model_lower = (model or "").lower()
    model_type = None
    if "reasoner" in model_lower or "r1" in model_lower:
        model_type = "expert"

    client = create_client()

    # ── Session reuse logic ──────────────────────────────────
    # Hash all messages except the last to identify the conversation prefix.
    # If we've seen this prefix before, reuse the session and send only the
    # new user message with parent_message_id.
    last_msg = messages[-1] if messages else {}
    last_content = ""
    c = last_msg.get("content")
    if isinstance(c, str):
        last_content = c
    elif isinstance(c, list):
        last_content = "\n".join(
            item["text"] for item in c if item.get("type") == "text"
        )

    session_id: str
    prompt: str
    parent_message_id: int | None = None

    if last_msg.get("role") == "user":
        pkey = _prefix_key(messages)
        _log(f"lookup key={pkey} store_keys={list(_session_store.keys())}")
        existing = _session_store.get(pkey)
        if existing:
            session_id, parent_message_id = existing
            _log(f"REUSE session {session_id} (parent={parent_message_id})")
        else:
            session_id = await client.create_session()
            _log(f"NEW session {session_id}")
        prompt = last_content
    else:
        session_id = await client.create_session()
        prompt = last_content

    # ── Streaming ────────────────────────────────────────────
    if stream:
        async def run_stream(on_chunk, on_done, on_error):
            try:
                chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
                created = int(time.time())
                on_chunk(openai_chunk(chunk_id, created, model, "", None))

                result = await client.complete(
                    session_id=session_id,
                    prompt=prompt,
                    model_type=model_type,
                    parent_message_id=parent_message_id,
                    thinking_enabled=False,
                    search_enabled=False,
                    on_text=lambda text: on_chunk(openai_chunk(chunk_id, created, model, text, None)),
                )

                # Store session for reuse — key is hash of user messages including this turn
                if result and result.get("lastAssistantMessageId"):
                    nkey = _hash_messages(_user_messages(messages))
                    _session_store[nkey] = (session_id, result["lastAssistantMessageId"])
                    _log(f"STORE key={nkey} session={session_id}")

                on_chunk(openai_chunk(chunk_id, created, model, "", "stop"))
                on_chunk(openai_done())
                on_done()
            except Exception as e:
                on_error(e)

        return {"type": "stream", "run": run_stream}

    # ── Non-streaming ────────────────────────────────────────
    full_text = ""

    def on_text(text: str):
        nonlocal full_text
        full_text += text

    result = await client.complete(
        session_id=session_id,
        prompt=prompt,
        model_type=model_type,
        parent_message_id=parent_message_id,
        thinking_enabled=False,
        search_enabled=False,
        on_text=on_text,
    )

    # Store session for reuse — key is hash of user messages including this turn
    if result and result.get("lastAssistantMessageId"):
        nkey = _hash_messages(_user_messages(messages))
        _session_store[nkey] = (session_id, result["lastAssistantMessageId"])
        _log(f"STORE key={nkey} session={session_id}")

    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    return {"type": "json", "body": openai_full(chunk_id, created, model, full_text)}


# ─── HTTP routes ──────────────────────────────────────────

async def handle_options(request: web.Request) -> web.Response:
    return web.Response(status=204)


async def handle_models(request: web.Request) -> web.Response:
    now = int(time.time() * 1000)
    models = [
        {"id": "deepseek-chat", "object": "model", "created": now, "owned_by": "deepseek"},
        {"id": "deepseek-reasoner", "object": "model", "created": now, "owned_by": "deepseek"},
        {"id": "deepseek-r1", "object": "model", "created": now, "owned_by": "deepseek"},
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
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid_json"}, status=400)

    messages = body.get("messages", [])

    try:
        result = await handle_completion(body)
    except AuthError as e:
        return web.json_response({"error": "auth_required", "message": str(e)}, status=401)
    except Exception as e:
        return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    if result["type"] == "json":
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

    print(f"""
╔══════════════════════════════════════════════════╗
║     DeepSeek Free -> OpenAI Proxy                ║
║══════════════════════════════════════════════════║
║  Порт:    {str(port):<39}║
║  Хост:    {host:<39}║
║  {proxy_line:<47}║
║══════════════════════════════════════════════════║
║  POST http://localhost:{port}/v1/chat/completions    ║
║  GET  http://localhost:{port}/v1/models               ║
║  GET  http://localhost:{port}/health                   ║
╚══════════════════════════════════════════════════╝
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
    args = parse_args()

    if args.proxy:
        import os
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
