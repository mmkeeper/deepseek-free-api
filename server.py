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
import base64
import binascii
import hashlib
import json
import logging
import mimetypes
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEBUG = False  # Переключи в True для отладки или используй --debug

MAX_TOOL_RESULT_CHARS = 50000  # Truncate tool output to avoid DeepSeek prompt length limits

from aiohttp import web

from src.auth import (
    connect_to_running_chrome,
    import_cookies,
    login_and_save_auth,
    print_manual_instructions,
    read_saved_auth,
)
from src.client import AuthError, DeepSeekClient
from src.sse import DeepSeekError
from src.config import BASE_URL
from src.proxy import get_http_client, get_proxy_info


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


# ─── Attachment handling (единая модель: изображения и файлы) ───────────

# mimetypes.guess_extension() has surprising mappings for common types
# (e.g. application/xml → .xsl), override them here.
_MIME_EXT_OVERRIDES = {
    "application/xml": ".xml",
    "text/xml": ".xml",
    "application/octet-stream": ".bin",
}


def _guess_ext(mime: str, fallback: str) -> str:
    mime = mime.strip().lower()
    if mime in _MIME_EXT_OVERRIDES:
        return _MIME_EXT_OVERRIDES[mime]
    return mimetypes.guess_extension(mime) or fallback


def _image_ext(url: str, content_type: str = "") -> str:
    if url.startswith("data:"):
        mime = url[5:].split(";", 1)[0]
        return _guess_ext(mime, ".png")
    ext = _guess_ext(content_type.split(";")[0], "")
    if not ext:
        path = url.split("?")[0].rsplit(".", 1)[-1]
        ext = "." + path if path and "/" not in path else ".png"
    return ext


def _mime_ext(url: str, content_type: str = "") -> str:
    """Extension from a data: URL mime or http content-type (default .bin)."""
    if url.startswith("data:"):
        mime = url[5:].split(";", 1)[0]
        return _guess_ext(mime, ".bin")
    ext = _guess_ext(content_type.split(";")[0], "")
    if not ext:
        path = url.split("?")[0].rsplit(".", 1)[-1]
        ext = "." + path if path and "/" not in path else ".bin"
    return ext


def _sanitize_filename(name: str) -> str:
    """Return a safe basename: path components stripped, control chars removed."""
    if not name:
        return ""
    name = name.replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if 32 <= ord(ch) <= 126 or ord(ch) > 126)
    return name[:255]


def _attachment_filename(item: dict) -> str:
    """Best-effort client-provided filename for a file/input_file content part."""
    f = item.get("file") if item.get("type") == "file" else item.get("input_file")
    if isinstance(f, dict):
        return f.get("filename") or item.get("filename") or ""
    return item.get("filename") or ""


async def _extract_attachments(messages: list[dict]) -> list[tuple[str, bytes, bool]]:
    """Extract attachment parts from the last message: images and files.

    Supported content part shapes (OpenAI-compatible):
      * {"type": "image_url", "image_url": {"url": "data:...|http..."}}
      * {"type": "image", "url": "data:...|http..."}
      * {"type": "file", "file": {"filename": "a.xlsx", "file_data": "data:..."}}
      * {"type": "input_file", "filename": "a.xlsx", "file_data": "data:..."}
      * {"type": "file", "file": "data:..."}

    Returns (filename, data, is_image) triples. Images without an explicit
    name fall back to image<ext> so DeepSeek renders them as pictures; other
    files fall back to file<ext>.
    """
    if not messages:
        return []
    c = messages[-1].get("content")
    if not isinstance(c, list):
        return []
    attachments = []
    client = get_http_client()
    for item in c:
        itype = item.get("type")
        if itype in ("image_url", "image"):
            is_image = True
            url = item.get("image_url") or item.get("url") or ""
            if isinstance(url, dict):
                url = url.get("url") or ""
            filename = _sanitize_filename(item.get("filename") or "")
        elif itype in ("file", "input_file"):
            is_image = False
            f = item.get("file") if itype == "file" else item.get("input_file")
            if isinstance(f, str):
                f = {"file_data": f}
            f = f or {}
            if isinstance(f, dict):
                filename = _sanitize_filename(_attachment_filename(item))
                url = f.get("file_data") or f.get("data") or item.get("file_data") or item.get("data") or ""
            else:
                filename, url = "", ""
        else:
            continue

        if url.startswith("data:"):
            try:
                header, b64 = url.split(",", 1)
                data = base64.b64decode(b64)
            except (ValueError, binascii.Error):
                log.debug(f"Bad data URL attachment skipped")
                continue
            if not filename:
                m = re.search(r";name=([^;,]+)", header)
                if m:
                    filename = _sanitize_filename(m.group(1))
            if not filename:
                ext = _image_ext(url) if is_image else _mime_ext(url)
                filename = f"image{ext}" if is_image else f"file{ext}"
            attachments.append((filename, data, is_image))
        elif url.startswith("http://") or url.startswith("https://"):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.content
            except Exception as e:
                log.debug(f"Failed to fetch attachment {url[:80]}: {e}")
                continue
            if not filename:
                ext = _image_ext(url, resp.headers.get("content-type", "")) if is_image else _mime_ext(url, resp.headers.get("content-type", ""))
                filename = f"image{ext}" if is_image else f"file{ext}"
            attachments.append((filename, data, is_image))
    return attachments


async def _upload_attachments(client: DeepSeekClient, attachments: list[tuple[str, bytes, bool]],
                              model_type: str, thinking_enabled: bool,
                              req_id: str) -> tuple[list[str], str]:
    """Upload attachments, waiting for each to become ready. Returns (file_ids, used_type).

    The unified model may advertise upload limits under any of the legacy
    model_type keys, so fall back until one works.
    """
    candidates = [model_type]
    for alt in ("vision", "default", "expert"):
        if alt not in candidates:
            candidates.append(alt)
    used_type = model_type
    ref_file_ids = []
    for filename, data, _is_image in attachments:
        last_err = None
        for mt in candidates:
            try:
                fid = await client.upload_and_confirm(filename, data, model_type=mt,
                                                      thinking_enabled=thinking_enabled, req_id=req_id)
                ref_file_ids.append(fid)
                if used_type != mt:
                    used_type = mt
                rlog(req_id, f"ATTACHMENT UPLOADED & CONFIRMED: {filename} → {fid} (model_type={mt})")
                break
            except Exception as e:
                last_err = e
                rlog(req_id, f"ATTACHMENT UPLOAD try {mt} failed: {e}")
        else:
            rlog(req_id, f"ATTACHMENT UPLOAD FAILED: {filename}: {last_err}")
    return ref_file_ids, used_type


# ─── Session store (reuse DeepSeek sessions within one server run) ───

# session_id, parent_message_id, had_tool_call, tool_calls_cache
_session_store: dict[str, tuple[str, int | None, bool, list | None]] = {}


class RetryLaterError(Exception):
    """Raised when an exact retry arrives while the original request is still pending."""
    pass

# ─── Logging ────────────────────────────────────────────────

def _log_setup(enabled: bool, log_dir: str | None = None):
    """Configure logging — structured format with timestamps, writes to
    logs/ds_session.log (or a custom directory) + stderr when --debug is on."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if enabled else logging.CRITICAL)

    class UtcMicroFormatter(logging.Formatter):
        """Форматтер, поддерживающий %f (микросекунды) и выводящий время в UTC."""

        def formatTime(self, record, datefmt=None):
            # Создаём datetime в UTC из unix-времени записи
            dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
            if datefmt:
                return dt.strftime(datefmt)
            # Если datefmt не задан — используем стандартное поведение, но с микросекундами
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # Использование
    fmt = UtcMicroFormatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%f",  # теперь %f работает
    )


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
    log.debug(f"[REQ-{req_id}] {msg}", stacklevel=2)


# ─── XML tag stripping — keep content clean from tool markup ─

_TOOL_TAG_RE = re.compile(r'</?(?:tool_calls|tool_call|invoke|parameter|name|arguments)[^>]*>')


def _strip_tool_tags(text: str) -> str:
    """Вырезает tool-разметку вне код-блоков (``` и ~~~) и вне `инлайн-кода`.

    Примеры формата должны доходить до клиента нетронутыми.
    """
    if "`" not in text and "~" not in text:
        return _TOOL_TAG_RE.sub("", text)
    out = []
    for seg, is_code in _code_segments(text):
        if is_code:
            out.append(seg)  # внутри фенса — не трогаем
            continue
        out.append(_walk_outside(seg, lambda p: p, lambda p: _TOOL_TAG_RE.sub("", p)))
    return "".join(out)


# ─── Markdown fence defusing — examples in ``` blocks are not tool calls ─

# Обезвреживать разметку внутри код-блоков И `инлайн-спанов` при ПОИСКЕ
# вызовов: слово tool -> t00l, чтобы примеры формата и упоминания тегов
# в тексте не исполнялись и не останавливали стрим. Только для поиска —
# клиенту текст доставляется как есть, без подстановок.
# False — искать разметку везде как в обычном тексте (эксперимент).
MASK_CODE_FENCES = True

_FENCE_RUN_RE = re.compile(r"(?m)^[ \t]{0,3}(?:`{3,}|~{3,})")
_SPAN_TOKEN_RE = re.compile(r"(`{3,}|`)")
# Гравис-обёрнутые туловые теги в прозе: `<tool_calls>`, `</tool_call>`,
# `<parameter name="...">` и т.п. Обезвреживаются точечно, без состояния.
_SPAN_TOOL_TAG_RE = re.compile(r"`(?:</?(?:tool_calls?|invoke|parameter|name|arguments)\b[^`]*)`")


def _walk_outside(seg: str, span_fn, outside_fn) -> str:
    """Обход вне-кодового сегмента: серии из 3+ бэктиков дословно (это не
    спаны и не фенсы), одиночные — границы инлайн-спана. К содержимому спанов
    применяется span_fn, к остальному тексту — outside_fn."""
    m = _FENCE_RUN_RE.match(seg)
    prefix = ""
    if m:
        prefix = m.group(0)
        seg = seg[m.end():]
    out = [prefix]
    in_span = False
    pos = 0
    for t in _SPAN_TOKEN_RE.finditer(seg):
        piece = seg[pos:t.start()]
        out.append(span_fn(piece) if in_span else outside_fn(piece))
        token = t.group(0)
        out.append(token)
        if len(token) == 1:
            in_span = not in_span
        pos = t.end()
    tail = seg[pos:]
    out.append(span_fn(tail) if in_span else outside_fn(tail))
    return "".join(out)


def _blank_span(piece: str) -> str:
    """Забеливает содержимое инлайн-спана, сохраняя длину и переводы строк."""
    return "".join(c if c == "\n" else " " for c in piece)


def _code_segments(text: str) -> list[tuple[str, bool]]:
    """Разбивает текст на (сегмент, inside_code).

    Фенс — серия из 3+ бэктиков или тильд НА ОТДЕЛЬНОЙ СТРОКЕ (отступ до
    3 пробелов); внутристрочные серии вроде '```' в значениях параметров —
    обычный текст. Закрывающим считается фенс того же символа и не короче
    открывающего, поэтому вложенные блоки образуют один регион.
    """
    segments: list[tuple[str, bool]] = []
    pos = 0
    inside = False
    open_ch = ""
    open_len = 0
    pending = ""  # фенс-разделитель прикрепляется к следующему сегменту
    for m in _FENCE_RUN_RE.finditer(text):
        segments.append((pending + text[pos:m.start()], inside))
        run = m.group()
        pending = run
        body_len = len(run.lstrip(" \t"))
        if not inside:
            inside, open_ch, open_len = True, run.lstrip()[0], body_len
        elif run.lstrip()[0] == open_ch and body_len >= open_len:
            inside = False
        pos = m.end()
    segments.append((pending + text[pos:], inside))
    return segments


def _defuse_segment(seg: str, inside: bool) -> str:
    """Сегмент кода: tool->t00l; вне кода гравис-обёрнутые туловые теги
    заменяются пробелами (длина сохраняется) — без stateful-обхода спанов,
    чтобы нечётный бэктик не обезвреживал весь хвост ответа."""
    if inside:
        return seg.replace("tool", "t00l")
    m = _FENCE_RUN_RE.match(seg)
    prefix = ""
    if m:
        prefix = m.group(0)
        seg = seg[m.end():]
    return prefix + _SPAN_TOOL_TAG_RE.sub(lambda mm: "`" + " " * (len(mm.group(0)) - 2) + "`", seg)


def _mask_code_fences(text: str) -> str:
    """Внутри код-блоков и инлайн-спанов заменяет 'tool' на 't00l'.

    Длина сохраняется — смещения совпадают с оригиналом.
    """
    if "`" not in text and "~" not in text:
        return text
    return "".join(_defuse_segment(seg, code) for seg, code in _code_segments(text))

PREFIX = "dsf-"

_tool_call_counter = 0

def _next_tool_call_id(tool_name: str) -> str:
    global _tool_call_counter
    _tool_call_counter += 1
    return f"call_{abs(hash(tool_name)) % 10000:04d}{_tool_call_counter:04x}"

def strip_prefix(model: str) -> str:
    return model[len(PREFIX):] if model.startswith(PREFIX) else model

def _hash_messages(msgs: list[dict]) -> str:
    raw = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _user_messages(msgs: list[dict]) -> list[dict]:
    """Extract only user + system messages — stable across turns."""
    return [{"role": m["role"], "content": m.get("content", "")}
            for m in msgs if m.get("role") in ("user", "system")]


def _strip_tool_results(msgs: list[dict]) -> list[dict]:
    """Remove tool result messages (role=tool, or user preceded by assistant with tool_calls)."""
    result: list[dict] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "tool":
            i += 1
            continue
        if m.get("role") == "assistant":
            result.append(m)
            i += 1
            if m.get("tool_calls"):
                # Skip following user messages (tool results)
                while i < len(msgs) and msgs[i].get("role") in ("user", "tool"):
                    i += 1
            continue
        result.append(m)
        i += 1
    return result


def _prefix_key(messages: list[dict]) -> str:
    """Hash of user/system messages in prefix (all except last user turn).

    Tool result messages (both role=tool and user role preceded by assistant
    with tool_calls) are excluded so the key stays stable across retries.

    Returns empty string when the prefix has no real user message (i.e. a fresh
    conversation's first turn). A prefix with the first user message is a valid
    continuation key for the second turn — it equals the first turn's nkey, so
    a second message continues the same DeepSeek session instead of starting a
    new one.

    The system prompt alone is NOT a valid key: Hermes sends the same large
    system prompt for every session, so hash([system]) would collide across
    conversations and continue the wrong DeepSeek session.
    """
    prefix = messages[:-1] if len(messages) >= 1 else []
    stable = _strip_tool_results(prefix)
    umsgs = _user_messages(stable)
    if not any(m.get("role") == "user" for m in umsgs):
        return ""
    return _hash_messages(umsgs)


# ─── Turn frames (rollback / regenerate / continuation) ─────
# per session an ordered list of committed turns, used to resolve the
# DeepSeek parent for continue, regenerate and edited rollback requests:
#   {"key": nkey of the user/system messages incl. this turn,
#    "parent_id": response id of the previous frame (DeepSeek parent used),
#    "response_msg_id": DeepSeek id of this turn's assistant message,
#    "user_text": last user content (debug only)}
# Frames are append-only; _frame_for_key returns the LATEST frame for a key,
# so continuing after a regenerated reply points to the newest response.
_session_frames: dict[str, list[dict]] = {}


def _frame_for_key(session_id: str, key: str) -> dict | None:
    for f in reversed(_session_frames.get(session_id, [])):
        if f.get("key") == key:
            return f
    return None


def _record_turn(session_id: str, key: str, parent_id: int | None,
                 response_msg_id: int, user_text: str):
    _session_frames.setdefault(session_id, []).append({
        "key": key,
        "parent_id": parent_id,
        "response_msg_id": response_msg_id,
        "user_text": user_text,
    })


def _truncate_content(text: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    tail = f"\n\n[... truncated (originally {len(text)} chars) ...]"
    half = (max_chars - len(tail)) // 2
    if half <= 0:
        return text[:max_chars]
    return text[:half] + tail + text[-half:]

def _pretty_json(text: str) -> str:
    """Pretty-print JSON string if valid, otherwise return as-is."""
    import json as _json
    try:
        obj = _json.loads(text)
        return _json.dumps(obj, indent=2, ensure_ascii=False)
    except (_json.JSONDecodeError, TypeError):
        return text


def _tool_calls_to_xml(tool_calls: list | None) -> str:
    """Convert OpenAI tool_calls to the taught <tool_call> XML format."""
    if not tool_calls:
        return ""
    import json as _json
    lt = chr(60)
    gt = chr(62)
    dq = chr(34)
    lines = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "?")
        args_str = func.get("arguments", "{}")
        try:
            args = _json.loads(args_str) if isinstance(args_str, str) else args_str
        except (_json.JSONDecodeError, TypeError):
            args = {}
        lines.append(f"{lt}tool_call name={dq}{name}{dq}{gt}")
        for k, v in args.items():
            lines.append(f"  {lt}parameter name={dq}{k}{dq}{gt}{v}{lt}/parameter{gt}")
        lines.append(f"{lt}/tool_call{gt}")
    return "\n".join(lines)


# ─── OpenAI -> DeepSeek conversion ─────────────────────────

# Инструменты, для которых грузим ПОЛНУЮ схему параметров сразу.
FULL_SCHEMA_TOOLS = {
    "terminal",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "execute_code",
    "web_search",
    "web_extract",
    "memory",
    "process",
    "tool_describe",
}


def _format_full_schema(params: dict) -> str:
    """Полная JSON-схема параметров в компактном (однострочном) виде."""
    if not params:
        return "{}"
    return json.dumps(params, ensure_ascii=False, separators=(",", ":"))


# Сокращать описания отложенных тулов до первого абзаца (см. _first_paragraph).
# Сейчас отключено: выводим полные описания, как их передаёт клиент.
SHORTEN_DEFERRED_DESC = False


def _first_paragraph(desc: str) -> str:
    """Описание до первой пустой строки; многоточие, если сокращено."""
    desc = desc.replace("\r\n", "\n").lstrip()
    parts = desc.split("\n\n", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[0].rstrip() + "…"
    return parts[0].rstrip()


def _strip_assistant_preamble(content: str) -> str:
    """Убирает остаточный служебный префикс " thinking/response", попавший в
    текст ассистента (старые версии прокси эмитили его как content-чанки).
    Если размышления вырезаются — вместе с тегами."""
    return re.sub(r'^\s*thinking\s*response\s*', '', content, count=1)


def messages_to_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    parts = []
    if tools:
        lt, gt = chr(60), chr(62)
        dq = chr(34)
        tc_open = lt + "tool_calls" + gt
        tc_close = lt + "/tool_calls" + gt
        immediate_descs = []
        deferred_descs = []
        for t in tools:
            func = t.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            if name in FULL_SCHEMA_TOOLS:
                # Полный режим: имя + описание + полная схема параметров.
                schema = _format_full_schema(params)
                immediate_descs.append(f"  - {name}: {desc}\n    params: {schema}")
            else:
                # Отложенный режим: схема параметров не показана.
                d = _first_paragraph(desc) if SHORTEN_DEFERRED_DESC else desc
                deferred_descs.append(f"  - {name} [deferred]: {d}")
#        tool_header = "Не используй описанные ранее инструменты, если такие есть. Не используй правила их вызова для описанных далее инструментов" + chr(10)
#        tool_header += "Для вызова инструментов выводи вызов инструмента простым текстом так, как это описано далее." + chr(10)
        tool_header = "You have access to the following tools. To call a tool, respond with:" + chr(10)
        tool_header += tc_open + chr(10)
        tool_header += "  " + lt + "tool_call" + " name=" + dq + "TOOL_NAME" + dq + gt + chr(10)
        tool_header += "    " + lt + "parameter" + " name=" + dq + "PARAM_NAME" + dq + gt + "VALUE" + lt + "/parameter" + gt + chr(10)
        tool_header += "  " + lt + "/tool_call" + gt + chr(10)
        tool_header += tc_close + chr(10)
        if immediate_descs:
            tool_header += "# Available tools:" + chr(10)
            tool_header += chr(10).join(immediate_descs) + chr(10)
        if deferred_descs:
            tool_header += chr(10)
            tool_header += "# Deferred tool catalog (call schemas via `tool_describe`, invoke via `tool_call`):" + chr(10)
            tool_header += chr(10).join(deferred_descs) + chr(10)
        tool_header += "Only call tools when the user explicitly asks. Otherwise respond normally." + chr(10)
        tool_header += chr(10)
        parts.insert(0, tool_header)
    lt = chr(60)
    gt = chr(62)
    dq = chr(34)
    # Префиксы System:/User:/Assistant: нужны только для отделения сообщений
    # от системного промпта. Если системного промпта нет — отправляем чистый
    # текст, чтобы первое сообщение на сайте выглядело естественно.
    has_system = any(m.get("role") == "system" for m in messages)
    user_pfx = "User: " if has_system else ""
    assistant_pfx = "Assistant: " if has_system else ""
    for i, m in enumerate(messages):
        role = m.get("role", "")
        c = m.get("content")
        if isinstance(c, str):
            content = c
        elif isinstance(c, list):
            texts = []
            has_attachments = False
            for item in c:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif item.get("type") in ("image_url", "image"):
                    has_attachments = True
                    # Картинки/файлы уходят через ref_file_ids, в текст ставим маркер
                    texts.append("[изображение]")
                elif item.get("type") in ("file", "input_file"):
                    has_attachments = True
                    fname = _sanitize_filename(_attachment_filename(item))
                    texts.append(f"[файл: {fname}]" if fname else "[файл]")
            content = "\n".join(texts)
            if has_attachments:
                log.debug(f"File markers inserted in messages_to_prompt; files go via ref_file_ids")
        else:
            content = ""
        content = _truncate_content(_pretty_json(content))
        if role == "tool":
            tc_id = m.get("tool_call_id", "unknown")
            parts.append(f"{user_pfx}{lt}tool_result id={dq}{tc_id}{dq}{gt}\n{content}\n{lt}/tool_result{gt}")
        elif role == "assistant":
            tc_xml = _tool_calls_to_xml(m.get("tool_calls"))
            if content and tc_xml:
                content = content + "\n" + tc_xml
            elif tc_xml:
                content = tc_xml
            content = _strip_assistant_preamble(content)
            parts.append(f"{assistant_pfx}{content}")
        elif role == "system":
            parts.append(f"System: {content}")
        elif role == "user" and i > 0:
            # Check if this user message follows an assistant with tool_calls
            prev = messages[i - 1]
            if prev.get("role") == "assistant" and prev.get("tool_calls"):
                tc_id = m.get("tool_call_id", "unknown")
                parts.append(f"{user_pfx}{lt}tool_result id={dq}{tc_id}{dq}{gt}\n{content}\n{lt}/tool_result{gt}")
            else:
                parts.append(f"{user_pfx}{content}")
        else:
            parts.append(f"{user_pfx}{content}")
    if has_system:
        return "\n\n".join(parts) + "\n\nAssistant:"
    return "\n\n".join(parts)


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
            "id": _next_tool_call_id(tc['name']),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"]
            }
        })
    payload = json.dumps({
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "tool_calls": formatted_calls},
            "logprobs": None,
            "finish_reason": None
        }]
    })
    return f"data: {payload}\n\n"


def openai_tool_calls_response(chunk_id, created, model, tool_calls):
    """Generate full OpenAI response with tool_calls."""
    formatted_calls = []
    for i, tc in enumerate(tool_calls):
        formatted_calls.append({
            "index": i,
            "id": _next_tool_call_id(tc['name']),
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
                "content": None,
                "tool_calls": formatted_calls
            },
            "logprobs": None,
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })


def parse_tool_calls(text, available_tools=None):
    """Parse tool calls from LLM text output.

    Разметка внутри ```-блоков обезвреживается (tool -> t00l): примеры
    формата в код-блоках не являются реальными вызовами.
    """
    import re, json
    if MASK_CODE_FENCES:
        text = _mask_code_fences(text)
    tool_calls = []
    available_names = set()
    if available_tools:
        for t in available_tools:
            if t.get("type") == "function":
                fn = t.get("function", {})
                name = fn.get("name", "")
                if name:
                    available_names.add(name)

    skip = {"thinking", "think", "tool_calls", "arguments", "name"}

    def _valid(name):
        if not name or name.lower() in skip:
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

    # Format 9: Hermes-style <tool_call name="..."> with parameter tags.
    # Пустые аргументы валидны: явное имя тула — уже вызов (skills_list и т.п.)
    for m in re.finditer('<tool_call\\s+name="([^"]+)"[^>]*>(.*?)</tool_call>', text, re.DOTALL):
        name, props = m.group(1), m.group(2)
        if _valid(name):
            args = _parse_props(props) or {}
            tool_calls.append({"name": name, "arguments": json.dumps(args)})

    # Format 10: nested elements <tool_call><name>X</name><arguments><p>v</p></arguments></tool_call>
    for m in re.finditer(r'<tool_call>\s*<name>([^<]+)</name>(.*?)</tool_call>', text, re.DOTALL):
        name, body = m.group(1), m.group(2)
        if _valid(name):
            am = re.search(r'<arguments>(.*?)</arguments>', body, re.DOTALL)
            inner = am.group(1) if am else body
            args = {}
            for pm in re.finditer(r'<([^/>\s]+)>(.*?)</\1>', inner, re.DOTALL):
                args[pm.group(1)] = _clean(pm.group(2))
            if not args:
                # Модель может сымитировать JSON-схему из промпта
                try:
                    c = inner.strip()
                    if c.startswith("{"):
                        args = json.loads(c)
                except (json.JSONDecodeError, ValueError):
                    pass
            tool_calls.append({"name": name, "arguments": json.dumps(args)})

    # Format 11: <tool_calls> wrapper with direct tool-name tags
    # <tool_calls><tool_name><param>value</param></tool_name></tool_calls>
    if not tool_calls:
        tc = re.search(r'<tool_calls>(.*?)</tool_calls>', text, re.DOTALL)
        if tc:
            for m in re.finditer(r'<([^/>\s]+)>(.*?)</\1>', tc.group(1), re.DOTALL):
                name, body = m.group(1), m.group(2)
                if not _valid(name):
                    continue
                args = {}
                for pm in re.finditer(r'<([^/>\s]+)>(.*?)</\1>', body, re.DOTALL):
                    args[pm.group(1)] = _clean(pm.group(2))
                if not args:
                    try:
                        c = body.strip()
                        if c.startswith("{"):
                            args = json.loads(c)
                    except (json.JSONDecodeError, ValueError):
                        pass
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


# ─── Cached tool call response builder ────────────────────

def _build_cached_tool_call_response(chunk_id: str, created: int, model: str, tool_calls: list, req_id: str) -> dict:
    """Build a streaming response that replays cached tool calls without calling DeepSeek."""
    async def run_stream(on_chunk, on_done, on_error):
        try:
            # Assistant role signal
            payload = json.dumps({
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"},
                              "logprobs": None, "finish_reason": None}]
            })
            chunk_str = f"data: {payload}\n\n"
            rlog(req_id, f"CACHED CHUNK: role=assistant  size={len(chunk_str)}")
            on_chunk(chunk_str)
            # All tool calls in one chunk with correct indices
            formatted_calls = []
            for i, tc in enumerate(tool_calls):
                formatted_calls.append({
                    "index": i,
                    "id": _next_tool_call_id(tc['name']),
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })
            payload = json.dumps({
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0,
                              "delta": {"role": "assistant", "tool_calls": formatted_calls},
                              "logprobs": None, "finish_reason": None}]
            })
            chunk_str = f"data: {payload}\n\n"
            names = [tc['name'] for tc in tool_calls]
            rlog(req_id, f"CACHED CHUNK: tool_calls({len(tool_calls)}) names={names} size={len(chunk_str)}\n{chunk_str.rstrip()}")
            on_chunk(chunk_str)
            # Final chunk
            payload = json.dumps({
                "id": chunk_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {},
                              "logprobs": None, "finish_reason": "tool_calls"}]
            })
            chunk_str = f"data: {payload}\n\n"
            rlog(req_id, f"CACHED CHUNK: finish_reason=tool_calls  size={len(chunk_str)}")
            on_chunk(chunk_str)
            on_done()
        except Exception as e:
            rlog(req_id, f"CACHED STREAM ERROR: {e}")
            on_error(e)
    return {"type": "stream", "run": run_stream}


# ─── Completion handler ────────────────────────────────────

async def handle_completion(body: dict, req_id: str) -> dict:
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = strip_prefix(body.get("model", "deepseek-flash"))
    tools = body.get("tools")

    rlog(req_id, f"=" * 60)
    rlog(req_id, f"→ HERMES → PROXY  model={body.get('model')} stream={stream} tools={len(tools) if tools else 0}")
    rlog(req_id, f"Raw body: {json.dumps(body, ensure_ascii=False)}")

    model_lower = (model or "").lower()
    # Единая модель deepseek-flash: режим (быстрый/expert/vision) больше не
    # выбирается именем модели. Имена legacy-псевдонимов сохраняются для
    # обратной совместимости, но тип берётся из контента запроса.
    model_type = "default"
    if "reasoner" in model_lower or "r1" in model_lower:
        model_type = "expert"
    elif "vision" in model_lower:
        model_type = "vision"

    # reasoning_effort от Hermes: none/minimal/low/medium/high/xhigh/max/ultra/show/hide
    # Перебивается флагом --no-thinking. В единой модели глубокое мышление
    # включено по умолчанию — выключается только явным "none"/"hide" или
    # thinking_enabled=false.
    if not default_thinking:
        thinking_enabled = False
    else:
        reasoning_effort = (body.get("reasoning_effort") or "").lower()
        if reasoning_effort in ("none", "off", "hide", "minimal"):
            thinking_enabled = False
        else:
            thinking_enabled = body.get("thinking_enabled", True)
    search_enabled = body.get("search_enabled", default_search)

    client = create_client()

    # ── Анализ вложений: единая модель сама определяет наличие картинок/файлов ──
    attachments = await _extract_attachments(messages)
    ref_file_ids: list[str] = []
    if any(is_image for _, _, is_image in attachments):
        model_type = "vision"
        rlog(req_id, f"IMAGES DETECTED: {len(attachments)} — switching model_type to vision")
    elif attachments:
        rlog(req_id, f"FILES DETECTED: {len(attachments)} — {[f for f, _, _ in attachments]}")

    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    # ── Session reuse logic ──────────────────────────────────
    last_msg = messages[-1] if messages else {}
    rlog(req_id, f"MESSAGES ({len(messages)}): roles={[m.get('role') for m in messages]}")
    rlog(req_id, f"LAST_MSG role={last_msg.get('role')} content_type={type(last_msg.get('content')).__name__} has_tool_calls={bool(last_msg.get('tool_calls'))}")
    if len(messages) >= 2:
        prev = messages[-2]
        rlog(req_id, f"PREV_MSG role={prev.get('role')} content_type={type(prev.get('content')).__name__} has_tool_calls={bool(prev.get('tool_calls'))}")
    last_content = ""
    c = last_msg.get("content")
    if isinstance(c, str):
        last_content = c
    elif isinstance(c, list):
        texts = []
        for item in c:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") in ("image_url", "image"):
                texts.append("[изображение]")
            elif item.get("type") in ("file", "input_file"):
                fname = _sanitize_filename(_attachment_filename(item))
                texts.append(f"[файл: {fname}]" if fname else "[файл]")
        last_content = "\n".join(texts)

    session_id: str
    prompt: str
    parent_message_id: int | None = None

    # ── Detect tool result ───────────────────────────────────
    last_tool_call_id = None
    is_tool_result = False
    _tool_results_acc = []  # (tool_call_id, content) for all consecutive tool msgs

    if last_msg.get("role") == "tool":
        # Collect ALL consecutive tool messages from the end
        j = len(messages) - 1
        while j >= 0 and messages[j].get("role") == "tool":
            msg = messages[j]
            tc_id = msg.get("tool_call_id") or "unknown"
            c = msg.get("content", "")
            if isinstance(c, list):
                texts = [item.get("text", "") for item in c if item.get("type") == "text"]
                c = "\n".join(texts)
            _tool_results_acc.append((tc_id, c))
            j -= 1
        _tool_results_acc.reverse()  # chronological order
        last_tool_call_id = _tool_results_acc[-1][0]  # last tool's id (for reuse key)
        is_tool_result = True
        rlog(req_id, f"DETECT: role=tool → tool_result ({len(_tool_results_acc)} messages, ids={[t[0] for t in _tool_results_acc]})")
    elif last_msg.get("role") == "user":
        # Scan back through consecutive user messages to find preceding assistant
        j = len(messages) - 2
        while j >= 0 and messages[j].get("role") == "user":
            j -= 1
        if j >= 0 and messages[j].get("role") == "assistant":
            prev = messages[j]
            prev_text = prev.get("content") or ""
            if isinstance(prev_text, list):
                prev_text = "\n".join(item.get("text", "") for item in prev_text if item.get("type") == "text")
            _masked_prev = _mask_code_fences(prev_text) if MASK_CODE_FENCES else prev_text
            if re.search(r'<tool_call\s+name=', _masked_prev) or re.search(r'<invoke\s+name=', _masked_prev) or re.search(r'<tool_call>\s*<name>', _masked_prev):
                is_tool_result = True
                rlog(req_id, f"DETECT: prev assistant (via scan) has tool_call XML → tool_result")
            elif prev.get("tool_calls"):
                is_tool_result = True
                if not last_tool_call_id:
                    tc0 = prev["tool_calls"][0]
                    last_tool_call_id = tc0.get("id")
                rlog(req_id, f"DETECT: prev assistant (via scan) has tool_calls field → tool_result")

    # ── Determine if this is a session continuation ──────────
    umsgs = _user_messages(messages)
    nkey = _hash_messages(umsgs)
    is_continuation = False
    is_rollback = False

    # ── Session lookup / create ──────────────────────────────
    if last_msg.get("role") in ("user", "tool"):
        rlog(req_id, f"COMPARE nkey={nkey} from {len(umsgs)} user/system msgs: roles={[m['role'] for m in umsgs]} contents={[m['content'] for m in umsgs]}")
        existing = _session_store.get(nkey)
        if existing:
            rlog(req_id, f"STORE HIT nkey={nkey} → session={existing[0]} parent_id={existing[1]} had_tc={existing[2] if len(existing)>=3 else '?'} cached={existing[3] if len(existing)>=4 else '?'}")

        # Tool call retry check — exact match on ALL user/system messages
        if existing and not is_tool_result:
            sid, pid, had_tool_call, cached_tool_calls = existing if len(existing) == 4 else (*existing, None)
            if had_tool_call and cached_tool_calls:
                rlog(req_id, f"TOOL CALL REUSE — returning cached tool call (exact message match)")
                return _build_cached_tool_call_response(chunk_id, created, model, cached_tool_calls, req_id)
            if pid is None:
                # Original request still pending — don't send duplicate to DeepSeek
                rlog(req_id, f"ORIGINAL PENDING — retry for {sid} (wait for stream to complete)")
                raise RetryLaterError()

        # Tool result — reuse session from nkey (same user msgs = same conversation)
        if existing and is_tool_result:
            session_id, parent_message_id, _, _ = existing if len(existing) == 4 else (*existing, None)
            rlog(req_id, f"TOOL RESULT — continue session {session_id} parent={parent_message_id}")
        elif existing:
            # Identical user/system messages already answered → regenerate from the
            # parent the original turn used (frames keep it, unlike _session_store).
            frame_r = _frame_for_key(existing[0], nkey)
            if frame_r is not None and frame_r.get("parent_id") is not None:
                session_id = existing[0]
                parent_message_id = frame_r["parent_id"]
                is_rollback = True
                rlog(req_id, f"REGENERATE nkey={nkey} → session {session_id} parent={parent_message_id}")
            else:
                # First-turn regenerate (no historical parent) → fresh session.
                existing = None
        if not existing:
            # Session continuation check — prefix match
            pkey = _prefix_key(messages)
            rlog(req_id, f"COMPARE pkey={pkey or '(empty)'} (prefix of {len(messages)-1} user/system msgs)")
            existing = _session_store.get(pkey) if pkey else None
            if existing:
                session_id = existing[0]
                # Parent = response of the frame matching this prefix. Frames are
                # append-only, so a rollback/truncation keeps the OLD continuation
                # point instead of following the clobbered latest parent.
                frame_p = _frame_for_key(session_id, pkey)
                parent_message_id = frame_p["response_msg_id"] if frame_p else existing[1]
                is_continuation = True
                rlog(req_id, f"CONTINUE via pkey={pkey} → session {session_id} parent={parent_message_id}"
                     f"{'' if frame_p else ' (fallback to store)'}")
            else:
                session_id = await client.create_session()
                parent_message_id = None
                _session_store[nkey] = (session_id, parent_message_id, False, None)
                if pkey:
                    _session_store[pkey] = (session_id, parent_message_id, False, None)
                rlog(req_id, f"STORE MISS → SESSION: NEW {session_id} nkey={nkey} pkey={pkey or '(empty)'}")
    else:
        pkey = _prefix_key(messages)
        rlog(req_id, f"COMPARE pkey={pkey or '(empty)'} (prefix of {len(messages)-1} user/system msgs, last=assistant)")
        existing = _session_store.get(pkey) if pkey else None
        if existing:
            session_id = existing[0]
            frame_p = _frame_for_key(session_id, pkey)
            parent_message_id = frame_p["response_msg_id"] if frame_p else existing[1]
            is_continuation = True
            rlog(req_id, f"CONTINUE via pkey={pkey} → session {session_id} parent={parent_message_id}"
                 f"{'' if frame_p else ' (fallback to store)'}")
        else:
            session_id = await client.create_session()
            parent_message_id = None
            _session_store[nkey] = (session_id, parent_message_id, False, None)
            if pkey:
                _session_store[pkey] = (session_id, parent_message_id, False, None)
            rlog(req_id, f"SESSION: NEW {session_id} (no user/tool role) nkey={nkey} pkey={pkey or '(empty)'}")

    # ── Build prompt ─────────────────────────────────────────
    if is_tool_result:
        if _tool_results_acc:
            # Multiple consecutive tool messages
            parts = []
            total_orig = 0
            total_display = 0
            for tc_id, tc_content in _tool_results_acc:
                formatted = _truncate_content(_pretty_json(tc_content))
                parts.append(f"<tool_result id=\"{tc_id}\">\n{formatted}\n</tool_result>")
                total_orig += len(tc_content)
                total_display += len(formatted)
            prompt = "\n".join(parts)
            rlog(req_id, f"ACTION: wrap tool_result → DeepSeek ({len(_tool_results_acc)} messages, {total_orig} chars → {total_display} chars{' — TRUNCATED' if total_display < total_orig else ''})")
            for i, (tc_id, tc_content) in enumerate(_tool_results_acc):
                rlog(req_id, f"  Tool #{i+1}: id={tc_id} content_len={len(tc_content)} preview={tc_content[:100]}")
        else:
            # Fallback: single tool result (e.g. detected via user scan)
            tc_id = last_tool_call_id or "unknown"
            content = _truncate_content(_pretty_json(last_content))
            prompt = f"<tool_result id=\"{tc_id}\">\n{content}\n</tool_result>"
            orig_len = len(last_content)
            display_len = len(content)
            rlog(req_id, f"ACTION: wrap tool_result (fallback) → DeepSeek id={tc_id}")
            rlog(req_id, f"Tool result content ({orig_len} chars → {display_len} chars{' — TRUNCATED' if display_len < orig_len else ''}): {content[:500]}")
    elif is_rollback:
        prompt = last_content
        rlog(req_id, f"ACTION: regenerate (rollback) → raw message (no User: prefix)")
    elif is_continuation:
        prompt = last_content
        rlog(req_id, f"ACTION: continue session → raw message (no User: prefix)")
    else:
        # New session — include full conversation history
        prompt = messages_to_prompt(messages, tools)
        rlog(req_id, f"ACTION: new session ({'tool_result' if is_tool_result else 'full history'}) → messages_to_prompt")

    # ── LOG: Outgoing to DeepSeek ────────────────────────────
    rlog(req_id, f"← PROXY → DEEPSEEK  session={session_id} parent={parent_message_id}")
    rlog(req_id, f"PROMPT ({len(prompt)} chars):\n{prompt}")

    # ── Upload attached images/files after session resolution ──────
    if attachments:
        ref_file_ids, used_type = await _upload_attachments(client, attachments, model_type, thinking_enabled, req_id)
        if ref_file_ids:
            model_type = used_type
            rlog(req_id, f"REF_FILE_IDS: {ref_file_ids} model_type={model_type}")

    # ── Streaming ────────────────────────────────────────────
    if stream:
        # Shared with handle_chat: current DeepSeek message_id so the proxy can
        # call stop_stream when the client disconnects mid-generation.
        stream_state = {"message_id": None}

        async def run_stream(on_chunk, on_done, on_error):
            session_cleaned = False
            try:
                chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
                created = int(time.time())
                payload = json.dumps({
                    "id": chunk_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"},
                                  "logprobs": None, "finish_reason": None}]
                })
                on_chunk(f"data: {payload}\n\n")

                full_text = ""
                think_text = ""        # accumulated thinking content
                text_buf = ""          # text to send as content
                in_tool_call = False   # true once we detect a tool call starting
                tool_text_buf = ""     # accumulated tool call XML

                def on_thinking_chunk(text: str):
                    nonlocal think_text
                    think_text += text
                    # Reasoning уходит только через reasoning_content. Литеральные
                    # маркеры в content ("  thinking"/" response") не шлём — они
                    # собирались клиентом в текст ассистента и попадали в саммари.
                    on_chunk(openai_chunk(chunk_id, created, model, "", None, reasoning_content=text))

                def on_text_chunk(text: str):
                    nonlocal full_text, text_buf, in_tool_call, tool_text_buf
                    full_text += text

                    if in_tool_call:
                        tool_text_buf += text
                        return

                    # Check accumulated context for cross-chunk tool call detection.
                    # Поиск ведётся по обезвреженной копии (длина совпадает с
                    # оригиналом, смещения валидны): разметка внутри фенсов
                    # невидима и не останавливает стрим.
                    context = text_buf + text
                    ctx = _mask_code_fences(context) if MASK_CODE_FENCES else context
                    m = re.search(r'<(?:invoke|tool_calls?)[\s>]', ctx)
                    if m:
                        tool_start = m.start()
                        before = _strip_tool_tags(context[:tool_start])
                        if before:
                            on_chunk(openai_chunk(chunk_id, created, model, before, None))
                        tool_text_buf = context[tool_start:]
                        in_tool_call = True
                    else:
                        text_buf += text

                def on_message_id_chunk(mid: int):
                    stream_state["message_id"] = mid

                result = await client.complete(
                    session_id=session_id,
                    prompt=prompt,
                    model_type=model_type,
                    parent_message_id=parent_message_id,
                    thinking_enabled=thinking_enabled,
                    search_enabled=search_enabled,
                    ref_file_ids=ref_file_ids,
                    req_id=req_id,
                    on_text=on_text_chunk,
                    on_thinking=on_thinking_chunk,
                    on_message_id=on_message_id_chunk,
                )

                # ── LOG: DeepSeek raw response
                rlog(req_id, f"DEEPSEEK RESPONSE ({len(full_text)} chars text + {len(think_text)} chars think):\n{full_text}")

                # ── At the end: always parse full_text for tool calls ──
                tool_calls = parse_tool_calls(full_text, tools)
                had_tool_call = bool(tool_calls)

                if had_tool_call:
                    rlog(req_id, f"TOOL CALLS detected ({len(tool_calls)}): {json.dumps(tool_calls, ensure_ascii=False)}")
                    # If mid-stream didn't fire, send text before first tool call now
                    if not in_tool_call:
                        masked_full = _mask_code_fences(full_text) if MASK_CODE_FENCES else full_text
                        m = re.search(r'<(?:invoke|tool_call|tool_calls)[\s>]', masked_full)
                        if m:
                            before = _strip_tool_tags(full_text[:m.start()])
                            if before:
                                chunk_str = openai_chunk(chunk_id, created, model, before, None)
                                rlog(req_id, f"STREAM CHUNK: text_before_tool ({len(before)} chars)  size={len(chunk_str)}")
                                on_chunk(chunk_str)
                    # All tool calls in one chunk with correct indices
                    chunk_str = openai_tool_calls_chunk(chunk_id, created, model, tool_calls)
                    names = [tc['name'] for tc in tool_calls]
                    rlog(req_id, f"STREAM CHUNK: tool_calls({len(tool_calls)}) names={names} size={len(chunk_str)}\n{chunk_str.rstrip()}")
                    on_chunk(chunk_str)
                elif in_tool_call:
                    # Mid-stream detected tool call but parsing failed
                    rlog(req_id, f"TOOL CALL PARSE FAILED — sending as filtered text")
                    remaining = _strip_tool_tags(tool_text_buf)
                    rlog(req_id, f"STREAM CHUNK: filtered_text raw={len(tool_text_buf)} sent={len(remaining)}\n{remaining[:1500]}")
                    if remaining:
                        on_chunk(openai_chunk(chunk_id, created, model, remaining, None))
                else:
                    # No tool calls — flush all buffered text
                    remaining = _strip_tool_tags(text_buf)
                    rlog(req_id, f"STREAM CHUNK: flush_text raw={len(text_buf)} sent={len(remaining)}\n{remaining[:1500]}")
                    if remaining:
                        on_chunk(openai_chunk(chunk_id, created, model, remaining, None))

                # ── Store session for reuse ──
                cached_tc = tool_calls if had_tool_call else None
                if result and result.get("lastAssistantMessageId"):
                    nkey = _hash_messages(_user_messages(messages))
                    pkey = _prefix_key(messages)
                    _session_store[nkey] = (session_id, result["lastAssistantMessageId"], had_tool_call, cached_tc)
                    if pkey:
                        _session_store[pkey] = (session_id, result["lastAssistantMessageId"], had_tool_call, cached_tc)
                    _record_turn(session_id, nkey, parent_message_id,
                                 result["lastAssistantMessageId"], last_content)
                    rlog(req_id, f"STORE session key={nkey} pkey={pkey or '(empty)'} had_tool_call={had_tool_call}")

                finish_reason = "tool_calls" if had_tool_call else "stop"
                chunk_str = openai_chunk(chunk_id, created, model, "", finish_reason)
                rlog(req_id, f"STREAM CHUNK: finish_reason={finish_reason}  size={len(chunk_str)}")
                on_chunk(chunk_str)

                tc_log = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else "[]"
                rlog(req_id, f"→ PROXY → HERMES  text={len(full_text)}chars think={len(think_text)}chars tool_calls={len(tool_calls)} finish={finish_reason}")
                rlog(req_id, f"→ PROXY → HERMES  text_content (первые 2000 из {len(full_text)}):\n{full_text[:2000]}")
                rlog(req_id, f"→ PROXY → HERMES  think_content (первые 2000 из {len(think_text)}):\n{think_text[:2000]}")
                rlog(req_id, f"→ PROXY → HERMES  tool_calls_content: {tc_log[:2000]}")
                on_done()
            except asyncio.CancelledError:
                rlog(req_id, "STREAM CANCELLED — client disconnected")
                if not session_cleaned:
                    session_cleaned = True
                    for k, v in list(_session_store.items()):
                        if v[0] == session_id and v[1] is None:
                            rlog(req_id, f"Removing pending session {session_id} key={k} from store")
                            del _session_store[k]
                raise
            except DeepSeekError as e:
                rlog(req_id, f"DEEPSEEK ERROR: {e} finish_reason={e.finish_reason}")
                on_error(e)
            except Exception as e:
                rlog(req_id, f"STREAM ERROR: {e}")
                if not session_cleaned:
                    session_cleaned = True
                    for k, v in list(_session_store.items()):
                        if v[0] == session_id:
                            rlog(req_id, f"Removing broken session {session_id} key={k} from store")
                            del _session_store[k]
                on_error(e)

        return {"type": "stream", "run": run_stream,
                "client": client, "session_id": session_id,
                "state": stream_state}

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
        ref_file_ids=ref_file_ids,
        req_id=req_id,
        on_text=on_text,
        on_thinking=on_thinking,
    )

    rlog(req_id, f"DEEPSEEK RESPONSE ({len(full_text)} chars):\n{full_text}")

    # Store session for reuse — key is hash of user messages including this turn
    tool_calls = parse_tool_calls(full_text, tools)
    had_tool_call = bool(tool_calls)
    if result and result.get("lastAssistantMessageId"):
        nkey = _hash_messages(_user_messages(messages))
        pkey = _prefix_key(messages)
        cached_tc = tool_calls if had_tool_call else None
        _session_store[nkey] = (session_id, result["lastAssistantMessageId"], had_tool_call, cached_tc)
        if pkey:
            _session_store[pkey] = (session_id, result["lastAssistantMessageId"], had_tool_call, cached_tc)
        _record_turn(session_id, nkey, parent_message_id,
                     result["lastAssistantMessageId"], last_content)
        rlog(req_id, f"STORE session key={nkey} pkey={pkey or '(empty)'} had_tool_call={had_tool_call}")

    chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

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
        {"id": f"{PREFIX}deepseek-flash", "object": "model", "created": now, "owned_by": "deepseek"},
        # Обратная совместимость: старые имена перенаправляются на ту же модель
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
    except RetryLaterError:
        return web.json_response({"error": "busy", "message": "Request already in progress"}, status=429)
    except Exception as e:
        return web.json_response({"error": "internal_error", "message": str(e)}, status=500)

    if result["type"] == "json":
        rlog(req_id, f"Response: {result['body']}")
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

    def on_chunk(chunk: str):
        if not closed:
            if '"finish_reason"' in chunk or '"tool_calls"' in chunk:
                rlog(req_id, f"on_chunk ({chunk.count(chr(10))-1} lines, {len(chunk)} bytes): {chunk.rstrip()}")
            write_queue.put_nowait(chunk.encode("utf-8"))

    def on_done():
        write_queue.put_nowait(None)

    def on_error(error: Exception):
        nonlocal closed
        try:
            print(f"[stream] {error}", file=sys.stderr)
        except OSError:
            pass
        if not closed:
            err_data = json.dumps({"error": str(error)})
            write_queue.put_nowait(f"data: {err_data}\n\n".encode("utf-8"))
            write_queue.put_nowait(None)

    writer_task = asyncio.create_task(writer())
    stream_task = asyncio.create_task(result["run"](on_chunk, on_done, on_error))

    await writer_task

    if closed:
        # Client disconnected mid-stream — stop upstream generation so DeepSeek
        # does not commit a response the client never received.
        stop_client = result.get("client")
        if stop_client:
            session_id = result.get("session_id")
            message_id = (result.get("state") or {}).get("message_id")
            rlog(req_id, f"CLIENT DISCONNECT — stop_stream session={session_id} message_id={message_id}")
            await stop_client.stop_stream(session_id, message_id)
        if not stream_task.done():
            stream_task.cancel()
        try:
            await stream_task
        except (asyncio.CancelledError, Exception):
            pass

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
    app = web.Application(middlewares=[cors_middleware], client_max_size=0)
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
║  {proxy_line:<48}║
║══════════════════════════════════════════════════║
║  POST http://localhost:{port}/v1/chat/completions ║
║  GET  http://localhost:{port}/v1/models           ║
║  GET  http://localhost:{port}/health              ║
╚══════════════════════════════════════════════════╝
        """)
    except UnicodeEncodeError:
        safe_line = proxy_line.encode("ascii", "replace").decode()
        print(f"""
+--------------------------------------------------+
|     DeepSeek Free -> OpenAI Proxy                |
+--------------------------------------------------+
|  Port:    {str(port):<39}|
|  Host:    {host:<39}|
|  {safe_line:<48}|
+--------------------------------------------------+
|  POST http://localhost:{port}/v1/chat/completions |
|  GET  http://localhost:{port}/v1/models           |
|  GET  http://localhost:{port}/health              |
+--------------------------------------------------+
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
