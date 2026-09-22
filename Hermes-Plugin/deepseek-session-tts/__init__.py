"""TTS через локальный DeepSeek-прокси: озвучка сообщения сессии по номеру.

Провайдер не синтезирует присланный Hermes текст: он находит по нему сессию и
номер ответа в сохранённых списках сообщений, отправляет список этой сессии в
прокси, а прокси восстанавливает сессию и озвучивает ответ с этим номером.

Снапшоты хранятся ПО СЕССИЯМ. Один общий список означал бы, что озвучка,
пришедшая из только что открытой сессии, найдёт текст в снапшоте предыдущей и
озвучит чужой ответ — ровно этот дефект и наблюдался при переключении сессий.
Не нашли текст ни в одной известной сессии — молчим с ошибкой: озвучить чужое
сообщение хуже, чем не озвучить.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:18632"
_DEFAULT_VOICE = "mira"
_DEFAULT_TIMEOUT = 300.0
# Сколько сессий держим в памяти. Озвучивают почти всегда свежую переписку,
# поэтому старые вытесняются: иначе процесс рос бы вместе с числом чатов.
_MAX_SESSIONS = 8

_lock = threading.Lock()
# session_id -> {"messages": [...], "history": [...], "touched": monotonic}
_sessions: Dict[str, Dict[str, Any]] = {}


def _touch(session_id: str, key: str, value: Any) -> None:
    """Сохраняет список *value* под ключом *key* для сессии *session_id*.

    Пустой *session_id* — запись безадресна: приписать её некому, а считать
    «текущей» нечего. Пропускаем, чтобы не свалить список в чужую сессию,
    но громко: это означает, что хук пришёл без ключа сессии и озвучка этой
    переписки работать не будет.
    """
    if not session_id:
        logger.warning(
            "deepseek-session: %s пришёл без session_id — запись отброшена, "
            "озвучка этой сессии работать не будет", key)
        return None
    with _lock:
        entry = _sessions.get(session_id)
        is_new = entry is None
        if entry is None:
            entry = {"messages": [], "history": [], "touched": 0.0}
            _sessions[session_id] = entry
        entry[key] = value
        entry["touched"] = time.monotonic()
        count = len(_sessions)
        evicted = None
        if count > _MAX_SESSIONS:
            oldest = min(_sessions, key=lambda name: _sessions[name]["touched"])
            if oldest != session_id:
                _sessions.pop(oldest, None)
                evicted = oldest
    if is_new:
        logger.info("deepseek-session: в памяти новая сессия %r (всего %d): %s",
                    session_id, count, sorted(_sessions))
    if evicted:
        logger.info("deepseek-session: вытеснена старая сессия %r", evicted)
    return None


def _on_llm_request(request: Optional[Dict[str, Any]] = None, session_id: str = "",
                    **kwargs: Any) -> None:
    """Запоминает ИСХОДЯЩИЙ список сообщений — канонический источник правды.

    Прокси записывает сессию (STORE) ровно по содержимому списка, который
    уходит в /v1/chat/completions. request.messages — это и есть тот список.
    История из post_llm_call нормализуется Hermes иначе (см. _on_post_llm_call),
    поэтому подменять ей этот список нельзя: ключ сессии не совпадёт.
    """
    if not isinstance(request, dict):
        return None
    messages = request.get("messages")
    if isinstance(messages, list) and messages:
        _touch(str(session_id or ""), "messages", messages)
    return None


def _on_post_llm_call(conversation_history: Any = None, session_id: str = "",
                      **kwargs: Any) -> None:
    """Сохраняет историю с готовым ответом как справочник принадлежности.

    НЕ заменяет канонический список из llm_request: Hermes отдаёт в пост-колл
    историю в своём формате, и ключ сессии, посчитанный прокси по такому
    списку, не совпадёт с ключом, под которым сессия реально записана (STORE).
    История нужна для одного: по ней видно, ЧЬЯ это реплика. Свежий ответ в
    снапшот messages попасть не может по построению (список уходит до
    генерации), поэтому найденный только здесь текст означает «эта сессия»,
    а номер ответа в payload не передаётся — прокси озвучит её свежий ответ.
    """
    if isinstance(conversation_history, list) and conversation_history:
        _touch(str(session_id or ""), "history", list(conversation_history))
    return None


# --- Порт sanitizeTextForSpeech (apps/desktop/src/lib/speech-text.ts) ---
#
# Desktop прогоняет текст ответа через СВОЙ очиститель перед отправкой на
# /api/audio/speak (lib/voice-playback.ts:669 -> sanitizeTextForSpeech). Он
# снимает markdown-таблицы ЦЕЛИКОМ, снимает заголовки и эмфазис, а перевод
# строки без завершающего знака препинания превращает в '.'.
#
# Снапшот же (llm_request / history) хранит СЫРОЙ markdown. Hermes затем
# нормализует пришедший текст ещё раз (tools/tts_tool.py:431), поэтому
# сравнение «сырое против пришедшего» расходится ровно на ответах с таблицей
# или заголовком: замер на 3876 живых ответах — 1691 промах (2185 совпало).
# С этой очисткой на обеих сторонах — 3877 из 3877, расхождений нет.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\U000E0020-\U000E007F]+")
_FENCED_RE = re.compile(r"```[\s\S]*?(?:```|$)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_PARA_BREAK_RE = re.compile(r"[ \t]*\n{2,}[ \t]*")
_PUNCT_PARA_BREAK_RE = re.compile(
    r"([.!?])([*_~`>\"\'\u2019\u201d)}\]]*)[ \t]*\n{2,}[ \t]*")
_SOFT_BREAK_RE = re.compile(r"[ \t]*\n[ \t]*")
_URL_RE = re.compile(r"\bhttps?://\S+", re.I)
_TABLE_DELIM_RE = re.compile(r"^:?-{3,}:?$")


def _md_row(line):
    """Разбирает строку markdown-таблицы -> (глубина цитаты, ячейки) или None."""
    row, depth = line, 0
    while True:
        indent = re.match(r"^[ \t]*", row).group(0)
        if "\t" in indent or len(indent) > 3:
            return None
        row = row[len(indent):]
        if not row.startswith(">"):
            break
        depth += 1
        row = row[1:]
        if row.startswith(" "):
            row = row[1:]
    row = row.rstrip()

    def unescaped(i):
        back, cur = 0, i - 1
        while cur >= 0 and row[cur] == "\\":
            back += 1
            cur -= 1
        return back % 2 == 0

    pipes = [i for i, ch in enumerate(row) if ch == "|" and unescaped(i)]
    if not pipes:
        return None
    lead, trail = pipes[0] == 0, pipes[-1] == len(row) - 1
    if lead:
        row = row[1:]
    if trail:
        row = row[:-1]
    cells, start = [], 0
    for i, ch in enumerate(row):
        if ch == "|" and unescaped(i):
            cells.append(row[start:i].strip())
            start = i + 1
    cells.append(row[start:].strip())
    if len(cells) < 2 and not (lead and trail and len(cells) == 1):
        return None
    return depth, cells


def _strip_md_tables(text):
    lines = re.sub(r"\r\n?", "\n", text).split("\n")
    drop, i = set(), 1
    while i <len(lines):
        delim, head = _md_row(lines[i]), _md_row(lines[i - 1])
        if (not delim or not head
                or not all(_TABLE_DELIM_RE.match(c) for c in delim[1])
                or len(head[1]) != len(delim[1]) or head[0] != delim[0]):
            i += 1
            continue
        drop.add(i - 1)
        drop.add(i)
        j = i + 1
        while j <len(lines):
            body = _md_row(lines[j])
            if not body or body[0] != delim[0]:
                break
            drop.add(j)
            j += 1
        i = j
    return "\n".join(l for k, l in enumerate(lines) if k not in drop)


def _sanitize_for_speech(text: str) -> str:
    """Очистка текста так, как её делает desktop перед отправкой на озвучку."""
    if not text:
        return ""
    t = _strip_md_tables(text)
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = _PUNCT_PARA_BREAK_RE.sub(r"\1\2 ", t)
    t = _PARA_BREAK_RE.sub(". ", t)
    t = _SOFT_BREAK_RE.sub(" ", t)
    t = _FENCED_RE.sub(" code block omitted ", t)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _INLINE_CODE_RE.sub(r"\1", t)
    t = _URL_RE.sub(" link ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.M)
    t = re.sub(r"[*_~>#]", "", t)
    t = re.sub(r"^\s*[-+*]\s+", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


# Канонический вид текста — неподвижная точка нормализатора Hermes.
# Кэш по сырому тексту: один и тот же ответ озвучивают повторно (кнопка у
# сообщения, авто-озвучка), а стабилизация длинных текстов не бесплатна.
_SPOKEN_CACHE: Dict[str, str] = {}
_SPOKEN_CACHE_MAX = 512
_STABLE_PASSES = 24


def _spoken(text: str) -> str:
    """Канонический вид текста: неподвижная точка prepare_spoken_text.

    Hermes прогоняет текст через prepare_spoken_text ПЕРЕД тем, как отдать его
    провайдеру (tools/tts_tool.py:431), поэтому провайдер получает уже
    нормализованный текст. Но нормализатор НЕ идемпотентен: на реальных ответах
    (многоточия, «..», markdown-огрызки, сноски) второй проход даёт другой
    результат — 55 расхождений из 240 замеров на живых сообщениях. Поэтому
    «один проход от сырого» и «один проход от уже нормализованного» — разные
    строки, и текст, лежащий в снапшоте, не находился пришедшим.

    Обе стороны приводим к неподвижной точке: canon(x) == canon(sp(x)), так что
    сырое содержимое снапшота и нормализованный вход дают один и тот же ключ.
    Сходимость быстрая (медиана 1-2 прохода, длинные ответы до ~10), циклов не
    наблюдается. При недостижении предела возвращаем последний результат —
    сравнение тогда может не совпасть, но чужой текст не озвучится.
    """
    if not text:
        return ""
    key = _sanitize_for_speech(str(text))
    hit = _SPOKEN_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        from tools.tts_text_normalize import prepare_spoken_text
    except Exception:
        result = key.strip()
    else:
        current = key
        for _ in range(_STABLE_PASSES):
            nxt = prepare_spoken_text(current, max_chars=None)
            nxt = " ".join((nxt or "").split())
            if nxt == current:
                break
            current = nxt
        result = current
    if len(_SPOKEN_CACHE) >= _SPOKEN_CACHE_MAX:
        _SPOKEN_CACHE.clear()
    _SPOKEN_CACHE[key] = result
    return result


def _find_message_index(messages: List[Dict[str, Any]], spoken: str) -> Optional[int]:
    """0-based номер assistant-ответа, чей нормализованный текст равен *spoken*.

    Идём с конца: озвучивают почти всегда свежие сообщения, поэтому при
    дубликатах выигрывает ближайшее к концу. None — совпадения нет.
    """
    if not spoken:
        return None
    index = sum(1 for m in messages if m.get("role") == "assistant") - 1
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        if _spoken(str(message.get("content") or "")) == spoken:
            return index
        index -= 1
    return None


def _search(session_id: str, key: str, spoken: str) -> Optional[int]:
    """Ищет *spoken* в списке *key* сессии *session_id*; None — не нашлось."""
    with _lock:
        entry = _sessions.get(session_id) or {}
        messages = list(entry.get(key) or [])
    return _find_message_index(messages, spoken)


def _session_order() -> List[str]:
    """Известные сессии, от самой свежей к самой старой."""
    with _lock:
        return sorted(_sessions, key=lambda name: _sessions[name]["touched"], reverse=True)


def _history_facts(session_id: str, spoken: str) -> Tuple[bool, bool]:
    """(нашлось ли в истории, последняя ли это реплика ассистента).

    Различать эти два случая обязательно. ``message_index=None`` в payload —
    просьба «озвучь САМЫЙ СВЕЖИЙ ответ сессии», поэтому её нельзя слать для
    любой найденной в истории реплики: старое сообщение озвучилось бы как
    последний ответ — вместо запрошенного звучал бы чужой текст.

    Не-последняя найденная реплика означает одно из двух, и в обоих случаях
    озвучить её нельзя: она выпала из канонического списка при компрессии
    контекста (прокси считает сессию по отправленному списку — изменившийся
    список даёт другой ключ, и такой реплики прокси не знает), либо это
    середина переписки, до которой исходящий список вообще не доходит.
    """
    with _lock:
        entry = _sessions.get(session_id) or {}
        history = list(entry.get("history") or [])
    assistants = [m for m in history if m.get("role") == "assistant"]
    if not assistants:
        return False, False
    index = _find_message_index(history, spoken)
    if index is None:
        return False, False
    return True, index == len(assistants) - 1


def _counts(session_id: str) -> Tuple[int, int]:
    """(assistant в каноническом списке, assistant в истории) — для лога."""
    with _lock:
        entry = _sessions.get(session_id) or {}
        messages = list(entry.get("messages") or [])
        history = list(entry.get("history") or [])
    return (sum(1 for m in messages if m.get("role") == "assistant"),
            sum(1 for m in history if m.get("role") == "assistant"))


def _locate(spoken: str) -> Optional[Tuple[str, Optional[int]]]:
    """Ищет *spoken* по всем сессиям и возвращает (session_id, message_index).

    Три исхода:
      1. Точное совпадение в каноническом списке — индекс ответа известен.
      2. Совпадение с ПОСЛЕДНЕЙ репликой в истории — ответ сгенерирован, но в
         исходящий список попасть не успел по построению; None вместо индекса
         просит прокси озвучить свежий ответ именно этой сессии.
      3. Не найдено (или найдено только в середине истории — такое сообщение
         прокси не знает: канонический список до него не доходит) — None
         целиком, озвучивать нечего.
    """
    if not spoken:
        return None
    order = _session_order()
    for session_id in order:
        index = _search(session_id, "messages", spoken)
        if index is not None:
            return session_id, index
    for session_id in order:
        found, is_last = _history_facts(session_id, spoken)
        if found and is_last:
            return session_id, None
    return None


def _messages_of(session_id: str) -> List[Dict[str, Any]]:
    with _lock:
        entry = _sessions.get(session_id) or {}
        return list(entry.get("messages") or [])


def _load_config() -> Dict[str, Any]:
    """Секция tts.deepseek-session из config.yaml ({} при недоступности)."""
    try:
        from hermes_cli.config import load_config
        tts_config = (load_config() or {}).get("tts") or {}
    except Exception:
        return {}
    section = tts_config.get("deepseek-session")
    return section if isinstance(section, dict) else {}


def _dump_diagnostics(payload: Dict[str, Any], code: int, detail: str) -> None:
    """Сохраняет тело TTS-запроса на ошибке прокси — для отладки ключей сессии."""
    try:
        scratch = os.environ.get("TMPDIR") or tempfile.gettempdir()
        path = os.path.join(scratch, "deepseek_tts_debug.json")
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": code,
            "detail": detail,
            "payload": payload,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _sniff_container(data: bytes) -> str:
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"RIFF":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and data[1] >= 0xE0:
        return "mp3"
    return ""


def _ffmpeg_convert(source: str, target: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", source, target],
        capture_output=True, stdin=subprocess.DEVNULL)
    return result.returncode == 0 and os.path.isfile(target) and os.path.getsize(target) > 0


class DeepSeekSessionTTS(TTSProvider):
    """Озвучка ответа из сессии DeepSeek по номеру сообщения."""

    @property
    def name(self) -> str:
        return "deepseek-session"

    @property
    def voice_compatible(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def list_voices(self) -> List[Dict[str, Any]]:
        return [
            {"id": "mira", "display": "Mira", "gender": "female"},
            {"id": "echo", "display": "Echo", "gender": "male"},
            {"id": "stella", "display": "Stella", "gender": "female"},
            {"id": "tide", "display": "Tide", "gender": "male"},
        ]

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None,
                   model: Optional[str] = None, speed: Optional[float] = None,
                   format: str = "mp3", **extra: Any) -> str:
        """*text* — ключ поиска: по нему находится сессия и номер ответа в ней."""
        config = _load_config()
        base_url = str(config.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
        payload: Dict[str, Any] = {
            "voice": voice or config.get("voice") or _DEFAULT_VOICE,
        }
        # Явный override из конфига бьёт поиск по тексту; сессию берём свежую.
        explicit = config.get("message_index")
        if isinstance(explicit, int) and not isinstance(explicit, bool):
            order = _session_order()
            if not order:
                raise RuntimeError(
                    "deepseek-session: список сообщений ещё не захвачен (llm_request не срабатывал)")
            payload["messages"] = _messages_of(order[0])
            payload["message_index"] = explicit
            logger.info("deepseek-session: message_index=%s из конфига, сессия %s",
                        explicit, order[0])
        else:
            spoken_text = _spoken(text)
            located = _locate(spoken_text)
            if located is None:
                # Никакого «озвучь последний ответ»: он принадлежит не той
                # сессии, из которой пришёл запрос, — именно так проигрывался
                # фрагмент предыдущего чата при переключении.
                order = _session_order()
                for name in order:
                    found, is_last = _history_facts(name, spoken_text)
                    if found and not is_last:
                        in_msgs, in_hist = _counts(name)
                        logger.warning(
                            "deepseek-session: реплика найдена в истории сессии %s, но не является "
                            "последним ответом (assistant: %d в списке, %d в истории) — прокси её "
                            "не знает; озвучка отменена, чтобы не прочитать чужое",
                            name, in_msgs, in_hist)
                        raise RuntimeError(
                            "deepseek-session: это сообщение не является последним ответом сессии "
                            "(выпало из отправляемой истории при компрессии контекста) — "
                            "прокси его не знает и озвучить не может")
                for name in order:
                    with _lock:
                        entry = _sessions.get(name) or {}
                        msgs = list(entry.get("messages") or [])
                        hist = list(entry.get("history") or [])
                    tails = []
                    for label, seq in (("msg", msgs), ("hist", hist)):
                        assistants = [x for x in seq if x.get("role") == "assistant"]
                        for off, item in enumerate(assistants[-4:]):
                            raw = str(item.get("content") or "")
                            tails.append("%s[%d/%d]=%r" % (
                                label, len(assistants) - 4 + off, len(assistants),
                                _spoken(raw)[:70]))
                    logger.warning(
                        "deepseek-session: ПРОМАХ. запрошено=%r (len %d). Хвосты сессии %s: %s",
                        spoken_text[:90], len(spoken_text), name, " | ".join(tails))
                logger.warning(
                    "deepseek-session: текст не найден ни в одной из %d известных сессий %s — "
                    "озвучка отменена, чтобы не прочитать чужой ответ",
                    len(order), order)
                raise RuntimeError(
                    "deepseek-session: озвучиваемое сообщение не найдено ни в одной сессии — "
                    "текст не совпал ни с одним ответом ассистента")
            session_id, message_index = located
            payload["messages"] = _messages_of(session_id)
            if message_index is not None:
                payload["message_index"] = message_index
            in_msgs, in_hist = _counts(session_id)
            logger.info(
                "deepseek-session: сессия %s, message_index=%s, assistant: в списке %d, в истории %d",
                session_id, message_index, in_msgs, in_hist)
        timeout = float(config.get("timeout") or _DEFAULT_TIMEOUT)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            base_url + "/v1/audio/tts", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        # The proxy reports 503 message_pending while the reply to the latest
        # prompt is still being generated (e.g. synthesize raced the stream).
        # Instead of voicing a predecessor, keep retrying until the reply is
        # committed — the same body resolves to it once generation ends.
        deadline = time.monotonic() + float(config.get("pending_wait") or 45.0)
        poll = float(config.get("pending_poll") or 0.7)
        try:
            while True:
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        audio = response.read()
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                    if (exc.code == 503 and '"message_pending"' in detail
                            and deadline > time.monotonic()):
                        time.sleep(poll)
                        continue
                    try:
                        _dump_diagnostics(payload, exc.code, detail)
                    except Exception:
                        pass
                    logger.warning("deepseek-session: прокси ответил %s: %s", exc.code, detail)
                    raise RuntimeError(
                        "deepseek-session: прокси ответил " + str(exc.code) + ": " + detail) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "deepseek-session: прокси недоступен: " + str(exc)) from exc
        if not audio:
            raise RuntimeError("deepseek-session: прокси вернул пустой ответ")
        target = os.path.expanduser(output_path)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        container = _sniff_container(audio)
        if container and container == os.path.splitext(target)[1].lower().lstrip("."):
            with open(target, "wb") as handle:
                handle.write(audio)
            return target
        handle_fd, temp_path = tempfile.mkstemp(suffix="." + (container or "bin"))
        os.close(handle_fd)
        try:
            with open(temp_path, "wb") as handle:
                handle.write(audio)
            if _ffmpeg_convert(temp_path, target):
                return target
            if container:
                fallback = os.path.splitext(target)[0] + "." + container
                shutil.copyfile(temp_path, fallback)
                return fallback
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise RuntimeError("deepseek-session: не удалось сохранить аудио (нужен ffmpeg)")


def register(ctx) -> None:
    ctx.register_middleware("llm_request", _on_llm_request)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_tts_provider(DeepSeekSessionTTS())
