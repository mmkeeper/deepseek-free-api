from __future__ import annotations

import logging
import re
from typing import Any, Callable

log = logging.getLogger("ds")

# DeepSeek wraps tool-call tags with a fullwidth-bars marker "||DSML||"
# (U+FF5C x2 + "DSML" + U+FF5C x2). It is a service delimiter that must not
# reach the client. The marker may arrive padded with whitespace around it
# and can be split across consecutive SSE events (e.g. "<", "｜", "｜", "DS",
# "ML", "｜", "｜", " invoke"), so it is stitched across events before being
# stripped.
_DSML_MARKER = "\uff5c\uff5cDSML\uff5c\uff5c"
_DSML_MARKER_RE = re.compile(r"\s*" + re.escape(_DSML_MARKER) + r"\s*")


def _strip_dsml(chunk: str, pending: str) -> tuple[str, str]:
    """Strip DSML markers from a chunk, tolerating markers split across chunks.

    Prepends the leftover tail of the previous chunk (`pending`) to `chunk`,
    removes every complete marker, and returns (text_to_emit, new_pending).
    The tail may be the start of a marker that is still arriving, so it is
    kept until the next chunk (or the end of the stream) decides whether it
    completes a marker.

    A COMPLETE marker at the very end of the buffer is also kept on hold: its
    padding space may arrive with the next chunk (DeepSeek tokenizes
    "<\\uff5c\\uff5cDSML\\uff5c\\uff5c calls>" as "<", bars+DSML, " calls>").
    The padding must be consumed together with the marker, otherwise the stray
    space survives and leaks to the client as "< calls>". The held marker is
    dropped by the caller at end-of-stream (only a genuinely incomplete tail —
    literal bars with no DSML — is flushed as text).
    """
    buf = pending + chunk
    hold_len = 0
    for k in range(1, len(_DSML_MARKER)):
        if buf.endswith(_DSML_MARKER[:k]):
            hold_len = k
    if buf.endswith(_DSML_MARKER):
        hold_len = len(_DSML_MARKER)
    if hold_len:
        return _DSML_MARKER_RE.sub("", buf[:-hold_len]), buf[-hold_len:]
    return _DSML_MARKER_RE.sub("", buf), ""


class DeepSeekError(Exception):
    def __init__(self, message: str, finish_reason: str | None = None):
        self.message = message
        self.finish_reason = finish_reason
        super().__init__(message)


def parse_sse_event(raw: str) -> dict:
    event = {"event": "", "data": ""}
    for line in raw.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            event["event"] = line[6:].strip()
        elif line.startswith("data:"):
            prefix = "\n" if event["data"] else ""
            event["data"] += prefix + line[5:].lstrip()
    return event


def extract_delta_text(value: Any, cache: dict, event_name: str = "") -> tuple[str, str, int | None]:
    message_id = None
    text = ""
    thinking = ""

    # Track fragment types in order — list of "THINK" or "RESPONSE" strings.
    # Updated from SNAPSHOT events; used to classify APPEND to fragments/-1.
    frag_types: list[str] = cache.setdefault("_frag_types", [])
    # Track content per fragment for delta calculation.
    # Keyed by fragment id, value is the full content string.
    frag_content: dict[int, str] = cache.setdefault("_frag_content", {})

    def visit(node: Any, path: str):
        nonlocal message_id, text, thinking

        if isinstance(node, list):
            for i, item in enumerate(node):
                visit(item, f"{path}.{i}")
            return

        if not isinstance(node, dict):
            return

        # Priority: response_message_id (the assistant message being generated,
        # which is what stop_stream targets) → assistant id → plain message_id.
        if isinstance(node.get("response_message_id"), int):
            message_id = node["response_message_id"]
        if isinstance(node.get("id"), int) and node.get("role") == "ASSISTANT":
            message_id = node["id"]
        if message_id is None and isinstance(node.get("message_id"), int):
            message_id = node["message_id"]

        # Content update without "o" marker: {"p": "response/fragments/-1/content", "v": "..."}
        if (
            path == "$"
            and isinstance(node.get("p"), str)
            and "content" in node["p"]
            and isinstance(node.get("v"), str)
            and "o" not in node
        ):
            if frag_types and frag_types[-1] == "THINK":
                thinking += node["v"]
            else:
                text += node["v"]
            return

        # Simple {"v": "text"} — raw token from the active fragment
        if (
            path == "$"
            and len(node) == 1
            and isinstance(node.get("v"), str)
        ):
            if frag_types and frag_types[-1] == "THINK":
                thinking += node["v"]
            else:
                text += node["v"]
            return

        # APPEND to fragment content: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "..."}
        if (
            node.get("o") == "APPEND"
            and isinstance(node.get("p"), str)
            and "fragments" in node["p"]
            and node["p"].endswith("content")
            and isinstance(node.get("v"), str)
        ):
            if frag_types and frag_types[-1] == "THINK":
                thinking += node["v"]
            else:
                text += node["v"]
            return

        # APPEND new fragment(s): {"p": "response/fragments", "o": "APPEND", "v": [{...}]}
        if (
            node.get("o") == "APPEND"
            and isinstance(node.get("p"), str)
            and node["p"].endswith("fragments")
            and isinstance(node.get("v"), list)
        ):
            for frag in node["v"]:
                if isinstance(frag, dict) and isinstance(frag.get("type"), str):
                    frag_types.append(frag["type"])
                    fid = frag.get("id")
                    fcontent = frag.get("content", "")
                    if isinstance(fid, int) and isinstance(fcontent, str):
                        frag_content[fid] = fcontent
                        if frag["type"] == "THINK":
                            thinking += fcontent
                        else:
                            text += fcontent
            return

        # BATCH: {"o": "BATCH", "v": [...]}
        if node.get("o") == "BATCH" and isinstance(node.get("v"), list):
            for i, item in enumerate(node["v"]):
                visit(item, f"{path}.v.{i}")
            return

        # SNAPSHOT with fragments — update frag_types and compute deltas
        # Structure: {"v": {"response": {"fragments": [...]}}}
        resp = node.get("v", {})
        if isinstance(resp, dict):
            resp = resp.get("response", resp)
        if isinstance(resp, dict) and isinstance(resp.get("fragments"), list):
            new_types = []
            for frag in resp["fragments"]:
                if isinstance(frag, dict) and isinstance(frag.get("type"), str):
                    new_types.append(frag["type"])
                    fid = frag.get("id")
                    fcontent = frag.get("content", "")
                    if isinstance(fid, int) and isinstance(fcontent, str):
                        prev = frag_content.get(fid, "")
                        if fcontent.startswith(prev):
                            delta = fcontent[len(prev):]
                        else:
                            delta = fcontent
                        frag_content[fid] = fcontent
                        if frag["type"] == "THINK":
                            thinking += delta
                        elif frag["type"] in ("RESPONSE", "TEMPLATE_RESPONSE"):
                            text += delta
            if new_types:
                frag_types[:] = new_types

        # OpenAI-style choices
        choices = node.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                text += delta["content"]

        for key, item in node.items():
            if key in ("content", "choices", "v"):
                continue
            visit(item, f"{path}.{key}")

    visit(value, "$")
    return text, thinking, message_id


async def stream_sse(
    response,
    on_text: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_message_id: Callable[[int], None] | None = None,
    debug: bool = False,
    req_id: str = "",
) -> dict:
    full_text = ""
    full_thinking = ""
    last_message_id = None
    fragments: dict[str, str] = {}
    buffer = ""
    sse_count = 0
    pending_text = ""
    pending_think = ""

    async for line in response.aiter_text():
        buffer += line
        while True:
            boundary = buffer.find("\n\n")
            if boundary < 0:
                break
            raw_event = buffer[:boundary]
            buffer = buffer[boundary + 2 :]

            event = parse_sse_event(raw_event)
            if not event["data"]:
                continue

            if debug:
                sse_count += 1
                log.debug(
                    f"[REQ-{req_id}] SSE#{sse_count} {event['event'] or 'message'}: "
                    f"{event['data'][:800]}"
                )

            import json

            try:
                parsed = json.loads(event["data"])
            except (json.JSONDecodeError, ValueError):
                continue

            if isinstance(parsed, dict) and parsed.get("type") == "error":
                err_msg = parsed.get("content", "Unknown DeepSeek error")
                finish_reason = parsed.get("finish_reason")
                raise DeepSeekError(err_msg, finish_reason=finish_reason)

            text, thinking, msg_id = extract_delta_text(parsed, fragments, event["event"])
            if msg_id is not None:
                last_message_id = msg_id
                if on_message_id:
                    on_message_id(msg_id)
            if text:
                text, pending_text = _strip_dsml(text, pending_text)
                full_text += text
                if text and on_text:
                    on_text(text)
            if thinking:
                thinking, pending_think = _strip_dsml(thinking, pending_think)
                full_thinking += thinking
                if thinking and on_thinking:
                    on_thinking(thinking)

    if pending_text:
        flushed = _DSML_MARKER_RE.sub("", pending_text)
        full_text += flushed
        if flushed and on_text:
            on_text(flushed)
    if pending_think:
        flushed = _DSML_MARKER_RE.sub("", pending_think)
        full_thinking += flushed
        if flushed and on_thinking:
            on_thinking(flushed)

    if debug:
        log.debug(f"[REQ-{req_id}] SSE done — {sse_count} events, {len(full_text)} chars text, {len(full_thinking)} chars thinking")
    return {"lastAssistantMessageId": last_message_id, "text": full_text, "thinking": full_thinking}
