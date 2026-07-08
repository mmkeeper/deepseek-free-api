from __future__ import annotations

from typing import Any, Callable


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

        if isinstance(node.get("response_message_id"), int):
            message_id = node["response_message_id"]
        if isinstance(node.get("message_id"), int):
            message_id = node["message_id"]
        if isinstance(node.get("id"), int) and node.get("role") == "ASSISTANT":
            message_id = node["id"]

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
    debug: bool = False,
) -> dict:
    full_text = ""
    full_thinking = ""
    last_message_id = None
    fragments: dict[str, str] = {}
    buffer = ""

    _dbg = None
    if debug:
        import os, tempfile
        _dbg = open(os.path.join(tempfile.gettempdir(), "ds_sse_debug.log"), "w", encoding="utf-8")

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
                import sys
                print(
                    f"[event] {event['event'] or 'message'} {event['data'][:500]}",
                    file=sys.stderr,
                )

            import json

            try:
                parsed = json.loads(event["data"])
            except (json.JSONDecodeError, ValueError):
                continue

            if _dbg:
                _dbg.write(json.dumps({"event": event["event"], "data": parsed}, ensure_ascii=False) + "\n")
                _dbg.flush()

            text, thinking, msg_id = extract_delta_text(parsed, fragments, event["event"])
            if msg_id is not None:
                last_message_id = msg_id
            if text:
                full_text += text
                if on_text:
                    on_text(text)
            if thinking:
                full_thinking += thinking
                if on_thinking:
                    on_thinking(thinking)

    if _dbg:
        _dbg.close()

    return {"lastAssistantMessageId": last_message_id, "text": full_text, "thinking": full_thinking}
