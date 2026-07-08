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

    # Track fragment types so APPEND events know whether content belongs to
    # THINK or RESPONSE.  Keyed by the visit path (e.g. "$.v.0").
    fragment_types: dict[str, str] = cache.setdefault("_fragment_types", {})
    # Buffer APPEND text that arrives before its SNAPSHOT establishes the type.
    # Keyed by the visit path derived from the APPEND path (e.g. "$.v.0").
    pending: dict[str, list[str]] = cache.setdefault("_pending_appends", {})

    # SSE event name can indicate fragment type (e.g. "thinking" vs "message")
    event_type_from_name = ""
    if event_name:
        ename = event_name.lower()
        if "think" in ename:
            event_type_from_name = "THINK"

    def _append_path_to_visit(append_path: str) -> str:
        """Convert APPEND path like '$.v.0/content' to visit path '$.v.0'."""
        if append_path.endswith("/content"):
            return append_path[:-len("/content")]
        return append_path

    def _resolve_type(append_path: str) -> str:
        visit_path = _append_path_to_visit(append_path)
        ftype = fragment_types.get(visit_path, "")
        if not ftype:
            for known_path, ft in fragment_types.items():
                if visit_path.startswith(known_path + "/") or visit_path == known_path:
                    ftype = ft
                    break
        # Fallback: use SSE event name to determine type
        if not ftype and event_type_from_name:
            ftype = event_type_from_name
        return ftype

    def _replay_pending(visit_path: str, ftype: str) -> int:
        """Replay buffered APPENDs for a visit path once its type is known.
        Returns the length of content that was replayed."""
        nonlocal text, thinking
        pieces = pending.pop(visit_path, [])
        if not pieces:
            return 0
        combined = "".join(pieces)
        if ftype == "THINK":
            thinking += combined
        else:
            text += combined
        return len(combined)

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

        if (
            path == "$"
            and len(node) == 1
            and isinstance(node.get("v"), str)
        ):
            text += node["v"]
            return

        if (
            node.get("o") == "APPEND"
            and isinstance(node.get("p"), str)
            and node["p"].endswith("/content")
            and isinstance(node.get("v"), str)
        ):
            append_path = node["p"]
            frag_type = _resolve_type(append_path)
            if frag_type == "THINK":
                thinking += node["v"]
            elif frag_type:
                text += node["v"]
            else:
                # Type unknown — buffer until SNAPSHOT arrives
                visit_path = _append_path_to_visit(append_path)
                pending.setdefault(visit_path, []).append(node["v"])
            return

        if node.get("o") == "BATCH" and isinstance(node.get("v"), list):
            for i, item in enumerate(node["v"]):
                visit(item, f"{path}.v.{i}")
            return

        if isinstance(node.get("content"), str) and node.get("type") in (
            "RESPONSE",
            "TEMPLATE_RESPONSE",
        ):
            key = f"{message_id or 'unknown'}:{path}:{node['type']}"
            previous = cache.get(key, "")
            current = node["content"]
            # Record type and replay any buffered APPENDs for this fragment
            fragment_types[path] = node["type"]
            replayed_len = _replay_pending(path, node["type"])
            # Adjust delta to skip content already contributed by replayed buffer
            skip = max(len(previous), replayed_len)
            delta = current[skip:] if current.startswith(current[:skip]) else current[len(previous):]
            cache[key] = current
            text += delta

        if isinstance(node.get("content"), str) and node.get("type") == "THINK":
            key = f"{message_id or 'unknown'}:{path}:THINK"
            previous = cache.get(key, "")
            current = node["content"]
            # Record type and replay any buffered APPENDs for this fragment
            fragment_types[path] = "THINK"
            replayed_len = _replay_pending(path, "THINK")
            # Adjust delta to skip content already contributed by replayed buffer
            skip = max(len(previous), replayed_len)
            delta = current[skip:] if current.startswith(current[:skip]) else current[len(previous):]
            cache[key] = current
            thinking += delta

        choices = node.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                text += delta["content"]

        for key, item in node.items():
            if key in ("content", "choices"):
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

    return {"lastAssistantMessageId": last_message_id, "text": full_text, "thinking": full_thinking}
