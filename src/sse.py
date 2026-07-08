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
    # THINK or RESPONSE.  Keyed by the content path that APPEND events use
    # (e.g. "$.v.0/content"), which we derive from the snapshot's visit path.
    fragment_types: dict[str, str] = cache.setdefault("_fragment_types", {})

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
            frag_type = fragment_types.get(append_path, "")
            if not frag_type:
                for known_path, ftype in fragment_types.items():
                    if append_path.startswith(known_path + "/"):
                        frag_type = ftype
                        break
            if frag_type == "THINK" or node.get("type") == "THINK":
                thinking += node["v"]
            else:
                text += node["v"]
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
            delta = current[len(previous) :] if current.startswith(previous) else current
            cache[key] = current
            text += delta
            # Record fragment type so future APPEND events on child paths
            # (e.g. "$.v.0/content") can look up via prefix match.
            fragment_types[path] = node["type"]

        if isinstance(node.get("content"), str) and node.get("type") == "THINK":
            key = f"{message_id or 'unknown'}:{path}:THINK"
            previous = cache.get(key, "")
            current = node["content"]
            delta = current[len(previous) :] if current.startswith(previous) else current
            cache[key] = current
            thinking += delta
            # Record fragment type for prefix-based APPEND lookup
            fragment_types[path] = "THINK"

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
