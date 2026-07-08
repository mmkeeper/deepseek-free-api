#!/usr/bin/env python3
"""Debug: trace visit function for event #3."""
import asyncio
import json
import sys
sys.path.insert(0, ".")
from src.client import DeepSeekClient
from src.auth import read_saved_auth

auth = read_saved_auth()
client = DeepSeekClient(cookie_header=auth["cookieHeader"], token=auth["token"])

async def test():
    from src.proxy import get_http_client
    from src.config import BASE_URL, COMPLETION_PATH

    session_id = await client.create_session()
    pow_header = await client.create_pow_header(COMPLETION_PATH)
    body = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": None,
        "preempt": False,
        "prompt": "User: Привет\n\nAssistant:",
        "ref_file_ids": [],
        "thinking_enabled": False,
        "search_enabled": False,
    }

    http_client = get_http_client()
    url = f"{BASE_URL}{COMPLETION_PATH}"
    headers = {**client._build_headers(), "X-DS-PoW-Response": pow_header}
    content = json.dumps(body)

    raw_buffer = ""
    event_num = 0

    async with http_client.stream("POST", url, headers=headers, content=content) as resp:
        async for chunk in resp.aiter_text():
            raw_buffer += chunk
            while True:
                boundary = raw_buffer.find("\n\n")
                if boundary < 0:
                    break
                raw_event = raw_buffer[:boundary]
                raw_buffer = raw_buffer[boundary + 2:]
                event_num += 1

                if event_num == 3:
                    # Parse all data lines
                    data_lines = []
                    for line in raw_event.split("\n"):
                        line = line.rstrip("\r")
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    for dl in data_lines:
                        parsed = json.loads(dl)
                        # Manually trace visit
                        def trace_visit(node, path, depth=0):
                            indent = "  " * depth
                            if isinstance(node, list):
                                print(f"{indent}LIST at {path} len={len(node)}")
                                for i, item in enumerate(node):
                                    trace_visit(item, f"{path}.{i}", depth+1)
                                return
                            if not isinstance(node, dict):
                                print(f"{indent}NON-DICT at {path}: {type(node).__name__}")
                                return
                            content = node.get("content")
                            node_type = node.get("type")
                            if isinstance(content, str) and node_type in ("RESPONSE", "TEMPLATE_RESPONSE", "THINK"):
                                print(f"{indent}FOUND content at {path}: type={node_type} content=[{content}]")
                            for key, val in node.items():
                                if key in ("content", "choices"):
                                    continue
                                trace_visit(val, f"{path}.{key}", depth+1)
                        trace_visit(parsed, "$")
                    return

asyncio.run(test())
