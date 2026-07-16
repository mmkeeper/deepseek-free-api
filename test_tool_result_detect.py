"""Test tool result detection logic."""
import re
import json
import sys
sys.path.insert(0, ".")


def detect_tool_result(messages):
    last_msg = messages[-1] if messages else {}
    last_tool_call_id = None
    is_tool_result = False

    if last_msg.get("role") == "tool":
        last_tool_call_id = last_msg.get("tool_call_id")
        is_tool_result = True
    elif last_msg.get("role") == "user" and len(messages) >= 2:
        prev = messages[-2]
        if prev.get("role") == "assistant":
            prev_text = prev.get("content") or ""
            if isinstance(prev_text, list):
                prev_text = "\n".join(item.get("text", "") for item in prev_text if item.get("type") == "text")
            if re.search(r'<tool_call\s+name=', prev_text) or re.search(r'<invoke\s+name=', prev_text):
                is_tool_result = True
            elif prev.get("tool_calls"):
                is_tool_result = True
                if not last_tool_call_id:
                    tc0 = prev["tool_calls"][0]
                    last_tool_call_id = tc0.get("id")
    return is_tool_result, last_tool_call_id


def test_xml_in_content():
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": '<tool_call name="search_files"></tool_call>'},
        {"role": "user", "content": "tool result here"},
    ]
    r, tid = detect_tool_result(msgs)
    assert r is True, f"expected True, got {r}"
    print(f"  PASS: XML in content -> tool_call_id={tid}")


def test_openai_tool_calls_field():
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "search_files", "arguments": "{}"}}
        ]},
        {"role": "user", "content": '{"matches": []}'},
    ]
    r, tid = detect_tool_result(msgs)
    assert r is True, f"expected True, got {r}"
    assert tid == "call_123", f"expected call_123, got {tid}"
    print(f"  PASS: OpenAI tool_calls field -> tool_call_id={tid}")


def test_role_tool_message():
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_456"}]},
        {"role": "tool", "tool_call_id": "call_456", "content": "result"},
    ]
    r, tid = detect_tool_result(msgs)
    assert r is True, f"expected True, got {r}"
    assert tid == "call_456", f"expected call_456, got {tid}"
    print(f"  PASS: role=tool message -> tool_call_id={tid}")


def test_regular_user_message():
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]
    r, tid = detect_tool_result(msgs)
    assert r is False, f"expected False, got {r}"
    print("  PASS: regular user msg -> not detected")


def test_assistant_with_content_no_tool():
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Sure, let me help."},
        {"role": "user", "content": "thanks"},
    ]
    r, tid = detect_tool_result(msgs)
    assert r is False, f"expected False, got {r}"
    print("  PASS: assistant with regular content -> not detected")


def test_openai_tool_calls_empty_content():
    """Hermes format: assistant has empty content + tool_calls in separate field."""
    msgs = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_789", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
        ]},
        {"role": "user", "content": "output here"},
    ]
    r, tid = detect_tool_result(msgs)
    assert r is True, f"expected True, got {r}"
    assert tid == "call_789", f"expected call_789, got {tid}"
    print(f"  PASS: empty content + tool_calls -> tool_call_id={tid}")


if __name__ == "__main__":
    tests = [
        test_xml_in_content,
        test_openai_tool_calls_field,
        test_role_tool_message,
        test_regular_user_message,
        test_assistant_with_content_no_tool,
        test_openai_tool_calls_empty_content,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")
