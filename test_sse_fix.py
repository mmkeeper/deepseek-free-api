"""Test SSE parsing with actual DeepSeek event format."""
import sys
sys.path.insert(0, ".")

from src.sse import extract_delta_text


def _snap(types_and_content):
    """Helper: build a SNAPSHOT event with fragments."""
    frags = [{"id": i + 1, "type": t, "content": c, "elapsed_secs": None, "references": [], "stage_id": 1}
             for i, (t, c) in enumerate(types_and_content)]
    return {"v": {"response": {"message_id": 2, "parent_id": 1, "role": "ASSISTANT",
                                "thinking_enabled": True, "fragments": frags}}}


def test_think_snapshot_delta():
    """SNAPSHOT with THINK fragment — full content returned."""
    cache = {}
    ev = _snap([("THINK", "Hello")])
    t, th, mid = extract_delta_text(ev, cache)
    assert th == "Hello", f"got '{th}'"
    assert t == ""


def test_think_snapshot_grows():
    """SNAPSHOT with growing THINK content — only delta returned."""
    cache = {}
    extract_delta_text(_snap([("THINK", "Hel")]), cache)
    t, th, _ = extract_delta_text(_snap([("THINK", "Hello")]), cache)
    assert th == "lo", f"got '{th}'"
    assert t == ""


def test_append_to_think():
    """APPEND to fragments/-1/content — goes to thinking."""
    cache = {}
    extract_delta_text(_snap([("THINK", "Hel")]), cache)
    ev = {"o": "APPEND", "p": "response/fragments/-1/content", "v": "lo"}
    t, th, _ = extract_delta_text(ev, cache)
    assert th == "lo", f"got '{th}'"
    assert t == ""


def test_append_to_response():
    """APPEND to fragments/-1/content when last frag is RESPONSE — goes to text."""
    cache = {}
    extract_delta_text(_snap([("THINK", "thinking"), ("RESPONSE", "Hel")]), cache)
    ev = {"o": "APPEND", "p": "response/fragments/-1/content", "v": "lo"}
    t, th, _ = extract_delta_text(ev, cache)
    assert t == "lo", f"got '{t}'"
    assert th == ""


def test_simple_v_routed_by_frag_type():
    """Simple {v: text} routed based on last fragment type."""
    cache = {}
    # THINK is last
    extract_delta_text(_snap([("THINK", "x")]), cache)
    t, th, _ = extract_delta_text({"v": "think_token"}, cache)
    assert th == "think_token", f"got '{th}'"
    assert t == ""

    # RESPONSE is last
    extract_delta_text(_snap([("THINK", "x"), ("RESPONSE", "y")]), cache)
    t, th, _ = extract_delta_text({"v": "resp_token"}, cache)
    assert t == "resp_token", f"got '{t}'"
    assert th == ""


def test_new_fragment_appended():
    """APPEND to fragments array — adds new fragment."""
    cache = {}
    extract_delta_text(_snap([("THINK", "thinking")]), cache)

    # Append RESPONSE fragment
    ev = {"o": "APPEND", "p": "response/fragments",
          "v": [{"id": 3, "type": "RESPONSE", "content": "Hi", "references": [], "stage_id": 1}]}
    t, th, _ = extract_delta_text(ev, cache)
    assert t == "Hi", f"got '{t}'"
    assert th == ""


def test_full_sequence():
    """Full sequence: SNAPSHOT THINK, APPENDs, new RESPONSE fragment."""
    cache = {}

    # SNAPSHOT with THINK
    extract_delta_text(_snap([("THINK", "1")]), cache)

    # APPEND to THINK content
    ev1 = {"o": "APPEND", "p": "response/fragments/-1/content", "v": "."}
    t1, th1, _ = extract_delta_text(ev1, cache)
    assert th1 == ".", f"append think: got '{th1}'"

    # Simple tokens routed to THINK
    ev2 = {"v": " thinking"}
    t2, th2, _ = extract_delta_text(ev2, cache)
    assert th2 == " thinking", f"simple v think: got '{th2}'"

    # New RESPONSE fragment
    ev3 = {"o": "APPEND", "p": "response/fragments",
           "v": [{"id": 3, "type": "RESPONSE", "content": "Answer", "references": [], "stage_id": 1}]}
    t3, th3, _ = extract_delta_text(ev3, cache)
    assert t3 == "Answer", f"new response: got '{t3}'"
    assert th3 == ""

    # Simple tokens now routed to RESPONSE
    ev4 = {"v": "!"}
    t4, th4, _ = extract_delta_text(ev4, cache)
    assert t4 == "!", f"simple v resp: got '{t4}'"


def test_delta_tracking_response():
    """RESPONSE fragment content grows — only delta returned."""
    cache = {}
    extract_delta_text(_snap([("RESPONSE", "Hel")]), cache)
    t, _, _ = extract_delta_text(_snap([("RESPONSE", "Hello")]), cache)
    assert t == "lo", f"got '{t}'"


if __name__ == "__main__":
    tests = [
        test_think_snapshot_delta,
        test_think_snapshot_grows,
        test_append_to_think,
        test_append_to_response,
        test_simple_v_routed_by_frag_type,
        test_new_fragment_appended,
        test_full_sequence,
        test_delta_tracking_response,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")
