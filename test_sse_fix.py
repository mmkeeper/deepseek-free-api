"""Test SSE first-char loss fix for THINK fragments."""
import sys
sys.path.insert(0, ".")

from src.sse import extract_delta_text


def test_append_think_after_snapshot():
    """THINK snapshot then APPEND — correctly routed to thinking."""
    cache = {}

    snap = {"o": "SNAPSHOT", "v": [{"type": "THINK", "content": "П", "id": 1, "role": "ASSISTANT"}]}
    t1, th1, mid = extract_delta_text(snap, cache)
    assert th1 == "П", f"snapshot delta: got '{th1}'"
    assert t1 == ""
    assert mid == 1

    append1 = {"o": "APPEND", "p": "$.v.0/content", "v": "ятные значения."}
    t2, th2, _ = extract_delta_text(append1, cache)
    assert th2 == "ятные значения.", f"append think: got '{th2}'"
    assert t2 == ""

    append2 = {"o": "APPEND", "p": "$.v.0/content", "v": "Погода"}
    t3, th3, _ = extract_delta_text(append2, cache)
    assert th3 == "Погода", f"append think 2: got '{th3}'"
    assert t3 == ""


def test_append_response_after_snapshot():
    """RESPONSE snapshot then APPEND — all goes to text."""
    cache = {}

    snap = {"o": "SNAPSHOT", "v": [{"type": "RESPONSE", "content": "Hello", "id": 2, "role": "ASSISTANT"}]}
    t1, th1, mid = extract_delta_text(snap, cache)
    assert t1 == "Hello"
    assert th1 == ""
    assert mid == 2

    append = {"o": "APPEND", "p": "$.v.0/content", "v": " world"}
    t2, th2, _ = extract_delta_text(append, cache)
    assert t2 == " world", f"append text: got '{t2}'"
    assert th2 == ""


def test_interleaved_fragments():
    """THINK and RESPONSE at different indices — APPENDs go to correct buckets."""
    cache = {}

    # Single SNAPSHOT with both THINK (index 0) and RESPONSE (index 1)
    snap = {"o": "SNAPSHOT", "v": [
        {"type": "THINK", "content": "thinking...", "id": 1, "role": "ASSISTANT"},
        {"type": "RESPONSE", "content": "answer", "id": 2, "role": "ASSISTANT"},
    ]}
    t1, th1, mid = extract_delta_text(snap, cache)
    assert th1 == "thinking...", f"think snapshot: got '{th1}'"
    assert t1 == "answer", f"response snapshot: got '{t1}'"

    # APPEND to THINK fragment (index 0)
    a1 = {"o": "APPEND", "p": "$.v.0/content", "v": " more"}
    t2, th2, _ = extract_delta_text(a1, cache)
    assert th2 == " more", f"think append: got '{th2}'"
    assert t2 == ""

    # APPEND to RESPONSE fragment (index 1)
    a2 = {"o": "APPEND", "p": "$.v.1/content", "v": "!"}
    t3, th3, _ = extract_delta_text(a2, cache)
    assert t3 == "!", f"response append: got '{t3}'"
    assert th3 == ""


def test_batch_append_think():
    """BATCH event wrapping APPEND to THINK fragment."""
    cache = {}

    snap = {"o": "SNAPSHOT", "v": [{"type": "THINK", "content": "x", "id": 1, "role": "ASSISTANT"}]}
    extract_delta_text(snap, cache)

    batch = {"o": "BATCH", "v": [{"o": "APPEND", "p": "$.v.0/content", "v": "y"}]}
    t, th, _ = extract_delta_text(batch, cache)
    assert th == "y", f"batch think: got '{th}'"
    assert t == ""


def test_simple_v_string():
    """Simple {v: 'text'} event goes to text."""
    cache = {}
    t, th, _ = extract_delta_text({"v": "hello"}, cache)
    assert t == "hello"
    assert th == ""


def test_delta_tracking():
    """Snapshot content grows — only delta is returned."""
    cache = {}

    snap1 = {"o": "SNAPSHOT", "v": [{"type": "THINK", "content": "ab", "id": 1, "role": "ASSISTANT"}]}
    _, th1, _ = extract_delta_text(snap1, cache)
    assert th1 == "ab"

    snap2 = {"o": "SNAPSHOT", "v": [{"type": "THINK", "content": "abcde", "id": 1, "role": "ASSISTANT"}]}
    _, th2, _ = extract_delta_text(snap2, cache)
    assert th2 == "cde", f"delta: got '{th2}'"


def test_append_before_snapshot():
    """APPEND before any snapshot — type unknown, goes to text (known limitation)."""
    cache = {}
    ev = {"o": "APPEND", "p": "$.v.0/content", "v": "П"}
    text, think, _ = extract_delta_text(ev, cache)
    # Without a prior snapshot, we can't know this is THINK content
    assert text == "П", f"expected text='П', got '{text}'"
    assert think == "", f"expected empty think, got '{think}'"


def test_response_delta_tracking():
    """RESPONSE snapshot content grows — only delta is returned."""
    cache = {}

    snap1 = {"o": "SNAPSHOT", "v": [{"type": "RESPONSE", "content": "Hel", "id": 1, "role": "ASSISTANT"}]}
    t1, _, _ = extract_delta_text(snap1, cache)
    assert t1 == "Hel"

    snap2 = {"o": "SNAPSHOT", "v": [{"type": "RESPONSE", "content": "Hello", "id": 1, "role": "ASSISTANT"}]}
    t2, _, _ = extract_delta_text(snap2, cache)
    assert t2 == "lo", f"response delta: got '{t2}'"


if __name__ == "__main__":
    tests = [
        test_append_think_after_snapshot,
        test_append_response_after_snapshot,
        test_interleaved_fragments,
        test_batch_append_think,
        test_simple_v_string,
        test_delta_tracking,
        test_append_before_snapshot,
        test_response_delta_tracking,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")
