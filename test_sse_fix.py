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


def test_append_before_snapshot_buffered():
    """APPEND before SNAPSHOT — buffered, then replayed on SNAPSHOT."""
    cache = {}

    # APPEND arrives first — type unknown, should be buffered
    a1 = {"o": "APPEND", "p": "$.v.0/content", "v": "П"}
    t1, th1, _ = extract_delta_text(a1, cache)
    assert t1 == "", f"should be buffered, got text='{t1}'"
    assert th1 == "", f"should be buffered, got think='{th1}'"

    # More APPENDs before snapshot
    a2 = {"o": "APPEND", "p": "$.v.0/content", "v": "ятные значения."}
    t2, th2, _ = extract_delta_text(a2, cache)
    assert t2 == "" and th2 == ""

    # SNAPSHOT arrives — replays buffered APPENDs into thinking
    snap = {"o": "SNAPSHOT", "v": [{"type": "THINK", "content": "Пятные значения.", "id": 1, "role": "ASSISTANT"}]}
    t3, th3, _ = extract_delta_text(snap, cache)
    # Buffer replayed "Пятные значения." into thinking, snapshot delta is empty
    assert th3 == "Пятные значения.", f"buffer replayed into thinking: got '{th3}'"
    assert t3 == ""


def test_append_response_before_snapshot_buffered():
    """RESPONSE APPEND before SNAPSHOT — buffered, then replayed as text."""
    cache = {}

    a1 = {"o": "APPEND", "p": "$.v.0/content", "v": "Hello"}
    t1, th1, _ = extract_delta_text(a1, cache)
    assert t1 == "" and th1 == ""

    snap = {"o": "SNAPSHOT", "v": [{"type": "RESPONSE", "content": "Hello world", "id": 2, "role": "ASSISTANT"}]}
    t2, th2, _ = extract_delta_text(snap, cache)
    # Buffered "Hello" replayed as text, snapshot delta is " world"
    # Total returned text = "Hello" + " world" = "Hello world"
    assert t2 == "Hello world", f"response accumulated text: got '{t2}'"
    assert th2 == ""


def test_interleaved_fragments():
    """THINK and RESPONSE at different indices — APPENDs go to correct buckets."""
    cache = {}

    snap = {"o": "SNAPSHOT", "v": [
        {"type": "THINK", "content": "thinking...", "id": 1, "role": "ASSISTANT"},
        {"type": "RESPONSE", "content": "answer", "id": 2, "role": "ASSISTANT"},
    ]}
    t1, th1, mid = extract_delta_text(snap, cache)
    assert th1 == "thinking...", f"think snapshot: got '{th1}'"
    assert t1 == "answer", f"response snapshot: got '{t1}'"

    a1 = {"o": "APPEND", "p": "$.v.0/content", "v": " more"}
    t2, th2, _ = extract_delta_text(a1, cache)
    assert th2 == " more", f"think append: got '{th2}'"
    assert t2 == ""

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


def test_response_delta_tracking():
    """RESPONSE snapshot content grows — only delta is returned."""
    cache = {}

    snap1 = {"o": "SNAPSHOT", "v": [{"type": "RESPONSE", "content": "Hel", "id": 1, "role": "ASSISTANT"}]}
    t1, _, _ = extract_delta_text(snap1, cache)
    assert t1 == "Hel"

    snap2 = {"o": "SNAPSHOT", "v": [{"type": "RESPONSE", "content": "Hello", "id": 1, "role": "ASSISTANT"}]}
    t2, _, _ = extract_delta_text(snap2, cache)
    assert t2 == "lo", f"response delta: got '{t2}'"


def test_append_think_full_sequence():
    """Full sequence: multiple APPENDs before SNAPSHOT, then more after."""
    cache = {}

    # APPENDs before snapshot
    for ch in "Пятные":
        extract_delta_text({"o": "APPEND", "p": "$.v.0/content", "v": ch}, cache)

    # SNAPSHOT with full content
    snap = {"o": "SNAPSHOT", "v": [{"type": "THINK", "content": "Пятные значения.Погода", "id": 1, "role": "ASSISTANT"}]}
    _, th1, _ = extract_delta_text(snap, cache)
    # Buffered "Пятные" replayed, snapshot delta is " значения.Погода"
    # Total returned thinking = "Пятные" + " значения.Погода" = "Пятные значения.Погода"
    assert th1 == "Пятные значения.Погода", f"full sequence: got '{th1}'"

    # More APPENDs after snapshot
    a = {"o": "APPEND", "p": "$.v.0/content", "v": " завтра"}
    _, th2, _ = extract_delta_text(a, cache)
    assert th2 == " завтра", f"post-snapshot append: got '{th2}'"


if __name__ == "__main__":
    tests = [
        test_append_think_after_snapshot,
        test_append_before_snapshot_buffered,
        test_append_response_before_snapshot_buffered,
        test_interleaved_fragments,
        test_batch_append_think,
        test_simple_v_string,
        test_delta_tracking,
        test_response_delta_tracking,
        test_append_think_full_sequence,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")
