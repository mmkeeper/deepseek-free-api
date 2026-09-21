"""Tests for the TTS message-resolution contract (server.py)."""

import pytest

import server
from server import (
    TtsResolveError,
    _hash_messages,
    _record_turn,
    _resolve_tts_by_messages,
    _resolve_tts_by_session_index,
    _resolve_tts_raw,
    _session_frames,
    _session_store,
    _user_messages,
)


@pytest.fixture(autouse=True)
def _clean_sessions():
    _session_store.clear()
    _session_frames.clear()
    yield
    _session_store.clear()
    _session_frames.clear()


def _frame_ok(session_id: str, key: str, mid: int, parent: int | None = None):
    _record_turn(session_id, key, parent, mid, "user text")


def test_session_turn_keys_distinct():
    _frame_ok("s", "k1", 1)
    _frame_ok("s", "k2", 2)
    _frame_ok("s", "k1", 3)  # regeneration of k1 — same slot
    assert server._session_turn_keys("s") == ["k1", "k2"]


def test_resolve_by_session_index_fresh_and_regenerated():
    _frame_ok("s", "k1", 1)
    _frame_ok("s", "k2", 2)
    _frame_ok("s", "k2", 20)  # regeneration → newest reply is used
    _frame_ok("s", "k3", 3)
    assert _resolve_tts_by_session_index("s", 0) == ("s", 1)
    assert _resolve_tts_by_session_index("s", 1) == ("s", 20)
    assert _resolve_tts_by_session_index("s", 2) == ("s", 3)
    assert _resolve_tts_by_session_index("s", -1) == ("s", 3)
    assert _resolve_tts_by_session_index("s", -3) == ("s", 1)
    # omitted index → newest
    assert _resolve_tts_by_session_index("s", None) == ("s", 3)


def test_resolve_by_session_index_errors():
    _frame_ok("s", "k1", 1)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_session_index("s", 5)
    assert ei.value.error_code == "invalid_message_index"
    assert ei.value.param == "message_index"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_session_index("s", "x")
    assert ei.value.error_code == "invalid_message_index"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_session_index("nobody", 0)
    assert ei.value.error_code == "message_not_voiceable"
    assert ei.value.param == "chat_session_id"


def test_resolve_raw():
    _frame_ok("s", "k1", 7)
    assert _resolve_tts_raw("s", 7) == ("s", 7)
    assert _resolve_tts_raw("s", "7") == ("s", 7)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_raw("s", 999)
    assert ei.value.error_code == "message_not_voiceable"
    assert ei.value.param == "message_id"


def _build_history() -> list[dict]:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    n1 = _hash_messages(_user_messages(messages[:2]))
    n2 = _hash_messages(_user_messages(messages[:4]))
    _session_store[n1] = ("s", 11, False, None)
    _session_store[n2] = ("s", 22, False, None)
    _record_turn("s", n1, None, 11, "u1")
    _record_turn("s", n2, 11, 22, "u2")
    return messages


def test_resolve_by_messages_with_index():
    messages = _build_history()
    assert _resolve_tts_by_messages(messages, 4) == ("s", 22)
    assert _resolve_tts_by_messages(messages, 2) == ("s", 11)
    assert _resolve_tts_by_messages(messages, -1) == ("s", 22)
    assert _resolve_tts_by_messages(messages, -3) == ("s", 11)


def test_resolve_by_messages_newest_scan():
    messages = _build_history()
    assert _resolve_tts_by_messages(messages, None) == ("s", 22)


def test_resolve_by_messages_errors():
    messages = _build_history()
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages([], 0)
    assert ei.value.error_code == "invalid_message_index"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages(messages, 9)
    assert ei.value.error_code == "invalid_message_index"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages(messages, -9)
    assert ei.value.error_code == "invalid_message_index"
    # index points at a non-assistant message (u1 at index 1)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages(messages, 1)
    assert ei.value.error_code == "not_assistant_message"
    assert ei.value.param == "message_index"


def test_resolve_by_messages_unrecorded_reply():
    messages = _build_history()
    # a third turn the proxy never produced → not voiceable
    messages.append({"role": "user", "content": "u3"})
    messages.append({"role": "assistant", "content": "a3"})
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages(messages, -1)
    assert ei.value.error_code == "message_not_voiceable"