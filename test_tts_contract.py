"""Tests for the TTS message-resolution contract (server.py)."""

import pytest

import server
from server import (
    TtsResolveError,
    _hash_messages,
    _prefix_key,
    _record_turn,
    _resolve_tts_by_messages,
    _resolve_tts_by_session_index,
    _resolve_tts_raw,
    _resolve_tts_target,
    _session_frames,
    _session_store,
    _tts_pending,
    _user_messages,
)


@pytest.fixture(autouse=True)
def _clean_sessions():
    _session_frames.clear()
    _session_store.clear()
    _tts_pending.clear()
    yield
    _session_frames.clear()
    _session_store.clear()
    _tts_pending.clear()


def _frame_ok(session_id: str, key: str, mid: int, parent: int | None = None):
    _record_turn(session_id, key, parent, mid, "user text")


def _user(text):
    return {"role": "user", "content": text}


def _system(text="# Система"):
    return {"role": "system", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _store_turn(msgs, session_id="s", mid=7, parent=None):
    """Mirror STORE + _record_turn at the end of a completion turn."""
    nkey = _hash_messages(_user_messages(msgs))
    pkey = _prefix_key(msgs)
    _session_store[nkey] = (session_id, mid, False, None)
    if pkey:
        _session_store[pkey] = (session_id, mid, False, None)
    _record_turn(session_id, nkey, parent, mid, "user text")
    return nkey, pkey


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
    assert ei.value.param == "message_index"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_session_index("nobody", 0)
    assert ei.value.error_code == "message_not_voiceable"
    assert ei.value.param == "chat_session_id"


def test_resolve_bad_index_type_wins_over_unknown_session():
    # A malformed index must surface as a type error even when the session
    # id is well-formed but unknown — the type check must not be masked.
    for bad in ("abc", 1.5, True):
        with pytest.raises(TtsResolveError) as ei:
            _resolve_tts_by_session_index("nobody", bad)
        assert ei.value.error_code == "invalid_message_index"
        assert ei.value.param == "message_index"


def test_resolve_invalid_session_id():
    for bad in (None, 12345, 0, False, ""):
        with pytest.raises(TtsResolveError) as ei:
            _resolve_tts_by_session_index(bad, 0)
        assert ei.value.error_code == "invalid_chat_session_id"
        assert ei.value.param == "chat_session_id"


def test_resolve_raw():
    _frame_ok("s", "k1", 7)
    assert _resolve_tts_raw("s", 7) == ("s", 7)
    assert _resolve_tts_raw("s", "7") == ("s", 7)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_raw("s", 999)
    assert ei.value.error_code == "message_not_voiceable"
    assert ei.value.param == "message_id"


def test_resolve_raw_invalid_message_id_type():
    _frame_ok("s", "k1", 7)
    for bad in ("chatcmpl-abc", "7.5", 1.5, True, None, ["7"]):
        with pytest.raises(TtsResolveError) as ei:
            _resolve_tts_raw("s", bad)
        assert ei.value.error_code == "invalid_message_id"
        assert ei.value.param == "message_id"


def test_resolve_raw_invalid_session_id():
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_raw("", 7)
    assert ei.value.error_code == "invalid_chat_session_id"
    assert ei.value.param == "chat_session_id"


def test_resolve_target_missing_reference_blames_session_id():
    # The message reference is chat_session_id-driven; a missing/empty session
    # must be reported as invalid_chat_session_id, not message_index.
    for body in (
        {},
        {"message_index": 0},
        {"chat_session_id": ""},
        {"message_index": 0, "chat_session_id": ""},
    ):
        with pytest.raises(TtsResolveError) as ei:
            _resolve_tts_target(body)
        assert ei.value.error_code == "invalid_chat_session_id"
        assert ei.value.param == "chat_session_id"


def test_resolve_target_session_only_uses_newest():
    _frame_ok("s", "k1", 1)
    _frame_ok("s", "k2", 2)
    # no message_index → latest assistant reply of the session
    assert _resolve_tts_target({"chat_session_id": "s"}) == ("s", 2)


def test_resolve_target_bad_index_type_with_unknown_session():
    _frame_ok("s", "k1", 1)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_target({"chat_session_id": "nobody", "message_index": 1.5})
    assert ei.value.error_code == "invalid_message_index"
    assert ei.value.param == "message_index"


def test_resolve_target_session_id_type():
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_target({"chat_session_id": 12345, "message_index": 0})
    assert ei.value.error_code == "invalid_chat_session_id"
    assert ei.value.param == "chat_session_id"


def test_resolve_target_index_given():
    _frame_ok("s", "k1", 1)
    _frame_ok("s", "k2", 2)
    assert _resolve_tts_target({"chat_session_id": "s", "message_index": 0}) == ("s", 1)


def test_resolve_by_messages_last_assistant_reply():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Солнечно."), _user("а неделя?")]
    t3 = t2 + [_assistant("Тоже.")]
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    _store_turn(t3, mid=3)
    # list ends with the reply to voice — that turn's reply wins
    assert _resolve_tts_by_messages(t3) == ("s", 3)
    assert _resolve_tts_target({"messages": t3}) == ("s", 3)


def test_resolve_by_messages_ends_with_fresh_prompt():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Солнечно.")]
    _store_turn(t1, mid=1)
    # client pastes history ending with a new prompt (reply not yet committed)
    assert _resolve_tts_by_messages(t2) == ("s", 1)


def test_resolve_by_messages_ignores_tool_roundtrip():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Солнечно."), _user("а неделя?")]
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    tooled = t1 + [{"role": "assistant", "content": "готово",
                    "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    # a tool roundtrip changes no user/system keys → same turn is found
    assert _resolve_tts_by_messages(tooled) == ("s", 1)


def test_resolve_by_messages_unknown_history():
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages([_system(), _user("кто я?")])
    assert ei.value.error_code == "message_not_voiceable"
    assert ei.value.param == "messages"


def test_resolve_by_messages_invalid_inputs():
    for bad in ([], "not-a-list", {"role": "user"}, [{"role": "npc", "content": "x"}], [42]):
        with pytest.raises(TtsResolveError) as ei:
            _resolve_tts_by_messages(bad)
        assert ei.value.error_code == "invalid_messages"
        assert ei.value.param == "messages"


def test_resolve_target_messages_with_refinement():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Солнечно."), _user("а неделя?")]
    t3 = t2 + [_assistant("Тоже.")]
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    _store_turn(t3, mid=3)
    # messages identify the conversation; message_index/message_id pick the reply
    assert _resolve_tts_target({"messages": t3, "message_index": 0}) == ("s", 1)
    assert _resolve_tts_target({"messages": t3, "message_index": -1}) == ("s", 3)
    assert _resolve_tts_target({"messages": t3, "message_id": "2"}) == ("s", 2)


def test_resolve_target_messages_takes_precedence_over_session_id():
    t1 = [_system(), _user("погода?")]
    _store_turn(t1, mid=7)
    # chat_session_id present but stale/unknown — messages wins
    assert _resolve_tts_target({"chat_session_id": "other", "messages": t1}) == ("s", 7)


def _pending_for(msgs, session_id="s", mid=None):
    """Register the conversation's key as in-flight (like run_stream start)."""
    nkey = _hash_messages(_user_messages(msgs))
    pkey = _prefix_key(msgs)
    _tts_pending[nkey] = (session_id, mid)
    if pkey:
        _tts_pending[pkey] = (session_id, mid)
    return nkey, pkey


def test_resolve_messages_pending_no_mid_not_stale():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Старый ответ."), _user("а неделя?")]
    # committed reply for this key exists — but the answer is re-generating
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    _pending_for(t2, mid=None)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages(t2)
    assert ei.value.error_code == "message_pending"
    assert ei.value.param == "messages"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_target({"messages": t2})
    assert ei.value.error_code == "message_pending"


def test_resolve_messages_pending_with_mid_returns_current():
    t1 = [_system(), _user("погода?")]
    _store_turn(t1, mid=1)
    # reply assigned its DeepSeek id but stream not committed yet
    _pending_for(t1, mid=9)
    assert _resolve_tts_by_messages(t1) == ("s", 9)
    assert _resolve_tts_target({"messages": t1}) == ("s", 9)


def test_resolve_messages_after_pending_cleared_uses_frame():
    t1 = [_system(), _user("погода?")]
    _store_turn(t1, mid=7)
    _pending_for(t1, mid=None)
    with pytest.raises(TtsResolveError):
        _resolve_tts_by_messages(t1)
    # stream ended → pending cleared → committed frame is authoritative again
    _tts_pending.clear()
    assert _resolve_tts_by_messages(t1) == ("s", 7)


def test_resolve_messages_pending_new_prompt_without_store():
    # A brand-new prompt's reply is being generated before the first STORE —
    # reported as pending (retryable), not as an unknown history.
    t1 = [_system(), _user("новый вопрос?")]
    _pending_for(t1, mid=None)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_by_messages(t1)
    assert ei.value.error_code == "message_pending"
    assert ei.value.param == "messages"


def test_resolve_messages_index_counts_client_list():
    u1 = _user("погода?")
    a1 = _assistant("Солнечно.")
    u2 = _user("а неделя?")
    a2 = _assistant("Тоже.")
    u3 = _user("вечером?")
    a3 = _assistant("Тепло.")
    t1 = [_system(), u1]
    t2 = t1 + [a1, u2]
    t3 = t2 + [a2, u3]
    full = t3 + [a3]
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    _store_turn(t3, mid=3)
    # index numbers the ASSISTANT messages of the sent list (0-based)
    assert _resolve_tts_target({"messages": full, "message_index": 0}) == ("s", 1)
    assert _resolve_tts_target({"messages": full, "message_index": 1}) == ("s", 2)
    assert _resolve_tts_target({"messages": full, "message_index": 2}) == ("s", 3)
    assert _resolve_tts_target({"messages": full, "message_index": -1}) == ("s", 3)
    assert _resolve_tts_target({"messages": full, "message_index": -2}) == ("s", 2)
    assert _resolve_tts_target({"messages": full, "message_index": -3}) == ("s", 1)


def test_resolve_messages_index_out_of_client_range():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Солнечно."), _user("а неделя?")]
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    for bad in (2, 9, -3):
        with pytest.raises(TtsResolveError) as ei:
            _resolve_tts_target({"messages": t2, "message_index": bad})
        assert ei.value.error_code == "invalid_message_index"
        assert ei.value.param == "message_index"
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_target({"messages": t2, "message_index": 1.5})
    assert ei.value.error_code == "invalid_message_index"
    assert ei.value.param == "message_index"


def test_resolve_messages_index_restored_session_tail_alignment():
    # Proxy restarted after turn 1: frames/store only for turns 2..3, but the
    # client list still contains all three assistant messages. The index must be
    # counted against the LIST, and the recorded tail resolves the ids.
    u1 = _user("привет")
    a1 = _assistant("старый ответ из жизни до рестарта")
    u2 = _user("дальше?")
    a2 = _assistant("новый ответ 2")
    u3 = _user("ещё")
    a3 = _assistant("новый ответ 3")
    t1 = [_system(), u1]
    t2 = t1 + [a1, u2]
    t3 = t2 + [a2, u3]
    full = t3 + [a3]
    # post-restart recording only
    _store_turn(t2, mid=2)
    _store_turn(t3, mid=3)
    assert _resolve_tts_target({"messages": full, "message_index": 2}) == ("s", 3)
    assert _resolve_tts_target({"messages": full, "message_index": 1}) == ("s", 2)
    assert _resolve_tts_target({"messages": full, "message_index": -1}) == ("s", 3)
    # before the recorded tail — predates this proxy process
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_target({"messages": full, "message_index": 0})
    assert ei.value.error_code == "message_not_voiceable"
    assert ei.value.param == "message_index"


def test_resolve_messages_index_pending_newest():
    t1 = [_system(), _user("погода?")]
    t2 = t1 + [_assistant("Солнечно."), _user("а неделя?")]
    t3 = t2 + [_assistant("Тоже."), _user("вечером?")]
    full = t3 + [_assistant("Тепло.")]
    _store_turn(t1, mid=1)
    _store_turn(t2, mid=2)
    _store_turn(t3, mid=3)
    # newest reply (index -1) matches the full list key → pending governs it
    _pending_for(full, mid=9)
    assert _resolve_tts_target({"messages": full, "message_index": -1}) == ("s", 9)
    _tts_pending.clear()
    _pending_for(full, mid=None)
    with pytest.raises(TtsResolveError) as ei:
        _resolve_tts_target({"messages": full, "message_index": -1})
    assert ei.value.error_code == "message_pending"
    assert ei.value.param == "messages"
    # EARLIER replies are unaffected by the pending newest one
    assert _resolve_tts_target({"messages": full, "message_index": 0}) == ("s", 1)
    assert _resolve_tts_target({"messages": full, "message_index": 1}) == ("s", 2)