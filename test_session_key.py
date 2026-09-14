"""Test session-key logic: _prefix_key must not collide across conversations.

Regression: a fresh "new session" request [system, user] computed
pkey = hash([system]) — the same system prompt is used by every session, so the
store matched the previous conversation and DeepSeek continued the wrong
session instead of creating a new one.
"""
import pytest
import sys

sys.path.insert(0, ".")

import server
from server import _hash_messages, _prefix_key, _user_messages

SYSTEM = "# Система"


@pytest.fixture(autouse=True)
def _clean_store():
    server._session_store.clear()
    server._session_frames.clear()
    yield
    server._session_store.clear()
    server._session_frames.clear()


def _system(text="# Система"):
    return {"role": "system", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


def _assistant_tc(text="готово", name="tool_x"):
    return {"role": "assistant", "content": text,
            "tool_calls": [{"id": "call_1", "function": {"name": name, "arguments": "{}"}}]}


def _tool(content='{"ok": true}'):
    return {"role": "tool", "tool_call_id": "call_1", "content": content}


def test_fresh_first_turn_system_only_prefix_is_empty():
    """[system, user] 'new session' must NOT produce a key — no collision."""
    msgs = [_system(), _user("ты живой?")]
    assert _prefix_key(msgs) == ""


def test_second_turn_with_system_matches_first_turn_nkey():
    """[system, u1, assistant, u2] continues turn 1's session."""
    turn1 = [_system(), _user("какая погода?")]
    turn2 = turn1 + [_assistant("Солнечно."), _user("а на неделю?")]
    n1 = _hash_messages(_user_messages(turn1))
    assert _prefix_key(turn2) == n1


def test_second_turn_without_system_matches_first_turn_nkey():
    """No system prompt: [user, assistant, user] still continues."""
    turn1 = [_user("какая погода?")]
    turn2 = turn1 + [_assistant("Солнечно."), _user("а на неделю?")]
    n1 = _hash_messages(_user_messages(turn1))
    assert _prefix_key(turn2) == n1


def test_prefix_skips_tool_result_messages():
    """Tool result turn keeps the same prefix key as a plain second turn."""
    base = [_system(), _user("погода?")]
    plain = base + [_assistant("Солнечно."), _user("а на неделю?")]
    tooled = base + [_assistant_tc(), _tool(), _user("а дальше?")]
    assert _prefix_key(tooled) == _prefix_key(plain)


def test_prefix_empty_for_first_turn_assistant_messages():
    assert _prefix_key([]) == ""
    assert _prefix_key([_system()]) == ""
    assert _prefix_key([_user("hello")]) == ""


def test_system_only_prefix_differs_from_user_prefix():
    """Hash of system+first user must not equal the system-only hash."""
    sys_msg = _system()
    sys_key = _hash_messages([{"role": "system", "content": sys_msg["content"]}])
    conv_key = _hash_messages([{"role": "system", "content": sys_msg["content"]},
                               {"role": "user", "content": "hello"}])
    assert sys_key != conv_key
    assert _prefix_key([_system(), _user("hello")]) == ""


def _store_turn(msgs, session_id="sid-old", last_id=7):
    """Mirror STORE at the end of handle_completion (server.py ~1493-1497)."""
    nkey = _hash_messages(_user_messages(msgs))
    pkey = _prefix_key(msgs)
    server._session_store[nkey] = (session_id, last_id, False, None)
    if pkey:
        server._session_store[pkey] = (session_id, last_id, False, None)


def _lookup(messages):
    """Mirror the session resolve branch of handle_completion (server.py ~1256-1293)."""
    nkey = _hash_messages(_user_messages(messages))
    existing = server._session_store.get(nkey)
    if not existing:
        pkey = _prefix_key(messages)
        existing = server._session_store.get(pkey) if pkey else None
    return existing


def test_fresh_first_turn_resolves_to_new_session_despite_poisoned_store():
    """Regression: new conversation [system, 'ты живой?'] must NOT continue a
    previous conversation whose first turn stored hash([system]).

    Reproduces the real bug: previous conversation A + an old pre-fix server
    instance left _session_store[hash([system])] pointing at sid-old. With the
    buggy _prefix_key the fresh request computed pkey = hash([system]) → store
    HIT → DeepSeek continued the wrong session.
    """
    conv_a = [_system(), _user("какая погода?")]
    _store_turn(conv_a, session_id="sid-old")

    # Leftover from a server that ran the buggy _prefix_key (still poisoned in
    # memory until restart): storage key for a system-only prefix.
    poisoned_key = _hash_messages(_user_messages([_system()]))
    server._session_store[poisoned_key] = ("sid-old", 3, False, None)

    fresh = [_system(), _user("ты живой?")]
    pkey = _prefix_key(fresh)
    assert pkey == "", "fresh first turn must not produce a pkey"

    existing = _lookup(fresh)
    assert existing is None, "fresh conversation must resolve to SESSION: NEW"


def test_second_turn_continues_via_pkey_from_store():
    """A real second turn still continues the first turn's session after the fix."""
    turn1 = [_system(), _user("какая погода?")]
    _store_turn(turn1, session_id="sid-old", last_id=7)

    turn2 = turn1 + [_assistant("Солнечно."), _user("а на неделю?")]
    existing = _lookup(turn2)
    assert existing is not None
    assert existing[0] == "sid-old"
    assert existing[1] == 7