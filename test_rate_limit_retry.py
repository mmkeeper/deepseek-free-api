"""Test rate_limit_reached retry logic in DeepSeekClient.complete()."""
import asyncio
import sys
import threading
import time
from unittest import mock

sys.path.insert(0, ".")

from src.client import DeepSeekClient, _RATE_LIMIT_BACKOFF, _RETRYABLE_FINISH_REASONS
from src.sse import DeepSeekError


def _client(**kwargs):
    c = DeepSeekClient("cookie", "token", debug=False)
    for k, v in kwargs.items():
        setattr(c, k, v)
    return c


def _run(coro):
    return asyncio.run(coro)


def test_retries_with_backoff_then_succeeds():
    """Raises rate limit twice, then succeeds — sleeps 1s then 2s."""
    sleeps = []
    attempts = []

    async def fake_once(**kwargs):
        attempts.append(kwargs.get("req_id"))
        if len(attempts) == 1:
            raise DeepSeekError("Слишком частые сообщения", "rate_limit_reached")
        if len(attempts) == 2:
            raise DeepSeekError("Слишком частые сообщения", "rate_limit_reached")
        return {"lastAssistantMessageId": 7, "text": "ok", "thinking": ""}

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        result = _run(c.complete("sid", "prompt", req_id="t1"))

    assert result == {"lastAssistantMessageId": 7, "text": "ok", "thinking": ""}
    assert sleeps == [1, 2], f"expected backoff 1s,2s got {sleeps}"
    assert len(attempts) == 3


def test_no_retry_for_other_errors():
    """Non-rate-limit errors propagate immediately without sleeping."""
    sleeps = []

    async def fake_once(**kwargs):
        raise DeepSeekError("boom", "some_other_reason")

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        try:
            _run(c.complete("sid", "prompt", req_id="t2"))
            assert False, "expected DeepSeekError"
        except DeepSeekError as e:
            assert e.finish_reason == "some_other_reason"
    assert sleeps == []


def test_all_retries_exhausted():
    """All retries fail — full backoff used, error propagates."""
    sleeps = []
    attempts = []

    async def fake_once(**kwargs):
        attempts.append(1)
        raise DeepSeekError("Слишком частые сообщения", "rate_limit_reached")

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        try:
            _run(c.complete("sid", "prompt", req_id="t3"))
            assert False, "expected DeepSeekError"
        except DeepSeekError as e:
            assert e.finish_reason == "rate_limit_reached"

    assert sleeps == _RATE_LIMIT_BACKOFF, f"expected {_RATE_LIMIT_BACKOFF} got {sleeps}"
    assert len(attempts) == len(_RATE_LIMIT_BACKOFF) + 1


def test_no_retry_after_partial_output():
    """If content already streamed before the rate limit, do not retry."""
    sleeps = []

    async def fake_once(on_text=None, on_thinking=None, on_message_id=None, **kwargs):
        if on_text:
            on_text("partial")
        raise DeepSeekError("Слишком частые сообщения", "rate_limit_reached")

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        try:
            _run(c.complete("sid", "prompt", req_id="t4", on_text=lambda t: None))
            assert False, "expected DeepSeekError"
        except DeepSeekError as e:
            assert e.finish_reason == "rate_limit_reached"

    assert sleeps == []


def test_retries_after_message_id_only():
    """A message_id arriving before the error must NOT block the retry.

    DeepSeek sends response_message_id in the very first 'ready' event,
    before any content — so it is not evidence of partial client output.
    """
    sleeps = []
    attempts = []

    async def fake_once(on_text=None, on_thinking=None, on_message_id=None, **kwargs):
        attempts.append(1)
        if on_message_id:
            on_message_id(184)
        if len(attempts) == 1:
            raise DeepSeekError("Слишком частые сообщения", "rate_limit_reached")
        return {"lastAssistantMessageId": 184, "text": "ok", "thinking": ""}

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        result = _run(c.complete("sid", "prompt", req_id="t6", on_message_id=lambda m: None))

    assert result == {"lastAssistantMessageId": 184, "text": "ok", "thinking": ""}
    assert sleeps == [1], f"expected backoff 1s got {sleeps}"
    assert len(attempts) == 2


def test_retries_on_expert_busy_use_default():
    """expert_busy_use_default is retryable too."""
    sleeps = []
    attempts = []

    async def fake_once(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise DeepSeekError("Сервер перегружен. Попробуйте позже или используйте быстрый режим.", "expert_busy_use_default")
        return {"lastAssistantMessageId": 9, "text": "ok", "thinking": ""}

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        result = _run(c.complete("sid", "prompt", req_id="t5"))

    assert result == {"lastAssistantMessageId": 9, "text": "ok", "thinking": ""}
    assert sleeps == [1], f"expected backoff 1s got {sleeps}"
    assert len(attempts) == 2


def test_retries_on_generation_timeout():
    """generation_timeout ("Сервер занят...") is retryable too."""
    sleeps = []
    attempts = []

    async def fake_once(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise DeepSeekError("Сервер занят, пожалуйста, попробуйте позже.", "generation_timeout")
        return {"lastAssistantMessageId": 10, "text": "ok", "thinking": ""}

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        result = _run(c.complete("sid", "prompt", req_id="t7"))

    assert result == {"lastAssistantMessageId": 10, "text": "ok", "thinking": ""}
    assert sleeps == [1], f"expected backoff 1s got {sleeps}"
    assert len(attempts) == 2


def test_retryable_finish_reasons_set():
    """The retryable finish reasons include all known transient errors."""
    assert "rate_limit_reached" in _RETRYABLE_FINISH_REASONS
    assert "expert_busy_use_default" in _RETRYABLE_FINISH_REASONS
    assert "generation_timeout" in _RETRYABLE_FINISH_REASONS
    assert "parallel_chat_limit" in _RETRYABLE_FINISH_REASONS


def test_retries_on_parallel_chat_limit():
    """parallel_chat_limit (concurrent generation) is retried like rate limit."""
    sleeps = []
    attempts = []

    async def fake_once(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise DeepSeekError("слишком много параллельных запросов", "parallel_chat_limit")
        return {"lastAssistantMessageId": 11, "text": "ok", "thinking": ""}

    c = _client(_complete_once=fake_once)
    with mock.patch("asyncio.sleep", side_effect=lambda s: sleeps.append(s)):
        result = _run(c.complete("sid", "prompt", req_id="t8"))

    assert result == {"lastAssistantMessageId": 11, "text": "ok", "thinking": ""}
    assert sleeps == [1], f"expected backoff 1s got {sleeps}"
    assert len(attempts) == 2


def test_completions_serialized_no_overlap():
    """Concurrent complete() calls to the same client never overlap.

    DeepSeek rejects parallel generations on one account — the semaphore must
    queue the second request until the first finishes, even when both start
    at the same moment (the bridge test fires 4 webhooks at once).
    """
    entries = []
    lock = threading.Lock()

    async def fake_once(req_id="", **kwargs):
        now = time.monotonic()
        with lock:
            active = [e for e in entries if e["end"] is None]
            assert len(active) == 0, f"overlap detected for {req_id}: {active}"
            entries.append({"req_id": req_id, "start": now, "end": None})
        await asyncio.sleep(0.05)
        with lock:
            for e in entries:
                if e["req_id"] == req_id:
                    e["end"] = time.monotonic()
        return {"lastAssistantMessageId": 12, "text": "ok", "thinking": ""}

    c = _client(_complete_once=fake_once)

    async def run(n=1):
        tasks = [asyncio.create_task(c.complete(f"s{i}", f"p{i}", req_id=f"race{i}"))
                 for i in range(n)]
        return await asyncio.gather(*tasks)

    results = _run(run(4))
    assert all(r["text"] == "ok" for r in results)
    assert len(entries) == 4

    # Entries must be strictly sequential — each next start after prev end.
    for prev, cur in zip(entries, entries[1:]):
        assert cur["start"] >= prev["end"] - 1e-9, f"overlap {prev['req_id']} -> {cur['req_id']}"


if __name__ == "__main__":
    tests = [
        test_retries_with_backoff_then_succeeds,
        test_no_retry_for_other_errors,
        test_all_retries_exhausted,
        test_no_retry_after_partial_output,
        test_retries_after_message_id_only,
        test_retries_on_expert_busy_use_default,
        test_retries_on_generation_timeout,
        test_retryable_finish_reasons_set,
        test_retries_on_parallel_chat_limit,
        test_completions_serialized_no_overlap,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")
