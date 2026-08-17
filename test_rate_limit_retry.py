"""Test rate_limit_reached retry logic in DeepSeekClient.complete()."""
import asyncio
import sys
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
    """All 7 retries fail — full backoff [1,2,4,8,16,32,64] used, error propagates."""
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


def test_retryable_finish_reasons_set():
    """The retryable finish reasons include both known transient errors."""
    assert "rate_limit_reached" in _RETRYABLE_FINISH_REASONS
    assert "expert_busy_use_default" in _RETRYABLE_FINISH_REASONS


if __name__ == "__main__":
    tests = [
        test_retries_with_backoff_then_succeeds,
        test_no_retry_for_other_errors,
        test_all_retries_exhausted,
        test_no_retry_after_partial_output,
        test_retries_on_expert_busy_use_default,
        test_retryable_finish_reasons_set,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            sys.exit(1)
    print("All tests passed.")
