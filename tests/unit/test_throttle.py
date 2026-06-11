"""Unit tests for throttle utilities (rate limiting + backoff retry)."""

from __future__ import annotations

import pytest

import src.libs.throttle as throttle
from src.libs.throttle import (
    RateLimiter,
    backoff_delay,
    is_dashscope_throttling,
    retry_call,
)


def test_rate_limiter_disabled_does_not_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(throttle.time, "sleep", lambda s: slept.append(s))
    RateLimiter(0.0).wait()
    assert slept == []


def test_rate_limiter_enforces_gap(monkeypatch):
    slept = []
    monkeypatch.setattr(throttle.time, "sleep", lambda s: slept.append(s))
    # monotonic stays constant -> second call must sleep ~full interval
    monkeypatch.setattr(throttle.time, "monotonic", lambda: 100.0)
    rl = RateLimiter(0.5)
    rl.wait()  # first call sets baseline (no prior gap)
    rl.wait()  # second call should request a sleep close to 0.5
    assert slept and slept[-1] == pytest.approx(0.5, abs=0.01)


def test_is_dashscope_throttling():
    assert is_dashscope_throttling(Exception("Requests rate limit exceeded"))
    assert is_dashscope_throttling(Exception("Throttling.RateQuota"))
    assert not is_dashscope_throttling(Exception("Authentication failed"))


def test_retry_call_returns_immediately_on_success():
    assert retry_call(lambda: 42) == 42


def test_retry_call_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(throttle.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("rate limit exceeded")
        return "ok"

    assert retry_call(flaky, max_retries=5, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_call_non_retriable_raises_immediately(monkeypatch):
    monkeypatch.setattr(throttle.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("auth failed")

    with pytest.raises(ValueError):
        retry_call(boom, max_retries=5, base_delay=0.0)
    assert calls["n"] == 1  # not retried


def test_retry_call_exhausts_and_raises(monkeypatch):
    monkeypatch.setattr(throttle.time, "sleep", lambda s: None)

    def always_throttled():
        raise Exception("Throttling")

    with pytest.raises(Exception, match="Throttling"):
        retry_call(always_throttled, max_retries=2, base_delay=0.0)


def test_backoff_delay_grows():
    d0 = backoff_delay(0, base=1.0)
    d2 = backoff_delay(2, base=1.0)
    assert 1.0 <= d0 <= 2.0
    assert 4.0 <= d2 <= 5.0
