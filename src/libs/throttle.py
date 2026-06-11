"""Throttle utilities — proactive rate limiting + reactive backoff retry.

Centralises rate-limit handling for DashScope (Qwen) calls, which share one
account quota across chat / embedding / vision and are easy to throttle during
a large ingest. Two complementary strategies:

- **Proactive**: a process-wide :class:`RateLimiter` enforces a minimum gap
  between calls (trade ingest time for staying under the QPS limit).
- **Reactive**: :func:`retry_call` retries throttled/transient calls with
  exponential backoff + jitter.

Both are configured via environment variables so behaviour can be tuned without
code changes (defaults keep current behaviour: no forced delay):

- ``DASHSCOPE_MIN_INTERVAL``  seconds between calls (default ``0`` = off)
- ``DASHSCOPE_MAX_RETRIES``   backoff attempts (default ``6``)
- ``DASHSCOPE_RETRY_BASE``    base seconds for backoff (default ``2.0``)
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to default on missing/invalid."""
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to default on missing/invalid."""
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


class RateLimiter:
    """Thread-safe minimum-interval limiter (proactive throttling)."""

    def __init__(self, min_interval: float = 0.0):
        self._min_interval = max(0.0, min_interval)
        self._last = 0.0
        self._lock = threading.Lock()

    @property
    def min_interval(self) -> float:
        """Current enforced minimum gap between calls (seconds)."""
        return self._min_interval

    def wait(self) -> None:
        """Block until at least ``min_interval`` has elapsed since the last call."""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            remaining = self._min_interval - (now - self._last)
            if remaining > 0:
                time.sleep(remaining)
            self._last = time.monotonic()


# dashscope_limiter shared limiter for all DashScope/Qwen calls (env-configured)
dashscope_limiter = RateLimiter(_env_float("DASHSCOPE_MIN_INTERVAL", 0.0))


def is_dashscope_throttling(exc: BaseException) -> bool:
    """Whether an exception looks like a retriable DashScope rate-limit/throttle."""
    text = str(exc).lower()
    return any(k in text for k in ("throttl", "rate limit", "rate_limit", "ratequota"))


def backoff_delay(attempt: int, base: float | None = None) -> float:
    """Exponential backoff with jitter for the given 0-based attempt."""
    if base is None:
        base = _env_float("DASHSCOPE_RETRY_BASE", 2.0)
    return base * (2 ** attempt) + random.uniform(0, 1.0)


def retry_call(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool] = is_dashscope_throttling,
    max_retries: int | None = None,
    base_delay: float | None = None,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn`` with exponential-backoff retry on retriable errors.

    Args:
        fn: Zero-arg callable performing the (potentially throttled) request.
        is_retryable: Predicate deciding whether an exception is worth retrying.
        max_retries: Max retry attempts (default env ``DASHSCOPE_MAX_RETRIES``=6).
        base_delay: Backoff base seconds (default env ``DASHSCOPE_RETRY_BASE``=2).
        on_retry: Optional hook ``(attempt, exc)`` invoked before each sleep.

    Returns:
        The result of ``fn()``.

    Raises:
        The last exception if retries are exhausted or the error is not retriable.
    """
    if max_retries is None:
        max_retries = _env_int("DASHSCOPE_MAX_RETRIES", 6)

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if is_retryable(e) and attempt < max_retries:
                if on_retry is not None:
                    on_retry(attempt, e)
                time.sleep(backoff_delay(attempt, base_delay))
                continue
            raise
    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("retry_call exhausted without returning")
