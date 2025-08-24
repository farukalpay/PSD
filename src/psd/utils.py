from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any, TypeVar

T = TypeVar("T")


def retry(
    exceptions: type[Exception] | tuple[type[Exception], ...],
    tries: int = 3,
    delay: float = 0.1,
    backoff: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            remaining, wait = tries, delay
            while remaining > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    time.sleep(wait)
                    remaining -= 1
                    wait *= backoff
            return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def critical_section(lock: Lock) -> Iterator[None]:
    """Context manager to guard critical sections with a lock."""

    lock.acquire()
    try:
        yield
    finally:
        lock.release()
