"""Concurrency policies behind `Dispatcher`: one after another, or on a thread pool.

Deployment code, chosen by `bootstrap` from the settings and never fingerprinted: a policy
changes when results arrive, never what they are.
"""

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from types import TracebackType
from typing import Self


class SerialDispatcher:
    """`Dispatcher` running the calls one after another on the calling thread."""

    def map[T, R](self, function: Callable[[T], R], items: Iterable[T]) -> Iterator[R]:
        """Apply ``function`` to each item as the result is consumed."""
        return map(function, items)


class ThreadDispatcher:
    """`Dispatcher` over a thread pool it owns; a context manager that drains the pool on exit.

    Suits work that releases the interpreter lock: decoding in OpenCV, waiting on a subprocess,
    talking to a store.
    """

    def __init__(self, workers: int, thread_name_prefix: str = "dispatch") -> None:
        """Start a pool of ``workers`` threads named after ``thread_name_prefix``.

        Raises:
            ValueError: When ``workers`` is below one.
        """
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self._workers = workers
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=thread_name_prefix
        )

    @property
    def workers(self) -> int:
        """Size of the pool."""
        return self._workers

    def map[T, R](self, function: Callable[[T], R], items: Iterable[T]) -> Iterator[R]:
        """Submit every item to the pool at once; yield the results in item order.

        When a call raises, the calls still waiting are cancelled (a worker freed meanwhile may
        pick one up first) and the ones running are waited for before the exception propagates,
        so no work of the batch outlives the iteration.
        """
        futures = [self._executor.submit(function, item) for item in items]
        return self._results(futures)

    @staticmethod
    def _results[R](futures: list[Future[R]]) -> Iterator[R]:
        try:
            for future in futures:
                yield future.result()
        finally:
            pending = [future for future in futures if not future.cancel()]
            wait(pending)

    def __enter__(self) -> Self:
        """Return the dispatcher itself."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Wait for the running calls and release the threads."""
        self._executor.shutdown(wait=True)
