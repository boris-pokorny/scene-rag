"""Concurrency policies: results in item order, failures stop the batch."""

import threading

import pytest

from scene_rag.retrieval.chunking.dispatch import SerialDispatcher, ThreadDispatcher


def square(value: int) -> int:
    return value * value


class TestGivenSerialDispatcher:
    """One after another on the calling thread."""

    def test_when_mapped_then_results_in_order(self) -> None:
        assert list(SerialDispatcher().map(square, [3, 1, 2])) == [9, 1, 4]


class TestGivenThreadDispatcher:
    """A pool of threads."""

    def test_when_mapped_then_results_in_item_order(self) -> None:
        with ThreadDispatcher(3, thread_name_prefix="test") as pool:
            assert pool.workers == 3
            assert list(pool.map(square, range(10))) == [i * i for i in range(10)]

    def test_when_one_call_raises_then_propagates_and_rest_finish_or_cancel(self) -> None:
        started: list[int] = []
        lock = threading.Lock()

        def work(value: int) -> int:
            with lock:
                started.append(value)
            if value == 1:
                raise ValueError("boom")
            return value

        with ThreadDispatcher(2) as pool, pytest.raises(ValueError, match="boom"):
            list(pool.map(work, range(50)))

        assert 1 in started

    def test_when_zero_workers_then_value_error(self) -> None:
        with pytest.raises(ValueError, match="workers"):
            ThreadDispatcher(0)

    def test_when_exited_then_pool_refuses_work(self) -> None:
        with ThreadDispatcher(1) as pool:
            pass

        with pytest.raises(RuntimeError, match="shutdown"):
            list(pool.map(square, [1]))
