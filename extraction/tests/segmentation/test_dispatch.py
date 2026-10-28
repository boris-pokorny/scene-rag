"""The concurrency policies: same results in the same order, whichever runs the calls."""

import threading
import time
from collections.abc import Iterator

import pytest

from scene_rag.extraction.segmentation.dispatch import SerialDispatcher, ThreadDispatcher
from scene_rag.extraction.segmentation.protocols import Dispatcher


def square(value: int) -> int:
    if value < 0:
        raise ValueError(f"negative: {value}")
    return value * value


@pytest.fixture(params=["serial", "threads"])
def dispatcher(request: pytest.FixtureRequest) -> Iterator[Dispatcher]:
    """Each policy in turn; a pool is drained after the case."""
    if request.param == "serial":
        yield SerialDispatcher()
        return
    with ThreadDispatcher(3, thread_name_prefix="test") as threads:
        yield threads


class TestGivenEitherPolicy:
    """The contract every `Dispatcher` keeps."""

    def test_when_mapped_then_results_in_item_order(self, dispatcher: Dispatcher) -> None:
        assert list(dispatcher.map(square, [3, 1, 2, 0, 5])) == [9, 1, 4, 0, 25]

    def test_when_function_raises_then_propagates_on_iteration(
        self, dispatcher: Dispatcher
    ) -> None:
        results = dispatcher.map(square, [1, -2, 3])

        assert next(results) == 1
        with pytest.raises(ValueError, match="negative: -2"):
            next(results)

    def test_when_no_items_then_no_results(self, dispatcher: Dispatcher) -> None:
        assert list(dispatcher.map(square, [])) == []


class TestGivenSerialDispatcher:
    """Calls run on the caller's thread, one at a time, as results are consumed."""

    def test_when_mapped_then_runs_on_the_calling_thread(self) -> None:
        threads = list(SerialDispatcher().map(lambda _: threading.get_ident(), range(3)))

        assert set(threads) == {threading.get_ident()}


class TestGivenThreadDispatcher:
    """A pool of threads the dispatcher owns."""

    def test_when_workers_below_one_then_rejected(self) -> None:
        with pytest.raises(ValueError, match="workers"):
            ThreadDispatcher(0)

    def test_when_mapped_then_items_run_at_once(self) -> None:
        barrier = threading.Barrier(2, timeout=5)

        with ThreadDispatcher(2) as dispatcher:
            assert dispatcher.workers == 2
            threads = set(
                dispatcher.map(lambda _: (barrier.wait(), threading.get_ident())[1], "ab")
            )

        assert len(threads) == 2 and threading.get_ident() not in threads

    def test_when_one_call_raises_then_running_calls_finish_before_it_propagates(self) -> None:
        started = threading.Event()
        finished = threading.Event()

        def work(item: str) -> str:
            if item == "raises":
                started.wait(5)
                raise ValueError(item)
            if item == "running":
                started.set()
                time.sleep(0.2)
                finished.set()
            return item

        with ThreadDispatcher(2) as dispatcher:
            with pytest.raises(ValueError, match="raises"):
                list(dispatcher.map(work, ["raises", "running"]))

            assert finished.is_set()

    def test_when_exited_then_pool_refuses_new_work(self) -> None:
        with ThreadDispatcher(1) as dispatcher:
            assert list(dispatcher.map(square, [2])) == [4]

        with pytest.raises(RuntimeError, match="shutdown"):
            list(dispatcher.map(square, [2]))
