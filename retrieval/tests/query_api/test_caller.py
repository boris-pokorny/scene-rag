"""The trusted entitlements header, its parsing and its binding per request."""

import asyncio
from collections.abc import Awaitable

import pytest

from scene_rag.retrieval.query_api import caller
from scene_rag.retrieval.query_api.caller import (
    CallerMiddleware,
    bind_caller,
    current_caller,
    parse_entitlements,
    reset_caller,
)
from scene_rag.retrieval.query_api.models import ANONYMOUS, Caller


def run[T](awaitable: Awaitable[T]) -> T:
    return asyncio.run(_await(awaitable))


async def _await[T](awaitable: Awaitable[T]) -> T:
    return await awaitable


class TestGivenHeaderValues:
    """Values the gateway may send."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, frozenset()),
            ("", frozenset()),
            (" , ,", frozenset()),
            ("public", {"public"}),
            ("public, team-platform ,public", {"public", "team-platform"}),
        ],
    )
    def test_when_parsed_then_trimmed_deduplicated(
        self, value: str | None, expected: frozenset[str]
    ) -> None:
        assert parse_entitlements(value) == Caller(frozenset(expected))


class TestGivenNoRequest:
    """Outside any request, or on a transport without headers."""

    def test_when_nothing_bound_then_anonymous(self) -> None:
        assert current_caller() == ANONYMOUS

    def test_when_bound_then_current_until_reset(self) -> None:
        token = bind_caller(Caller(frozenset({"a"})))
        try:
            assert current_caller() == Caller(frozenset({"a"}))
        finally:
            reset_caller(token)
        assert current_caller() == ANONYMOUS


class TestGivenMiddleware:
    """The middleware around a recording application."""

    def test_when_header_present_then_bound_during_request_only(self) -> None:
        seen: list[Caller] = []

        async def app(scope: caller.Scope, receive: caller.Receive, send: caller.Send) -> None:
            seen.append(current_caller())

        scope = {"type": "http", "headers": [(b"x-scene-rag-entitlements", b"a, b")]}
        run(CallerMiddleware(app)(scope, _receive, _send))

        assert seen == [Caller(frozenset({"a", "b"}))]
        assert current_caller() == ANONYMOUS

    def test_when_header_absent_then_anonymous(self) -> None:
        seen: list[Caller] = []

        async def app(scope: caller.Scope, receive: caller.Receive, send: caller.Send) -> None:
            seen.append(current_caller())

        run(CallerMiddleware(app)({"type": "http", "headers": []}, _receive, _send))

        assert seen == [ANONYMOUS]

    def test_when_not_http_then_passed_through(self) -> None:
        seen: list[str] = []

        async def app(scope: caller.Scope, receive: caller.Receive, send: caller.Send) -> None:
            seen.append(scope["type"])

        run(CallerMiddleware(app)({"type": "lifespan"}, _receive, _send))

        assert seen == ["lifespan"]

    def test_when_application_raises_then_unbound(self) -> None:
        async def app(scope: caller.Scope, receive: caller.Receive, send: caller.Send) -> None:
            raise RuntimeError("boom")

        scope = {"type": "http", "headers": [(b"x-scene-rag-entitlements", b"a")]}
        with pytest.raises(RuntimeError):
            run(CallerMiddleware(app)(scope, _receive, _send))

        assert current_caller() == ANONYMOUS


async def _receive() -> caller.Message:
    return {"type": "http.request"}


async def _send(message: caller.Message) -> None:
    return None
