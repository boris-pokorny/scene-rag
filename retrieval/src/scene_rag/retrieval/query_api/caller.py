"""Who is asking: the caller's entitlements, carried from the transport to the use cases.

The API gateway asserts the caller's identity and passes the entitlements in one trusted
header; this module parses it once per request, in a middleware below every adapter, into a
context variable the REST routes, the MCP tools and the MCP resources read. The stdio
transport has no headers: `bind_caller` sets the variable once for the process. Replacing
the header with a token is a change to this module alone.

The middleware is a plain ASGI callable, so it depends on no framework.
"""

from collections.abc import Awaitable, Callable, MutableMapping
from contextvars import ContextVar, Token
from typing import Any, Final

from scene_rag.retrieval.query_api.models import ANONYMOUS, Caller

ENTITLEMENTS_HEADER: Final = "X-Scene-Rag-Entitlements"
"""Comma-separated entitlements; absent or empty means the caller sees nothing."""

# ASGI messages are untyped dictionaries; `Any` is confined to this boundary.
type Scope = MutableMapping[str, Any]
type Message = MutableMapping[str, Any]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]
type App = Callable[[Scope, Receive, Send], Awaitable[None]]

_CALLER: ContextVar[Caller] = ContextVar("scene_rag_caller", default=ANONYMOUS)
_HEADER_NAME: Final = ENTITLEMENTS_HEADER.lower().encode()


def parse_entitlements(value: str | None) -> Caller:
    """Parse the header value: comma-separated names, trimmed, empties dropped."""
    if value is None:
        return ANONYMOUS
    names = frozenset(item.strip() for item in value.split(","))
    return Caller(names - {""})


def current_caller() -> Caller:
    """Return the caller of the request in progress; anonymous outside any."""
    return _CALLER.get()


def bind_caller(caller: Caller) -> Token[Caller]:
    """Make ``caller`` the current one, for a transport that has no per-request headers."""
    return _CALLER.set(caller)


def reset_caller(token: Token[Caller]) -> None:
    """Undo a `bind_caller`."""
    _CALLER.reset(token)


class CallerMiddleware:
    """ASGI middleware binding the caller of every HTTP request from the trusted header."""

    def __init__(self, app: App) -> None:
        """Wrap ``app``."""
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Bind the caller for the request, run the application, unbind."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        value: str | None = None
        for name, raw in scope.get("headers", ()):
            if name == _HEADER_NAME:
                value = raw.decode("latin-1")
                break
        token = _CALLER.set(parse_entitlements(value))
        try:
            await self._app(scope, receive, send)
        finally:
            _CALLER.reset(token)
