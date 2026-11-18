import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar, cast, overload

from temporalio import activity

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


class _HeartbeatDetails:
    """The details every heartbeat of one ``heartbeating`` block sends."""

    def __init__(self) -> None:
        self.details: tuple[Any, ...] = ()
        # Heartbeats sent so far. The worker sends an attempt's first heartbeat at
        # once and throttles later ones.
        self.sent = 0


# Each heartbeat replaces the details the server holds, so the timer resends the
# block's current details instead of none.
_current_details: ContextVar[_HeartbeatDetails | None] = ContextVar(
    "_current_details", default=None
)


def set_heartbeat_details(*details: Any) -> bool:
    """Heartbeat now with ``details``, and keep sending them on every later heartbeat.

    Inside a ``heartbeating`` block, the block's timer resends them. Outside one, this
    only heartbeats once. Returns whether this was the block's first heartbeat, which
    the worker sends at once rather than throttled; outside a block it returns True.
    """
    box = _current_details.get()
    first = True
    if box is not None:
        box.details = details
        first = box.sent == 0
        box.sent += 1
    activity.heartbeat(*details)
    return first


@overload
def auto_heartbeater(fn: F) -> F: ...


@overload
def auto_heartbeater(*, interval: float) -> Callable[[F], F]: ...


def auto_heartbeater(
    fn: F | None = None, *, interval: float | None = None
) -> F | Callable[[F], F]:
    """Decorator that heartbeats while the activity runs.

    By default it heartbeats at half the activity's heartbeat timeout, and not
    at all without one. With ``interval``, it heartbeats every ``interval``
    seconds whatever the timeout.
    """

    def decorate(fn: F) -> F:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with heartbeating(interval):
                return await fn(*args, **kwargs)

        return cast(F, wrapper)

    return decorate(fn) if fn is not None else decorate


@asynccontextmanager
async def heartbeating(interval: float | None = None) -> AsyncGenerator[None]:
    """Heartbeat in the background while the block runs, as ``auto_heartbeater`` does.

    Stopping the heartbeats on exit awaits the heartbeat task and swallows a cancel
    that arrives meanwhile. An activity that must act on every requested cancel checks
    ``activity.cancellation_details()`` after the block.
    """
    delay = interval
    if delay is None:
        heartbeat_timeout = activity.info().heartbeat_timeout
        if heartbeat_timeout:
            delay = heartbeat_timeout.total_seconds() / 2
    # Set before the timer starts, so the timer's copy of the context shares it.
    details_token = _current_details.set(_HeartbeatDetails())
    heartbeat_task = None
    if delay:
        heartbeat_task = asyncio.create_task(_heartbeat_every(delay))
    try:
        yield
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        _current_details.reset(details_token)


async def _heartbeat_every(delay: float) -> None:
    while True:
        await asyncio.sleep(delay)
        box = _current_details.get()
        if box is None:
            activity.heartbeat()
            continue
        box.sent += 1
        activity.heartbeat(*box.details)
