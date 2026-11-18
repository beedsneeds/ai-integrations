"""Background model calls that a retried model activity attempt continues.

A non-streaming model call that sets ``background=True`` in ``ModelSettings.extra_args``
or ``extra_body`` asks OpenAI to run the response as a job and answer at once with its
ID. The model activity records that ID in its heartbeat details and waits for the job.
A retried attempt reads the ID and waits for the same job instead of starting another,
so an interrupted attempt is not paid for twice. A job that no later attempt will read
is cancelled.

The Agents SDK returns a background response as soon as OpenAI accepts it, unfinished.
Waiting for it here is what lets a retry continue it, since Temporal owns retries.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from openai import (
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
)
from openai.resources.responses import AsyncResponses
from openai.types.responses import Response

from temporalio import activity
from temporalio.api.enums.v1 import PendingActivityState
from temporalio.client import WorkflowExecutionStatus
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._heartbeat_decorator import set_heartbeat_details
from temporalio.service import RPCError, RPCStatusCode

# The heartbeat details key that carries a background response's ID to the next
# attempt. Attempts on newer workers read what older workers wrote, so keep it stable.
_DETAILS_KEY = "openai_background_response_id"

UNSUPPORTED_ERROR_TYPE = "OpenAIBackgroundUnsupported"
_EXPIRED_ERROR_TYPE = "OpenAIBackgroundResponseExpired"
_FAILED_ERROR_TYPE = "OpenAIBackgroundResponseFailed"
_CANCELLED_ERROR_TYPE = "OpenAIBackgroundResponseCancelled"

# Error codes of a failed response that another attempt can get past. The other
# codes describe the request itself, such as a blocked prompt or an invalid image,
# so a new response for the same request would fail the same way.
_TRANSIENT_ERROR_CODES = frozenset(
    {"server_error", "rate_limit_exceeded", "vector_store_timeout"}
)
_RUNNING_STATUSES = frozenset({"queued", "in_progress"})

# Request options that retrieve and cancel repeat from the create, such as the
# headers a gateway authenticates and the output to include.
_RETRIEVE_OPTIONS = ("include", "extra_headers", "extra_query", "timeout")
_CANCEL_OPTIONS = ("extra_headers", "extra_query", "timeout")

# Seconds a cancel or a describe may take, so that ending the attempt is not held up.
_CANCEL_TIMEOUT = 5.0
_DESCRIBE_TIMEOUT = 5.0
# Seconds a cancelled attempt waits for the reply to a create already sent, so that
# the job OpenAI may have accepted can still be cancelled or continued.
_CREATE_REPLY_TIMEOUT = 10.0


@dataclass(frozen=True)
class _PollSchedule:
    """Seconds between reads of a running background response."""

    initial: float = 1.0
    factor: float = 1.5
    maximum: float = 5.0


_POLL = _PollSchedule()


@dataclass(frozen=True)
class HeartbeatThrottle:
    """How the Worker throttles heartbeats.

    The Worker's ``default_heartbeat_throttle_interval`` and
    ``max_heartbeat_throttle_interval`` options, with the Worker's own defaults.
    """

    default: timedelta = timedelta(seconds=30)
    maximum: timedelta = timedelta(seconds=60)

    def interval(self, heartbeat_timeout: timedelta | None) -> timedelta:
        """How long the Worker can hold a heartbeat other than an attempt's first.

        Computed as Core does, which takes a zero heartbeat timeout for none.
        """
        held = heartbeat_timeout * 0.8 if heartbeat_timeout else self.default
        return min(held, self.maximum)


def requests_background(extra_args: Mapping[str, Any] | None, extra_body: Any) -> bool:
    """Whether model settings ask for a background response."""
    if extra_args and extra_args.get("background") is True:
        return True
    return (
        isinstance(extra_body, Mapping)
        and cast(Mapping[str, Any], extra_body).get("background") is True
    )


def inherited_response_id() -> str | None:
    """The background response an earlier attempt of this activity recorded."""
    for detail in activity.info().heartbeat_details:
        if isinstance(detail, Mapping):
            value = cast(Mapping[str, Any], detail).get(_DETAILS_KEY)
            if isinstance(value, str) and value:
                return value
    return None


def _record(response_id: str) -> bool:
    """Heartbeat the ID. Returns whether the worker sends it at once."""
    return set_heartbeat_details({_DETAILS_KEY: response_id})


def _is_transient_failure(response: Response) -> bool:
    error = response.error
    return error is None or error.code in _TRANSIENT_ERROR_CODES


def _permanent_failure(response: Response) -> ApplicationError:
    error = response.error
    code = error.code if error is not None else None
    message = error.message if error is not None else None
    return ApplicationError(
        f"OpenAI background response failed with the non-retryable error code "
        f"{code}: {message}",
        type=_FAILED_ERROR_TYPE,
        non_retryable=True,
    )


class BackgroundCall:
    """One model activity attempt's background response.

    Holds the ID the attempt continues or started, and whether the response may still
    be running, so that the attempt can cancel a response no later attempt will read.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        inherited_id: str | None,
        *,
        seed: bool = True,
        throttle: HeartbeatThrottle | None = None,
    ) -> None:
        """Start the attempt, continuing ``inherited_id`` if an earlier attempt left one.

        With ``seed``, the attempt's heartbeats carry ``inherited_id`` on to the next.
        ``throttle`` is the Worker's heartbeat throttling, which decides when an ID
        this attempt records has reached the server.
        """
        self._throttle = throttle or HeartbeatThrottle()
        self._responses: AsyncResponses = client.responses
        self._request_options: dict[str, Any] = {}
        self.response_id = inherited_id
        self._may_be_running = inherited_id is not None
        # Whether the Agents SDK asked for the response through this attempt.
        self.intercepted = False
        # Whether response_id came from an earlier attempt, so the server holds it.
        self._id_inherited = inherited_id is not None
        # Event loop time this attempt recorded the ID of a response it started.
        self._id_recorded_at: float | None = None
        # Whether the worker sent that heartbeat at once instead of throttling it.
        self._id_sent_at_once = False
        # Event loop time at which a cancel interrupted the create, if one did.
        self.cancelled_at: float | None = None
        if inherited_id is not None and seed:
            # Heartbeat now: the timer's next heartbeat would otherwise send no ID,
            # and a failure before it would carry none to the next attempt.
            _record(inherited_id)

    async def create(
        self, responses: AsyncResponses, kwargs: dict[str, Any]
    ) -> Response:
        """Continue the earlier attempt's response, or start one, and wait for it."""
        if self.intercepted:
            # The attempt records one response ID, so a second would be lost.
            raise ApplicationError(
                "background=True supports one Responses API request per model call",
                type=UNSUPPORTED_ERROR_TYPE,
                non_retryable=True,
            )
        self.intercepted = True
        self._responses = responses
        self._request_options = {
            key: kwargs[key] for key in _RETRIEVE_OPTIONS if key in kwargs
        }
        if self.response_id is not None:
            response = await self._reattach()
            if response is not None:
                return await self._wait(response)
        if activity.cancellation_details() is not None:
            # A cancel already reached this attempt: start no response it would leave.
            raise asyncio.CancelledError()
        response = await self._start(kwargs)
        return await self._wait(response)

    async def _retrieve(self, response_id: str) -> Response:
        return await cast(Any, self._responses).retrieve(
            response_id, **self._request_options
        )

    async def _reattach(self) -> Response | None:
        response_id = self.response_id
        assert response_id is not None
        try:
            response = await self._retrieve(response_id)
        except NotFoundError:
            self._may_be_running = False
            return None
        except (AuthenticationError, PermissionDeniedError):
            activity.logger.warning(
                "This worker can't read background response %s from an earlier "
                "attempt, so it starts a new one and can't cancel the earlier one",
                response_id,
                exc_info=True,
            )
            self._may_be_running = False
            return None
        status = response.status
        if status in _RUNNING_STATUSES or status == "completed":
            return response
        self._may_be_running = False
        if status == "failed" and not _is_transient_failure(response):
            # The same request would fail the same way: fail as the attempt that
            # saw it fail did, without paying for another response.
            raise _permanent_failure(response)
        return None

    async def _start(self, kwargs: dict[str, Any]) -> Response:
        create = asyncio.ensure_future(cast(Any, self._responses).create(**kwargs))
        try:
            response = cast(Response, await asyncio.shield(create))
        except asyncio.CancelledError:
            self.cancelled_at = asyncio.get_running_loop().time()
            await self._record_late_reply(create)
            raise
        self._started(response.id)
        return response

    async def _record_late_reply(self, create: asyncio.Future[Any]) -> None:
        """After a cancel, wait for a create already sent, to learn the job's ID."""
        try:
            response = cast(
                Response,
                await asyncio.wait_for(asyncio.shield(create), _CREATE_REPLY_TIMEOUT),
            )
        except asyncio.TimeoutError:
            create.cancel()
            activity.logger.warning(
                "A background response OpenAI may have started got no reply before "
                "this attempt ended, so it can be neither cancelled nor continued"
            )
            return
        except Exception:
            # The create failed, so OpenAI started no response.
            return
        self._started(response.id)

    def _started(self, response_id: str) -> None:
        self.response_id = response_id
        self._may_be_running = True
        self._id_inherited = False
        self._id_recorded_at = asyncio.get_running_loop().time()
        # The worker sends an attempt's first heartbeat at once.
        self._id_sent_at_once = _record(response_id)
        activity.logger.debug("Started background response %s", response_id)

    def id_reached_server(self, cancelled_at: float) -> bool:
        """Whether a retry after a cancel at ``cancelled_at`` can read the response ID.

        Once the worker knows an attempt is gone, it drops the heartbeats it has not
        sent, so a later attempt finds only an ID sent before then.
        """
        if self._id_inherited:
            return True
        recorded_at = self._id_recorded_at
        if recorded_at is None or recorded_at > cancelled_at:
            return False
        held = self._throttle.interval(activity.info().heartbeat_timeout)
        return (
            self._id_sent_at_once or cancelled_at - recorded_at >= held.total_seconds()
        )

    async def _wait(self, response: Response) -> Response:
        delay = _POLL.initial
        while response.status in _RUNNING_STATUSES:
            await asyncio.sleep(delay)
            delay = min(delay * _POLL.factor, _POLL.maximum)
            try:
                response = await self._retrieve(response.id)
            except NotFoundError as err:
                self._may_be_running = False
                raise ApplicationError(
                    "The OpenAI background response was deleted before it finished. "
                    "Without store=True, OpenAI keeps a background response only for "
                    "a limited time; set ModelSettings(store=True) for calls that can "
                    "run longer.",
                    type=_EXPIRED_ERROR_TYPE,
                    non_retryable=True,
                ) from err
        self._may_be_running = False
        if response.status == "failed" and not _is_transient_failure(response):
            raise _permanent_failure(response)
        if response.status == "cancelled":
            # Not this attempt's cancel: the next attempt starts a new response.
            raise ApplicationError(
                "The OpenAI background response was cancelled before it finished",
                type=_CANCELLED_ERROR_TYPE,
            )
        # Completed, incomplete, or failed in a way a new response can get past:
        # the Agents SDK handles it as in a native run.
        return response

    def adopt(self, response_id: str) -> None:
        """Track a response this attempt did not see created, so it can be cancelled."""
        self.response_id = response_id
        self._may_be_running = True

    async def cancel(self, reason: str) -> None:
        """Cancel the response if it may still be running. Never raises."""
        response_id = self.response_id
        if response_id is None or not self._may_be_running:
            return
        self._may_be_running = False
        options = {
            key: value
            for key, value in self._request_options.items()
            if key in _CANCEL_OPTIONS
        }
        try:
            await asyncio.wait_for(
                self._responses.cancel(response_id, **options), _CANCEL_TIMEOUT
            )
        except Exception:
            activity.logger.warning(
                "Could not cancel background response %s (%s); it may run to the "
                "end and be billed",
                response_id,
                reason,
                exc_info=True,
            )
        else:
            activity.logger.debug(
                "Cancelled background response %s (%s)", response_id, reason
            )


_current_call: ContextVar[BackgroundCall | None] = ContextVar(
    "_current_call", default=None
)


@contextmanager
def attempt_call(call: BackgroundCall) -> Generator[None]:
    """Route the background create the Agents SDK makes in this block to ``call``."""
    token = _current_call.set(call)
    try:
        yield
    finally:
        _current_call.reset(token)


def _create_requests_background(kwargs: Mapping[str, Any]) -> bool:
    return kwargs.get("stream") is not True and requests_background(
        {"background": kwargs.get("background")}, kwargs.get("extra_body")
    )


class _BackgroundResponses:
    """``client.responses`` with background creates routed to the current attempt."""

    def __init__(self, responses: AsyncResponses) -> None:
        self._responses = responses

    def __getattr__(self, name: str) -> Any:
        if name == "_responses":
            # Not set yet, as on an object copied without __init__.
            raise AttributeError(name)
        return getattr(self._responses, name)

    async def create(self, **kwargs: Any) -> Any:
        call = _current_call.get()
        if call is None or not _create_requests_background(kwargs):
            return await cast(Any, self._responses).create(**kwargs)
        return await call.create(self._responses, kwargs)


class BackgroundOpenAIClient:
    """Delegates to an OpenAI client, except background ``responses.create`` calls.

    Uses the client object it wraps for every request, so the application's client
    keeps its own settings. A background create made inside ``attempt_call`` goes to
    that attempt's :class:`BackgroundCall`; every other call goes to the client as is.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        """Wrap ``client``."""
        self._client = client
        self._responses: _BackgroundResponses | None = None

    @property
    def responses(self) -> _BackgroundResponses:
        """The client's ``responses``, read only once a call needs it."""
        if self._responses is None:
            self._responses = _BackgroundResponses(self._client.responses)
        return self._responses

    def __getattr__(self, name: str) -> Any:
        """Read any other attribute, such as ``base_url``, from the client."""
        if name in ("_client", "_responses"):
            # Not set yet, as on an object copied without __init__.
            raise AttributeError(name)
        return getattr(self._client, name)

    def with_options(self, **kwargs: Any) -> BackgroundOpenAIClient:
        """Wrap the client's copy with other options, as the client's own method does.

        The Agents SDK copies its client this way when it turns provider retries off,
        and a copy left unwrapped would skip the background routing.
        """
        return BackgroundOpenAIClient(self._client.with_options(**kwargs))

    # The client's own copy is an alias of with_options.
    copy = with_options


def _next_retry_delay(override: timedelta | None) -> timedelta:
    """How long the server waits before the next attempt, as its retry policy says."""
    if override is not None:
        return override
    info = activity.info()
    policy = info.retry_policy
    if policy is None:
        return timedelta(0)
    initial = policy.initial_interval.total_seconds()
    # The server's default maximum interval is 100 times the initial interval.
    maximum = (
        policy.maximum_interval.total_seconds()
        if policy.maximum_interval
        else initial * 100
    )
    # Bounded so a long run of attempts can't overflow the power.
    exponent = min(max(info.attempt - 1, 0), 64)
    return timedelta(
        seconds=min(initial * policy.backoff_coefficient**exponent, maximum)
    )


def _no_retry_left(next_delay: timedelta | None = None) -> bool:
    """Whether Temporal will not run another attempt of this activity.

    ``next_delay`` is the failure's own next retry delay, if it sets one.
    """
    info = activity.info()
    policy = info.retry_policy
    if (
        policy is not None
        and policy.maximum_attempts > 0
        and info.attempt >= policy.maximum_attempts
    ):
        return True
    if info.schedule_to_close_timeout:
        scheduled = info.scheduled_time
        if scheduled.tzinfo is None:
            scheduled = scheduled.replace(tzinfo=timezone.utc)
        # The server does not start an attempt after the schedule-to-close deadline.
        next_start = datetime.now(timezone.utc) + _next_retry_delay(next_delay)
        if next_start >= scheduled + info.schedule_to_close_timeout:
            return True
    return False


def is_final_failure(err: BaseException) -> bool:
    """Whether Temporal will not retry this activity after it fails with ``err``."""
    if isinstance(err, ApplicationError) and err.non_retryable:
        return True
    policy = activity.info().retry_policy
    error_type = (err.type if isinstance(err, ApplicationError) else None) or type(
        err
    ).__name__
    if policy is not None and error_type in (policy.non_retryable_error_types or ()):
        return True
    return _no_retry_left(
        err.next_retry_delay if isinstance(err, ApplicationError) else None
    )


async def _activity_still_pending() -> bool | None:
    """Whether the server still runs this activity, None if it can't be told."""
    info = activity.info()
    if info.workflow_id is None:
        # A standalone activity has no workflow to describe.
        return None
    try:
        handle = activity.client().get_workflow_handle(
            info.workflow_id, run_id=info.workflow_run_id
        )
        description = await asyncio.wait_for(handle.describe(), _DESCRIBE_TIMEOUT)
    except Exception as err:
        if isinstance(err, RPCError) and err.status == RPCStatusCode.NOT_FOUND:
            # The workflow run is gone, so no attempt follows.
            return False
        activity.logger.warning(
            "Could not describe the workflow, so the background response is left "
            "running for a retry that may not come. The worker's client needs "
            "permission to describe workflows in this namespace.",
            exc_info=True,
        )
        return None
    if description.status != WorkflowExecutionStatus.RUNNING:
        return False
    for pending in description.raw_description.pending_activities:
        if pending.activity_id == info.activity_id:
            return (
                pending.state
                != PendingActivityState.PENDING_ACTIVITY_STATE_CANCEL_REQUESTED
            )
    return False


async def cancel_reason(call: BackgroundCall) -> str | None:
    """Why a cancelled attempt's response should be cancelled, None to leave it.

    A response is left running only for a later attempt to continue.
    """
    details = activity.cancellation_details()
    if details is None:
        return None
    if details.cancel_requested:
        return "the workflow cancelled the call"
    if details.reset:
        return "the activity was reset"
    if details.paused:
        return None
    if details.worker_shutdown:
        # The worker reports the attempt's last heartbeat details with its failure.
        return "no attempt follows" if _no_retry_left() else None
    # A timeout can also arrive as not_found, so check it first.
    if details.timed_out or details.not_found:
        if _no_retry_left():
            return "no attempt follows"
        cancelled_at = call.cancelled_at
        if cancelled_at is None:
            cancelled_at = asyncio.get_running_loop().time()
        if not call.id_reached_server(cancelled_at):
            return "its ID never reached the server, so no attempt can continue it"
        if details.timed_out:
            return None
        # The workflow may have closed, or the attempt timed out with a retry to follow.
        pending = await _activity_still_pending()
        return "the workflow no longer runs the call" if pending is False else None
    return None
