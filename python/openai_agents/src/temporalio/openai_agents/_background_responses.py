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
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
)
from openai.resources.responses import AsyncResponses
from openai.types.responses import Response

from temporalio import activity, workflow
from temporalio.api.enums.v1 import PendingActivityState
from temporalio.client import WorkflowExecutionStatus
from temporalio.converter import ActivitySerializationContext
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._heartbeat_decorator import set_heartbeat_details
from temporalio.service import RPCError, RPCStatusCode

BACKGROUND_RESPONSES_PATCH = "temporalio.openai_agents.background-responses"
"""Patch ID that gates waiting for, continuing and refusing background model calls.

Runs that started an agent run before the patch keep the earlier behavior: a background
call returns its response unfinished, and a streamed one is not refused.
"""


def background_responses_enabled() -> bool:
    """Return whether this workflow run's background model calls get the plugin's handling.

    ``Runner.run`` and ``Runner.run_streamed`` call this before they schedule any model
    activity, so a run that started an agent run on an earlier release replays the call
    without the patch marker and keeps the earlier behavior. The SDK memoizes the answer
    for the rest of the run.
    """
    return workflow.patched(BACKGROUND_RESPONSES_PATCH)


# The heartbeat details key that carries a background response's ID to the next
# attempt. Attempts on newer workers read what older workers wrote, so keep it stable.
_DETAILS_KEY = "openai_background_response_id"

UNSUPPORTED_ERROR_TYPE = "OpenAIBackgroundUnsupported"
_EXPIRED_ERROR_TYPE = "OpenAIBackgroundResponseExpired"
_FAILED_ERROR_TYPE = "OpenAIBackgroundResponseFailed"
_CANCELLED_ERROR_TYPE = "OpenAIBackgroundResponseCancelled"

# Error codes of a failed response that describe the request itself, such as a
# blocked prompt or an invalid image, so a new response for the same request would
# fail the same way. Other codes, including ones OpenAI adds later, are left to the
# retry policy, as an error without an HTTP status is.
_PERMANENT_ERROR_CODES = frozenset(
    {
        "invalid_prompt",
        "data_residency_mismatch",
        "bio_policy",
        "misalignment_policy_violation",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
    }
)
_RUNNING_STATUSES = frozenset({"queued", "in_progress"})

# Request options that retrieve and cancel repeat from the create, such as the
# headers a gateway authenticates and the output to include.
_RETRIEVE_OPTIONS = ("include", "extra_headers", "extra_query", "timeout")
_CANCEL_OPTIONS = ("extra_headers", "extra_query", "timeout")

# Seconds a cancel or a describe may take, so that ending the attempt is not held up.
_CANCEL_TIMEOUT = 5.0
_DESCRIBE_TIMEOUT = 5.0
# Describe errors that only a permission change fixes, as the stateful MCP session has.
_DESCRIBE_PERMISSION_ERRORS = frozenset(
    {RPCStatusCode.PERMISSION_DENIED, RPCStatusCode.UNAUTHENTICATED}
)
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


def _record(response_id: str) -> None:
    """Heartbeat the ID, and keep it in every later heartbeat of the attempt."""
    set_heartbeat_details({_DETAILS_KEY: response_id})


def _is_permanent_failure(response: Response) -> bool:
    error = response.error
    return error is not None and error.code in _PERMANENT_ERROR_CODES


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
    ) -> None:
        """Start the attempt, continuing ``inherited_id`` if an earlier attempt left one.

        With ``seed``, the attempt's heartbeats carry ``inherited_id`` on to the next.
        """
        self._responses: AsyncResponses = client.responses
        self._request_options: dict[str, Any] = {}
        self.response_id = inherited_id
        self._may_be_running = inherited_id is not None
        # Whether the Agents SDK asked for the response through this attempt.
        self.intercepted = False
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
        """Read the response, reading it again after a transient error."""
        delay = _POLL.initial
        while True:
            try:
                return await cast(Any, self._responses).retrieve(
                    response_id, **self._request_options
                )
            except Exception as err:
                wait = _read_again_after(err)
                if wait is None:
                    raise
                pause = max(delay, wait.total_seconds())
                activity.logger.debug(
                    "Could not read background response %s, reading it again in %.1f s",
                    response_id,
                    pause,
                    exc_info=True,
                )
            await asyncio.sleep(pause)
            delay = min(delay * _POLL.factor, _POLL.maximum)

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
        if status == "failed" and _is_permanent_failure(response):
            # The same request would fail the same way: fail as the attempt that
            # saw it fail did, without paying for another response.
            raise _permanent_failure(response)
        return None

    async def _start(self, kwargs: dict[str, Any]) -> Response:
        create = asyncio.ensure_future(cast(Any, self._responses).create(**kwargs))
        try:
            response = cast(Response, await asyncio.shield(create))
        except asyncio.CancelledError:
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
        _record(response_id)
        activity.logger.debug("Started background response %s", response_id)

    @property
    def may_be_running(self) -> bool:
        """Whether the response this attempt tracks may still be running."""
        return self._may_be_running

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
        if response.status == "failed" and _is_permanent_failure(response):
            raise _permanent_failure(response)
        if response.status == "cancelled":
            # Not this attempt's cancel: the next attempt starts a new response.
            raise ApplicationError(
                "The OpenAI background response was cancelled before it finished",
                type=_CANCELLED_ERROR_TYPE,
            )
        # Completed, incomplete, or failed with a code not known to be permanent:
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
    try:
        delay = initial * policy.backoff_coefficient ** max(info.attempt - 1, 0)
    except OverflowError:
        # After enough attempts, the delay passed the maximum long ago.
        delay = maximum
    return timedelta(seconds=min(delay, maximum))


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


def _retryable(err: BaseException) -> bool:
    """Whether the retry policy retries ``err``, if attempts and time remain."""
    if isinstance(err, ApplicationError) and err.non_retryable:
        return False
    policy = activity.info().retry_policy
    error_type = (err.type if isinstance(err, ApplicationError) else None) or type(
        err
    ).__name__
    return policy is None or error_type not in (policy.non_retryable_error_types or ())


def is_final_failure(err: BaseException) -> bool:
    """Whether Temporal will not retry this activity after it fails with ``err``."""
    return not _retryable(err) or _no_retry_left(
        err.next_retry_delay if isinstance(err, ApplicationError) else None
    )


def _read_again_after(err: Exception) -> timedelta | None:
    """How long to wait before reading a response again after ``err``, None to raise.

    Reading resends no request, so the attempt reads again after the errors the OpenAI
    client retries, as Temporal's frequent-polling sample does, and waits at least as
    long as the provider asks. The retry policy's ``non_retryable_error_types`` govern
    the server's retries of the activity, not these reads. Errors the background call
    handles itself, such as a deleted response, and other errors end the attempt.
    """
    if isinstance(err, (NotFoundError, AuthenticationError, PermissionDeniedError)):
        return None
    if isinstance(err, APIConnectionError):
        # Also a timeout.
        return timedelta(0)
    if not isinstance(err, APIStatusError):
        return None
    # Imported here because that module imports this one.
    from temporalio.openai_agents._invoke_model_activity import (
        _raise_for_openai_status,
    )

    try:
        _raise_for_openai_status(err)
    except ApplicationError as translated:
        if translated.non_retryable:
            return None
        return translated.next_retry_delay or timedelta(0)


async def _next_attempt_reads(response_id: str) -> bool | None:
    """Whether a later attempt will read ``response_id``, None if the server can't tell.

    After an attempt times out, the server has usually decided whether to retry it: the
    activity is pending at another attempt, or is gone. Its heartbeat details are what
    that attempt starts with, read back as the stateful MCP session reads back its own.
    The answer is None while the server still shows this attempt, or when it can't be
    asked. A heartbeat still on its way can land after the read; the response is then
    cancelled although a retry could have continued it, which costs a new response but
    leaves none running.
    """
    info = activity.info()
    if info.workflow_id is None:
        # A standalone activity has no workflow to describe.
        return None
    workflow_id = info.workflow_id

    async def read_back() -> bool | None:
        client = activity.client()
        handle = client.get_workflow_handle(workflow_id, run_id=info.workflow_run_id)
        description = await handle.describe()
        if description.status != WorkflowExecutionStatus.RUNNING:
            return False
        pending = next(
            (
                pending
                for pending in description.raw_description.pending_activities
                if pending.activity_id == info.activity_id
            ),
            None,
        )
        if (
            pending is None
            or pending.state
            == PendingActivityState.PENDING_ACTIVITY_STATE_CANCEL_REQUESTED
        ):
            return False
        converter = client.data_converter.with_context(
            ActivitySerializationContext(
                namespace=info.namespace,
                workflow_id=workflow_id,
                workflow_type=info.workflow_type,
                activity_type=info.activity_type,
                activity_id=info.activity_id,
                activity_task_queue=info.task_queue,
                is_local=info.is_local,
            )
        )
        details = await converter.decode(pending.heartbeat_details.payloads)
        if not any(
            isinstance(detail, Mapping)
            and cast(Mapping[str, Any], detail).get(_DETAILS_KEY) == response_id
            for detail in details
        ):
            return False
        # Another attempt means the server retried the activity, or an operator reset
        # it; the same attempt means the server hasn't processed the timeout yet.
        return True if pending.attempt != info.attempt else None

    try:
        return await asyncio.wait_for(read_back(), _DESCRIBE_TIMEOUT)
    except Exception as err:
        if isinstance(err, RPCError) and err.status == RPCStatusCode.NOT_FOUND:
            # The workflow run is gone, so no attempt follows.
            return False
        if isinstance(err, RPCError) and err.status in _DESCRIBE_PERMISSION_ERRORS:
            reason = (
                f"the server refused it ({err.status.name}). Give the worker's client "
                "permission to describe workflows in this namespace"
            )
        elif isinstance(err, asyncio.TimeoutError):
            reason = f"it took longer than {_DESCRIBE_TIMEOUT:g} seconds"
        else:
            reason = "it failed"
        activity.logger.warning(
            "Could not describe the workflow to tell whether a retry will continue "
            "background response %s, so the retry policy decides: %s",
            response_id,
            reason,
            exc_info=True,
        )
        return None


async def cancel_reason(call: BackgroundCall) -> str | None:
    """Why a cancelled attempt's response should be cancelled, None to leave it.

    A response is left running only for a later attempt to continue.
    """
    details = activity.cancellation_details()
    if details is None:
        return None
    if details.cancel_requested:
        return "the workflow cancelled the call"
    if details.reset or details.paused:
        # Both keep the heartbeat details unless the operator asks to clear them, so
        # the next attempt continues the response.
        return None
    if details.worker_shutdown:
        # The worker reports the attempt's last heartbeat details with its failure.
        return "no attempt follows" if _no_retry_left() else None
    # A timeout, or a cancel from a workflow that closed (both can arrive as not_found).
    if details.timed_out or details.not_found:
        if call.response_id is None or not call.may_be_running:
            return None
        # The server has usually decided by now whether to retry, so ask it first.
        reads = await _next_attempt_reads(call.response_id)
        if reads is None:
            return "no attempt follows" if _no_retry_left() else None
        return None if reads else "no later attempt will read its ID"
    return None
