"""Background model calls that a retried model activity attempt continues."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import email.utils
import itertools
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import pytest_asyncio
from agents import (
    Agent,
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelProvider,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    OpenAIResponsesModel,
    Runner,
    Tool,
    TResponseInputItem,
    Usage,
)
from agents.exceptions import ModelBehaviorError
from agents.models import _openai_shared
from openai import APIConnectionError, AsyncOpenAI

from temporalio import workflow
from temporalio.activity import ActivityCancellationDetails
from temporalio.api.enums.v1 import EventType, PendingActivityState
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents import (
    ModelActivityParameters,
    OpenAIAgentsPlugin,
    OpenAIPayloadConverter,
    _background_responses,
)
from temporalio.openai_agents._background_responses import (
    _DETAILS_KEY,
    UNSUPPORTED_ERROR_TYPE,
    BackgroundCall,
    BackgroundOpenAIClient,
    HeartbeatThrottle,
    _PollSchedule,
    attempt_call,
)
from temporalio.openai_agents._heartbeat_decorator import (
    heartbeating,
    set_heartbeat_details,
)
from temporalio.openai_agents._invoke_model_activity import (
    ActivityModelInput,
    ModelActivity,
    ModelTracingInput,
)
from temporalio.openai_agents.testing import AgentEnvironment
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import ActivityEnvironment
from temporalio.worker import Worker
from tests.helpers import assert_eventually, new_worker

if TYPE_CHECKING:
    # Imported where used: the workflow sandbox re-imports this module.
    import httpx2

BACKGROUND = ModelSettings(extra_args={"background": True})

CHAT_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Done."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@dataclass
class Job:
    """A background response the fake API runs."""

    id: str
    status: str = "queued"
    # The statuses the job reports on its next reads. It keeps the last one.
    script: list[str] = field(default_factory=lambda: ["in_progress", "completed"])
    # While held, reads report in_progress without advancing the script.
    held: bool = False
    error_code: str = "server_error"
    gone: bool = False

    def read(self) -> None:
        if self.held or self.status in (
            "completed",
            "failed",
            "incomplete",
            "cancelled",
        ):
            return
        if self.script:
            self.status = self.script.pop(0)

    def body(self) -> dict[str, Any]:
        completed = self.status == "completed"
        return {
            "id": self.id,
            "object": "response",
            "created_at": 0,
            "model": "gpt-4o",
            "status": self.status,
            "background": True,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": (
                [
                    {
                        "type": "message",
                        "id": f"msg_{self.id}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Done.", "annotations": []}
                        ],
                    }
                ]
                if completed
                else []
            ),
            "error": (
                {"code": self.error_code, "message": "The response failed."}
                if self.status == "failed"
                else None
            ),
            "incomplete_details": (
                {"reason": "max_output_tokens"} if self.status == "incomplete" else None
            ),
            "usage": (
                {
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                }
                if completed
                else None
            ),
        }


class FakeResponsesAPI:
    """An in-memory OpenAI Responses API that runs background responses as jobs."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.creates: list[dict[str, Any]] = []
        self.retrieves: list[httpx2.Request] = []
        self.cancels: list[str] = []
        # Status codes to answer the next retrieves and cancels with, alone or with
        # response headers. 0 drops the connection.
        self.retrieve_errors: list[int | tuple[int, dict[str, str]]] = []
        self.cancel_errors: list[int | tuple[int, dict[str, str]]] = []
        self.chat_completions = 0
        # Applied to each new background job.
        self.script: list[str] = ["in_progress", "completed"]
        self.error_code = "server_error"
        self.hold = False
        # Seconds between accepting a create and replying to it.
        self.create_delay = 0.0
        self._ids = itertools.count(1)
        import httpx2

        self.client = AsyncOpenAI(
            api_key="not-a-real-key",
            base_url="https://openai.invalid/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
            max_retries=0,
        )

    def add_job(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job

    @property
    def last_job(self) -> Job:
        return list(self.jobs.values())[-1]

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        import httpx2

        def error(reply: int | tuple[int, dict[str, str]]) -> httpx2.Response:
            status, headers = reply if isinstance(reply, tuple) else (reply, {})
            if status == 0:
                raise httpx2.ConnectError("connection dropped", request=request)
            return httpx2.Response(
                status,
                headers=headers,
                json={"error": {"message": "error", "type": "error", "code": None}},
            )

        assert request.url.host == "openai.invalid"
        path = request.url.path
        if request.method == "POST" and path == "/v1/chat/completions":
            self.chat_completions += 1
            return httpx2.Response(200, json=CHAT_COMPLETION)
        if request.method == "POST" and path == "/v1/responses":
            body = json.loads(request.content)
            self.creates.append(body)
            created = Job(
                id=f"resp_{next(self._ids)}",
                script=list(self.script),
                error_code=self.error_code,
            )
            if not body.get("background"):
                created.status = "completed"
            created.held = self.hold
            self.add_job(created)
            if self.create_delay:
                await asyncio.sleep(self.create_delay)
            return httpx2.Response(200, json=created.body())
        match = re.fullmatch(r"/v1/responses/([^/]+)(/cancel)?", path)
        assert match, f"unexpected request {request.method} {path}"
        job = self.jobs.get(match[1])
        if match[2]:
            self.cancels.append(match[1])
            if self.cancel_errors:
                return error(self.cancel_errors.pop(0))
            if job is not None and job.status in ("queued", "in_progress"):
                job.status = "cancelled"
            return httpx2.Response(200, json=(job or Job(id=match[1])).body())
        self.retrieves.append(request)
        if self.retrieve_errors:
            return error(self.retrieve_errors.pop(0))
        if job is None or job.gone:
            return httpx2.Response(
                404,
                json={"error": {"message": "not found", "type": "invalid_request"}},
            )
        job.read()
        return httpx2.Response(200, json=job.body())


@pytest_asyncio.fixture
async def api(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeResponsesAPI]:
    """The fake API, installed as the framework's default client."""
    fake = FakeResponsesAPI()
    monkeypatch.setattr(_openai_shared, "_default_openai_client", fake.client)
    monkeypatch.setattr(_openai_shared, "_use_responses_by_default", True)
    monkeypatch.setattr(_openai_shared, "_default_openai_responses_transport", "http")
    monkeypatch.setattr(_background_responses, "_POLL", _PollSchedule(0.01, 1, 0.01))
    yield fake
    await fake.client.close()


def model_input(
    settings: ModelSettings = BACKGROUND, conversation_id: str | None = None
) -> ActivityModelInput:
    return ActivityModelInput(
        model_name="gpt-4o",
        input="Hi",
        model_settings=settings,
        tracing=ModelTracingInput.DISABLED,
        tools=[],
        handoffs=[],
        conversation_id=conversation_id,
    )


def activity_env(
    *,
    heartbeat_details: list[Any] | None = None,
    attempt: int = 1,
    retry_policy: RetryPolicy | None = None,
    maximum_attempts: int = 0,
    schedule_to_close_timeout: timedelta | None = None,
    heartbeat_timeout: timedelta | None = None,
    is_local: bool = False,
    client: Any = None,
) -> tuple[ActivityEnvironment, list[tuple[Any, ...]]]:
    env = ActivityEnvironment(client)
    env.info = dataclasses.replace(
        env.info,
        heartbeat_details=heartbeat_details or [],
        attempt=attempt,
        retry_policy=retry_policy or RetryPolicy(maximum_attempts=maximum_attempts),
        # The default info's deadline is long past, which would make every attempt
        # final, so tests that need a deadline set one from now.
        schedule_to_close_timeout=schedule_to_close_timeout,
        scheduled_time=datetime.now(timezone.utc),
        heartbeat_timeout=heartbeat_timeout,
        is_local=is_local,
    )
    heartbeats: list[tuple[Any, ...]] = []
    env.on_heartbeat = lambda *details: heartbeats.append(details)
    return env, heartbeats


@contextlib.asynccontextmanager
async def running(
    env: ActivityEnvironment, model_activity: ModelActivity | None = None
) -> AsyncGenerator[asyncio.Task[ModelResponse]]:
    """Run the model activity in a task that is cancelled if the test ends early."""
    task = asyncio.create_task(
        env.run(
            (model_activity or ModelActivity()).invoke_model_activity, model_input()
        )
    )
    try:
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


def text(response: ModelResponse) -> str:
    item: Any = response.output[0]
    return item.content[0].text


def inherited(response_id: str) -> list[Any]:
    return [{_DETAILS_KEY: response_id}]


def polling(api: FakeResponsesAPI) -> Callable[[], Awaitable[None]]:
    async def check() -> None:
        assert api.retrieves

    return check


def sending(api: FakeResponsesAPI) -> Callable[[], Awaitable[None]]:
    async def check() -> None:
        assert api.creates

    return check


# Heartbeat details


async def test_heartbeat_timer_resends_the_details() -> None:
    env, heartbeats = activity_env(heartbeat_timeout=timedelta(milliseconds=20))

    async def body() -> None:
        async with heartbeating():
            await asyncio.sleep(0.05)
            set_heartbeat_details("marker")
            await asyncio.sleep(0.1)

    await env.run(body)
    marked = heartbeats.index(("marker",))
    # Before the details, the timer sends none, as it always has.
    assert set(heartbeats[:marked]) == {()}
    # After them, every timer heartbeat carries them instead of erasing them.
    assert len(heartbeats) > marked + 1
    assert set(heartbeats[marked:]) == {("marker",)}


async def test_set_heartbeat_details_outside_a_block_heartbeats_once() -> None:
    env, heartbeats = activity_env()

    async def body() -> None:
        set_heartbeat_details("marker")

    await env.run(body)
    assert heartbeats == [("marker",)]


def test_heartbeat_details_survive_the_payload_converter() -> None:
    converter = OpenAIPayloadConverter()
    details = {_DETAILS_KEY: "resp_1"}
    assert converter.from_payloads(converter.to_payloads([details])) == [details]


@pytest.mark.parametrize(
    ("throttle", "heartbeat_timeout", "interval"),
    [
        (HeartbeatThrottle(), timedelta(seconds=10), timedelta(seconds=8)),
        (HeartbeatThrottle(), None, timedelta(seconds=30)),
        # Core takes a zero heartbeat timeout for none.
        (HeartbeatThrottle(), timedelta(0), timedelta(seconds=30)),
        (
            HeartbeatThrottle(maximum=timedelta(seconds=5)),
            timedelta(seconds=10),
            timedelta(seconds=5),
        ),
        (
            HeartbeatThrottle(default=timedelta(seconds=90)),
            None,
            timedelta(seconds=60),
        ),
    ],
    ids=[
        "heartbeat-timeout",
        "default",
        "zero-heartbeat-timeout",
        "maximum-caps-heartbeat-timeout",
        "maximum-caps-default",
    ],
)
def test_heartbeat_throttle_is_computed_as_the_worker_does(
    throttle: HeartbeatThrottle,
    heartbeat_timeout: timedelta | None,
    interval: timedelta,
) -> None:
    assert throttle.interval(heartbeat_timeout) == interval


# Starting and waiting


async def test_background_call_waits_for_the_response(api: FakeResponsesAPI) -> None:
    env, heartbeats = activity_env()
    response = await env.run(ModelActivity().invoke_model_activity, model_input())

    assert text(response) == "Done."
    assert response.response_id == "resp_1"
    assert response.usage.total_tokens == 8
    assert [create["background"] for create in api.creates] == [True]
    assert len(api.retrieves) == 2
    assert api.cancels == []
    # The ID reaches the server as soon as OpenAI names the response.
    assert heartbeats[0] == ({_DETAILS_KEY: "resp_1"},)


async def test_extra_body_background_waits_for_the_response(
    api: FakeResponsesAPI,
) -> None:
    env, _ = activity_env()
    settings = ModelSettings(extra_body={"background": True})
    response = await env.run(
        ModelActivity().invoke_model_activity, model_input(settings)
    )
    assert text(response) == "Done."
    assert len(api.retrieves) == 2


async def test_call_without_background_is_unchanged(
    api: FakeResponsesAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapped: list[str] = []
    original = _background_responses._BackgroundResponses.create

    async def recording_create(self: Any, **kwargs: Any) -> Any:
        wrapped.append("create")
        return await original(self, **kwargs)

    monkeypatch.setattr(
        _background_responses._BackgroundResponses, "create", recording_create
    )
    env, heartbeats = activity_env()
    response = await env.run(
        ModelActivity().invoke_model_activity, model_input(ModelSettings())
    )
    assert text(response) == "Done."
    assert "background" not in api.creates[0]
    assert api.retrieves == []
    assert heartbeats == []
    # The call went to the provider the plugin has always used, not the wrapper.
    assert wrapped == []


async def test_retrieve_repeats_the_create_options(api: FakeResponsesAPI) -> None:
    env, _ = activity_env()
    settings = ModelSettings(
        extra_args={"background": True},
        extra_headers={"x-gateway-key": "gateway"},
        response_include=["reasoning.encrypted_content"],
    )
    await env.run(ModelActivity().invoke_model_activity, model_input(settings))
    assert api.retrieves
    for request in api.retrieves:
        assert request.headers["x-gateway-key"] == "gateway"
        assert "reasoning.encrypted_content" in str(request.url)


# Continuing an earlier attempt's response


async def test_retry_continues_the_earlier_response(api: FakeResponsesAPI) -> None:
    api.add_job(Job(id="resp_old", status="in_progress", script=["completed"]))
    env, heartbeats = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    response = await env.run(ModelActivity().invoke_model_activity, model_input())

    assert text(response) == "Done."
    assert response.response_id == "resp_old"
    assert api.creates == []
    assert api.cancels == []
    # Sent before the timer's first heartbeat could send none.
    assert heartbeats[0] == ({_DETAILS_KEY: "resp_old"},)


async def test_retry_returns_a_response_that_finished_meanwhile(
    api: FakeResponsesAPI,
) -> None:
    api.add_job(Job(id="resp_old", status="completed"))
    env, _ = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert text(response) == "Done."
    assert api.creates == []
    assert len(api.retrieves) == 1


@pytest.mark.parametrize(
    "job",
    [
        {"status": "failed", "error_code": "server_error"},
        {"status": "incomplete"},
        {"status": "cancelled"},
        {"gone": True},
    ],
    ids=["failed-transiently", "incomplete", "cancelled", "gone"],
)
async def test_retry_starts_a_new_response_when_the_earlier_one_cant_be_used(
    api: FakeResponsesAPI, job: dict[str, Any]
) -> None:
    api.add_job(Job(id="resp_old", **job))
    env, _ = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert text(response) == "Done."
    assert response.response_id == "resp_1"
    assert len(api.creates) == 1
    assert api.cancels == []


async def test_retry_starts_a_new_response_when_it_cant_read_the_earlier_one(
    api: FakeResponsesAPI,
) -> None:
    api.retrieve_errors = [401]
    env, _ = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert response.response_id == "resp_1"
    assert api.cancels == []


async def test_retry_fails_fast_when_the_earlier_response_failed_for_good(
    api: FakeResponsesAPI,
) -> None:
    api.add_job(Job(id="resp_old", status="failed", error_code="invalid_prompt"))
    env, _ = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert err.value.type == "OpenAIBackgroundResponseFailed"
    assert err.value.non_retryable
    assert "invalid_prompt" in str(err.value)
    # The same request would fail again, so no new response is paid for.
    assert api.creates == []


async def test_retry_keeps_the_earlier_response_after_a_transient_read_error(
    api: FakeResponsesAPI,
) -> None:
    api.add_job(Job(id="resp_old", status="in_progress"))
    api.retrieve_errors = [500]
    env, heartbeats = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert not err.value.non_retryable
    assert api.creates == []
    assert api.cancels == []
    assert heartbeats[-1] == ({_DETAILS_KEY: "resp_old"},)


# How a watched response ends


async def test_permanent_failure_is_not_retried(api: FakeResponsesAPI) -> None:
    api.script = ["failed"]
    api.error_code = "invalid_prompt"
    env, _ = activity_env()
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    # Without background mode, the same request gets a 400 status, which isn't retried.
    assert err.value.type == "OpenAIBackgroundResponseFailed"
    assert err.value.non_retryable
    assert api.cancels == []


async def test_transient_failure_fails_as_in_a_native_run(
    api: FakeResponsesAPI,
) -> None:
    api.script = ["failed"]
    env, _ = activity_env()
    with pytest.raises(ModelBehaviorError):
        await env.run(ModelActivity().invoke_model_activity, model_input())


async def test_response_deleted_while_waiting_says_to_store_it(
    api: FakeResponsesAPI,
) -> None:
    api.script = ["in_progress"]
    env, _ = activity_env()
    async with running(env) as task:
        await assert_eventually(polling(api))
        api.last_job.gone = True
        with pytest.raises(ApplicationError) as err:
            await task
    assert err.value.type == "OpenAIBackgroundResponseExpired"
    assert err.value.non_retryable
    assert "store=True" in str(err.value)


async def test_response_cancelled_elsewhere_is_retried(api: FakeResponsesAPI) -> None:
    api.script = ["cancelled"]
    env, _ = activity_env()
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert err.value.type == "OpenAIBackgroundResponseCancelled"
    assert not err.value.non_retryable


@pytest.mark.parametrize(
    ("attempt", "cancelled"), [(2, False), (3, True)], ids=["retry-left", "last"]
)
async def test_retryable_error_cancels_only_on_the_last_attempt(
    api: FakeResponsesAPI, attempt: int, cancelled: bool
) -> None:
    api.script = ["in_progress"]
    api.retrieve_errors = [503]
    env, _ = activity_env(attempt=attempt, maximum_attempts=3)
    with pytest.raises(ApplicationError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert api.cancels == (["resp_1"] if cancelled else [])


async def test_error_type_the_policy_wont_retry_cancels(api: FakeResponsesAPI) -> None:
    api.script = ["in_progress"]
    api.retrieve_errors = [503]
    env, _ = activity_env(
        retry_policy=RetryPolicy(non_retryable_error_types=["APIStatusError"])
    )
    with pytest.raises(ApplicationError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert api.cancels == ["resp_1"]


@pytest.mark.parametrize(
    ("schedule_to_close", "initial_interval", "cancelled"),
    [
        # The next attempt would start after the deadline, so none follows.
        (timedelta(seconds=30), timedelta(seconds=60), True),
        (timedelta(hours=1), timedelta(seconds=1), False),
    ],
    ids=["retry-past-deadline", "retry-before-deadline"],
)
async def test_schedule_to_close_deadline_decides_the_last_attempt(
    api: FakeResponsesAPI,
    schedule_to_close: timedelta,
    initial_interval: timedelta,
    cancelled: bool,
) -> None:
    api.script = ["in_progress"]
    api.retrieve_errors = [503]
    env, _ = activity_env(
        retry_policy=RetryPolicy(initial_interval=initial_interval),
        schedule_to_close_timeout=schedule_to_close,
    )
    with pytest.raises(ApplicationError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert api.cancels == (["resp_1"] if cancelled else [])


def retry_after(form: str, seconds: int) -> dict[str, str]:
    """A provider retry delay, in one of the forms the OpenAI client reads."""
    if form == "ms":
        return {"retry-after-ms": str(seconds * 1000)}
    if form == "seconds":
        return {"retry-after": str(seconds)}
    return {"retry-after": email.utils.formatdate(time.time() + seconds, usegmt=True)}


@pytest.mark.parametrize("form", ["ms", "seconds", "http-date"])
@pytest.mark.parametrize(
    ("delay", "cancelled"),
    [(60, True), (5, False)],
    ids=["retry-past-deadline", "retry-before-deadline"],
)
async def test_provider_retry_delay_decides_the_last_attempt(
    api: FakeResponsesAPI, form: str, delay: int, cancelled: bool
) -> None:
    # The policy alone would retry before the deadline, so only the provider's delay
    # can put the next attempt after it.
    api.script = ["in_progress"]
    api.retrieve_errors = [(429, retry_after(form, delay))]
    env, heartbeats = activity_env(
        retry_policy=RetryPolicy(initial_interval=timedelta(seconds=1)),
        schedule_to_close_timeout=timedelta(seconds=30),
    )
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert err.value.type == "APIStatusError"
    assert not err.value.non_retryable
    # An HTTP date has whole seconds, and the test takes time to get there.
    next_delay = err.value.next_retry_delay
    assert next_delay is not None
    assert timedelta(seconds=delay - 2) < next_delay <= timedelta(seconds=delay)
    assert api.cancels == (["resp_1"] if cancelled else [])
    if not cancelled:
        # The retry continues the response.
        assert heartbeats[-1] == ({_DETAILS_KEY: "resp_1"},)


async def test_connection_error_while_waiting_is_retried(
    api: FakeResponsesAPI,
) -> None:
    api.script = ["in_progress"]
    api.retrieve_errors = [0]
    env, heartbeats = activity_env()
    with pytest.raises(APIConnectionError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    # The next attempt continues the response.
    assert api.cancels == []
    assert heartbeats[-1] == ({_DETAILS_KEY: "resp_1"},)


# Cancels


@pytest.mark.parametrize(
    ("details", "attempt", "cancelled"),
    [
        (ActivityCancellationDetails(cancel_requested=True), 1, True),
        (ActivityCancellationDetails(reset=True), 1, True),
        (ActivityCancellationDetails(cancel_requested=True, paused=True), 1, True),
        (ActivityCancellationDetails(paused=True), 1, False),
        (ActivityCancellationDetails(worker_shutdown=True), 1, False),
        (ActivityCancellationDetails(worker_shutdown=True), 3, True),
        (ActivityCancellationDetails(timed_out=True, not_found=True), 1, False),
        (ActivityCancellationDetails(timed_out=True, not_found=True), 3, True),
        (ActivityCancellationDetails(not_found=True), 3, True),
        # With a retry to follow and no client to describe the workflow with, keep it.
        (ActivityCancellationDetails(not_found=True), 1, False),
        (ActivityCancellationDetails(), 1, False),
    ],
    ids=[
        "workflow-cancel",
        "reset",
        "cancel-while-paused",
        "paused",
        "shutdown-retry-left",
        "shutdown-last",
        "timeout-retry-left",
        "timeout-last",
        "not-found-last",
        "not-found-undescribed",
        "no-reason",
    ],
)
async def test_cancel_stops_the_response_only_if_no_attempt_follows(
    api: FakeResponsesAPI,
    details: ActivityCancellationDetails,
    attempt: int,
    cancelled: bool,
) -> None:
    api.hold = True
    env, _ = activity_env(attempt=attempt, maximum_attempts=3)
    async with running(env) as task:
        await assert_eventually(polling(api))
        env.cancel(details)
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == (["resp_1"] if cancelled else [])


class FakeTemporalClient:
    """Answers the describe the activity makes to tell whether a retry follows."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    def get_workflow_handle(self, *_args: Any, **_kwargs: Any) -> Any:
        return self

    async def describe(self) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def description(
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.RUNNING,
    state: Any = PendingActivityState.PENDING_ACTIVITY_STATE_STARTED,
) -> Any:
    pending = (
        [SimpleNamespace(activity_id="test", state=state)] if state is not None else []
    )
    return SimpleNamespace(
        status=status, raw_description=SimpleNamespace(pending_activities=pending)
    )


@pytest.mark.parametrize(
    ("outcome", "cancelled"),
    [
        (description(), False),
        (
            description(
                state=PendingActivityState.PENDING_ACTIVITY_STATE_CANCEL_REQUESTED
            ),
            True,
        ),
        (description(state=None), True),
        (description(status=WorkflowExecutionStatus.CANCELED, state=None), True),
        (RPCError("gone", RPCStatusCode.NOT_FOUND, b""), True),
        (RPCError("refused", RPCStatusCode.PERMISSION_DENIED, b""), False),
        (ValueError("could not decode the description"), False),
    ],
    ids=[
        "still-pending",
        "cancel-requested",
        "no-longer-pending",
        "workflow-closed",
        "workflow-gone",
        "describe-refused",
        "describe-failed",
    ],
)
async def test_unknown_attempt_asks_the_server_whether_a_retry_follows(
    api: FakeResponsesAPI, outcome: Any, cancelled: bool
) -> None:
    api.hold = True
    env, _ = activity_env(maximum_attempts=3, client=FakeTemporalClient(outcome))
    async with running(env) as task:
        await assert_eventually(polling(api))
        env.cancel(ActivityCancellationDetails(not_found=True))
        # A failed describe never replaces the cancel.
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == (["resp_1"] if cancelled else [])


async def test_cancel_during_create_still_cancels_the_response(
    api: FakeResponsesAPI,
) -> None:
    api.create_delay = 0.2
    env, heartbeats = activity_env()
    async with running(env) as task:
        await assert_eventually(sending(api))
        env.cancel(ActivityCancellationDetails(cancel_requested=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    # OpenAI accepted the response before the cancel: its ID was still recorded,
    # and the response cancelled.
    assert heartbeats[0] == ({_DETAILS_KEY: "resp_1"},)
    assert api.cancels == ["resp_1"]


async def test_id_recorded_after_a_timeout_cancels_the_response(
    api: FakeResponsesAPI,
) -> None:
    # The worker drops heartbeats once it knows the attempt is gone, so a retry
    # could never continue a response whose ID arrived after the timeout.
    api.create_delay = 0.2
    env, _ = activity_env(maximum_attempts=3)
    async with running(env) as task:
        await assert_eventually(sending(api))
        env.cancel(ActivityCancellationDetails(timed_out=True, not_found=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == ["resp_1"]


async def test_throttled_id_cancels_the_response_after_a_timeout(
    api: FakeResponsesAPI,
) -> None:
    # The timer sends the attempt's first heartbeat before OpenAI answers, so the
    # worker throttles the heartbeat carrying the ID, for up to 0.8 s here.
    api.hold = True
    api.create_delay = 0.7
    env, heartbeats = activity_env(
        maximum_attempts=3, heartbeat_timeout=timedelta(seconds=1)
    )
    async with running(env) as task:
        await assert_eventually(polling(api))
        assert heartbeats[0] == ()
        env.cancel(ActivityCancellationDetails(timed_out=True, not_found=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == ["resp_1"]


@pytest.mark.parametrize(
    ("throttle", "cancelled"),
    [
        (HeartbeatThrottle(default=timedelta(milliseconds=50)), False),
        (HeartbeatThrottle(), True),
    ],
    ids=["throttle-elapsed", "throttle-not-elapsed"],
)
async def test_workers_throttle_decides_whether_a_timeout_cancels_the_response(
    api: FakeResponsesAPI, throttle: HeartbeatThrottle, cancelled: bool
) -> None:
    # The earlier attempt's response is gone. The attempt's first heartbeat carried
    # its ID, so the worker throttles the one that carries the new response's ID.
    api.hold = True
    env, _ = activity_env(
        heartbeat_details=inherited("resp_old"), attempt=2, maximum_attempts=3
    )
    async with running(env, ModelActivity(heartbeat_throttle=throttle)) as task:
        await assert_eventually(sending(api))
        # Longer than the 50 ms throttle, far shorter than the default 30 seconds.
        await asyncio.sleep(0.2)
        env.cancel(ActivityCancellationDetails(timed_out=True, not_found=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == (["resp_1"] if cancelled else [])


# The fake API is the default client, which the plugin's model activity needs.
@pytest.mark.usefixtures("api")
async def test_plugin_gives_the_model_activity_the_workers_throttle(
    client: Client,
) -> None:
    worker = Worker(
        client,
        task_queue=str(uuid.uuid4()),
        plugins=[OpenAIAgentsPlugin()],
        default_heartbeat_throttle_interval=timedelta(seconds=90),
        max_heartbeat_throttle_interval=timedelta(seconds=120),
    )
    [model_activity] = {
        owner
        for act in worker.config(active_config=True).get("activities", [])
        if isinstance(owner := getattr(act, "__self__", None), ModelActivity)
    }
    assert model_activity._heartbeat_throttle == HeartbeatThrottle(
        default=timedelta(seconds=90), maximum=timedelta(seconds=120)
    )


async def test_create_reply_that_never_comes_is_logged(
    api: FakeResponsesAPI,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(_background_responses, "_CREATE_REPLY_TIMEOUT", 0.05)
    api.create_delay = 5
    env, _ = activity_env()
    async with running(env) as task:
        await assert_eventually(sending(api))
        env.cancel(ActivityCancellationDetails(cancel_requested=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == []
    assert "neither cancelled nor continued" in caplog.text


async def test_failed_cancel_is_logged_and_the_cancel_still_reported(
    api: FakeResponsesAPI, caplog: pytest.LogCaptureFixture
) -> None:
    api.hold = True
    api.cancel_errors = [500]
    env, _ = activity_env()
    async with running(env) as task:
        await assert_eventually(polling(api))
        env.cancel(ActivityCancellationDetails(cancel_requested=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == ["resp_1"]
    assert "Could not cancel background response resp_1" in caplog.text


# Refusals and pass-through


class ProviderOf(ModelProvider):
    def __init__(self, model: Model) -> None:
        self.model = model

    def get_model(self, model_name: str | None) -> Model:
        return self.model


@pytest.mark.parametrize(
    ("activity_kwargs", "input_fields", "message"),
    [
        ({"custom_provider": True}, {}, "default model provider"),
        ({"is_local": True}, {}, "use_local_activity"),
        ({}, {"conversation_id": "conv_1"}, "conversation_id"),
    ],
    ids=["custom-provider", "local-activity", "conversation"],
)
async def test_unsupported_background_call_is_refused_before_any_request(
    api: FakeResponsesAPI,
    activity_kwargs: dict[str, Any],
    input_fields: dict[str, Any],
    message: str,
) -> None:
    env, _ = activity_env(is_local=activity_kwargs.get("is_local", False))
    activity = (
        ModelActivity(
            ProviderOf(OpenAIResponsesModel(model="gpt-4o", openai_client=api.client))
        )
        if activity_kwargs.get("custom_provider")
        else ModelActivity()
    )
    with pytest.raises(ApplicationError) as err:
        await env.run(activity.invoke_model_activity, model_input(**input_fields))
    assert err.value.type == UNSUPPORTED_ERROR_TYPE
    assert err.value.non_retryable
    assert message in str(err.value)
    assert api.creates == []
    assert api.retrieves == []


async def test_refusal_cancels_an_earlier_attempts_response(
    api: FakeResponsesAPI,
) -> None:
    api.add_job(Job(id="resp_old", status="in_progress"))
    env, _ = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    with pytest.raises(ApplicationError):
        await env.run(
            ModelActivity().invoke_model_activity,
            model_input(conversation_id="conv_1"),
        )
    assert api.cancels == ["resp_old"]


class RecordingModel(Model):
    def __init__(self) -> None:
        self.settings: list[ModelSettings] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        self.settings.append(model_settings)
        return ModelResponse(output=[], usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


async def test_other_models_get_background_as_in_a_native_run() -> None:
    model = RecordingModel()
    env, _ = activity_env()
    await env.run(ModelActivity(ProviderOf(model)).invoke_model_activity, model_input())
    assert model.settings[0].extra_args == {"background": True}


async def test_sdk_that_bypasses_create_is_detected(
    api: FakeResponsesAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As if the Agents SDK sent the request through another client method.
    monkeypatch.setattr(OpenAIResponsesModel, "_get_client", lambda self: api.client)
    env, _ = activity_env()
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert err.value.type == UNSUPPORTED_ERROR_TYPE
    assert err.value.non_retryable
    # The unfinished response it got back is not left running.
    assert api.cancels == ["resp_1"]


async def test_inherited_id_survives_the_heartbeat_timer(api: FakeResponsesAPI) -> None:
    api.add_job(
        Job(id="resp_old", status="in_progress", script=["completed"], held=True)
    )
    env, heartbeats = activity_env(
        heartbeat_details=inherited("resp_old"),
        attempt=2,
        heartbeat_timeout=timedelta(milliseconds=40),
    )
    async with running(env) as task:
        await asyncio.sleep(0.2)
        api.jobs["resp_old"].held = False
        response = await task
    assert response.response_id == "resp_old"
    # Several timer heartbeats, and every one carried the ID on.
    assert len(heartbeats) > 2
    assert all(details == ({_DETAILS_KEY: "resp_old"},) for details in heartbeats)


async def test_worker_without_background_support_cancels_an_inherited_response(
    api: FakeResponsesAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As on a worker set to use Chat Completions, during a rolling deploy.
    monkeypatch.setattr(_openai_shared, "_use_responses_by_default", False)
    api.add_job(Job(id="resp_old", status="in_progress"))
    env, heartbeats = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    # The Chat Completions client refuses a background keyword argument, so send it
    # in the body.
    settings = ModelSettings(extra_body={"background": True})
    response = await env.run(
        ModelActivity().invoke_model_activity, model_input(settings)
    )
    assert text(response) == "Done."
    assert api.chat_completions == 1
    assert api.cancels == ["resp_old"]
    # A later attempt doesn't find the cancelled response again.
    assert heartbeats == [()]


async def test_second_create_in_one_call_is_refused(api: FakeResponsesAPI) -> None:
    env, _ = activity_env()

    async def two_creates() -> None:
        client = BackgroundOpenAIClient(api.client)
        with attempt_call(BackgroundCall(api.client, None)):
            await client.responses.create(model="gpt-4o", input="Hi", background=True)
            with pytest.raises(ApplicationError) as err:
                await client.responses.create(
                    model="gpt-4o", input="Hi", background=True
                )
        assert err.value.type == UNSUPPORTED_ERROR_TYPE
        assert err.value.non_retryable

    await env.run(two_creates)
    assert len(api.creates) == 1


async def test_streamed_background_create_passes_through(api: FakeResponsesAPI) -> None:
    env, _ = activity_env()

    async def streamed() -> None:
        client = BackgroundOpenAIClient(api.client)
        call = BackgroundCall(api.client, None)
        with attempt_call(call):
            stream = await client.responses.create(
                model="gpt-4o", input="Hi", background=True, stream=True
            )
            await stream.close()
        assert not call.intercepted

    await env.run(streamed)
    assert api.creates[0]["stream"] is True
    assert api.retrieves == []


def test_client_wrapper_reads_the_client_lazily_and_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClientStub:
        base_url = "https://stub.invalid/v1"

    stub = cast(AsyncOpenAI, cast(object, ClientStub()))
    # A default client without responses still builds a worker's model activity.
    monkeypatch.setattr(_openai_shared, "_default_openai_client", stub)
    ModelActivity()
    wrapper = BackgroundOpenAIClient(stub)
    assert wrapper.base_url == "https://stub.invalid/v1"
    # An object copied without __init__ raises AttributeError, not RecursionError.
    bare = object.__new__(BackgroundOpenAIClient)
    with pytest.raises(AttributeError):
        _ = bare.base_url


async def test_client_wrapper_copies_stay_wrapped(api: FakeResponsesAPI) -> None:
    wrapper = BackgroundOpenAIClient(api.client)
    assert isinstance(wrapper.with_options(max_retries=1), BackgroundOpenAIClient)
    assert isinstance(wrapper.copy(max_retries=1), BackgroundOpenAIClient)


# With a Temporal server


@dataclass
class BackgroundRun:
    cancel_call: bool = False


@workflow.defn
class BackgroundAgentWorkflow:
    def __init__(self) -> None:
        self._cancel = False
        self._finish = False

    @workflow.run
    async def run(self, run: BackgroundRun) -> str:
        agent = Agent[None](
            name="Assistant",
            instructions="Be brief.",
            model="gpt-4o",
            model_settings=BACKGROUND,
        )
        if not run.cancel_call:
            result = await Runner.run(starting_agent=agent, input="Hi")
            return f"{result.final_output} {result.context_wrapper.usage.total_tokens}"
        call = asyncio.create_task(Runner.run(starting_agent=agent, input="Hi"))
        await workflow.wait_condition(lambda: self._cancel)
        call.cancel()
        try:
            await call
        except (asyncio.CancelledError, Exception):
            pass
        await workflow.wait_condition(lambda: self._finish)
        return "cancelled"

    @workflow.signal
    def cancel(self) -> None:
        self._cancel = True

    @workflow.signal
    def finish(self) -> None:
        self._finish = True


def agent_environment(
    start_to_close_timeout: timedelta = timedelta(seconds=30),
    heartbeat_timeout: timedelta = timedelta(seconds=2),
) -> AgentEnvironment:
    return AgentEnvironment(
        model_params=ModelActivityParameters(
            start_to_close_timeout=start_to_close_timeout,
            heartbeat_timeout=heartbeat_timeout,
            retry_policy=RetryPolicy(
                initial_interval=timedelta(milliseconds=100),
                maximum_interval=timedelta(milliseconds=100),
            ),
        ),
    )


async def model_attempts(client: Client, workflow_id: str) -> list[int]:
    attempts: list[int] = []
    async for event in client.get_workflow_handle(workflow_id).fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            attempts.append(event.activity_task_started_event_attributes.attempt)
    return attempts


async def test_workflow_gets_the_background_response(
    client: Client, api: FakeResponsesAPI
) -> None:
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            result = await client.execute_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(),
                id=f"background-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )
    assert result == "Done. 8"
    assert len(api.creates) == 1
    assert api.cancels == []


async def test_another_worker_continues_the_response_after_a_shutdown(
    client: Client, api: FakeResponsesAPI
) -> None:
    api.hold = True
    workflow_id = f"background-shutdown-{uuid.uuid4()}"
    task_queue = str(uuid.uuid4())
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow, task_queue=task_queue):
            handle = await client.start_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(),
                id=workflow_id,
                task_queue=task_queue,
            )

            await assert_eventually(polling(api), timeout=timedelta(seconds=20))
        # The first worker is gone and left the response running.
        assert api.cancels == []
        api.last_job.held = False
        async with new_worker(client, BackgroundAgentWorkflow, task_queue=task_queue):
            result = await handle.result()
    assert result == "Done. 8"
    assert len(api.creates) == 1
    assert api.cancels == []
    assert await model_attempts(client, workflow_id) == [2]


async def test_retry_after_an_attempt_timeout_continues_the_response(
    client: Client, api: FakeResponsesAPI
) -> None:
    api.hold = True
    workflow_id = f"background-timeout-{uuid.uuid4()}"
    async with agent_environment(
        start_to_close_timeout=timedelta(seconds=3),
        heartbeat_timeout=timedelta(seconds=1),
    ) as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            handle = await client.start_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(),
                id=workflow_id,
                task_queue=worker.task_queue,
            )

            async def first_attempt_timed_out() -> None:
                pending = (await handle.describe()).raw_description.pending_activities
                assert pending and pending[0].attempt >= 2

            # The held response outlasts the first attempt, which times out.
            await assert_eventually(
                first_attempt_timed_out, timeout=timedelta(seconds=30)
            )
            api.last_job.held = False
            result = await handle.result()
    assert result == "Done. 8"
    # The retry continued the response the timed-out attempt started, from the ID
    # the server stored with that attempt's heartbeats.
    assert len(api.creates) == 1
    assert api.cancels == []
    [attempt] = await model_attempts(client, workflow_id)
    assert attempt >= 2


async def test_workflow_cancelling_the_call_cancels_the_response(
    client: Client, api: FakeResponsesAPI
) -> None:
    api.hold = True
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            handle = await client.start_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(cancel_call=True),
                id=f"background-cancel-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

            await assert_eventually(polling(api), timeout=timedelta(seconds=20))
            await handle.signal(BackgroundAgentWorkflow.cancel)

            async def cancelled() -> None:
                assert api.cancels == ["resp_1"]

            await assert_eventually(cancelled, timeout=timedelta(seconds=20))
            await handle.signal(BackgroundAgentWorkflow.finish)
            assert await handle.result() == "cancelled"


async def test_cancelling_the_workflow_cancels_the_response(
    client: Client, api: FakeResponsesAPI
) -> None:
    api.hold = True
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            handle = await client.start_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(),
                id=f"background-workflow-cancel-{uuid.uuid4()}",
                task_queue=worker.task_queue,
            )

            await assert_eventually(polling(api), timeout=timedelta(seconds=20))
            # The workflow closes in the same task, so the activity may hear of the
            # cancel only as the server no longer knowing the attempt.
            await handle.cancel()
            with pytest.raises(WorkflowFailureError):
                await handle.result()

            async def cancelled() -> None:
                assert api.cancels == ["resp_1"]

            await assert_eventually(cancelled, timeout=timedelta(seconds=20))
