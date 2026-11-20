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
from typing import TYPE_CHECKING, Any, cast, get_args

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
from agents.items import TResponseStreamEvent
from agents.models import _openai_shared
from openai import AsyncOpenAI
from openai.types.responses import ResponseError

from temporalio import workflow
from temporalio.activity import ActivityCancellationDetails
from temporalio.api.enums.v1 import EventType, PendingActivityState
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.converter import DataConverter
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents import (
    ModelActivityParameters,
    OpenAIPayloadConverter,
    _background_responses,
    _openai_runner,
    _temporal_model_stub,
)
from temporalio.openai_agents._background_responses import (
    _DETAILS_KEY,
    _PERMANENT_ERROR_CODES,
    BACKGROUND_RESPONSES_PATCH,
    UNSUPPORTED_ERROR_TYPE,
    BackgroundCall,
    BackgroundOpenAIClient,
    _PollSchedule,
    attempt_call,
    is_final_failure,
)
from temporalio.openai_agents._heartbeat_decorator import (
    heartbeating,
    set_heartbeat_details,
)
from temporalio.openai_agents._invoke_model_activity import (
    ActivityModelInput,
    ModelActivity,
    ModelTracingInput,
    StreamingActivityModelInput,
)
from temporalio.openai_agents.testing import AgentEnvironment
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import ActivityEnvironment
from tests.helpers import assert_eventually, new_worker

if TYPE_CHECKING:
    # Imported where used: the workflow sandbox re-imports this module.
    import httpx2

BACKGROUND = ModelSettings(extra_args={"background": True})
# A retrieve answer whose body the OpenAI client can't parse, an error it doesn't retry.
MALFORMED = -1

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
        # response headers. 0 drops the connection, MALFORMED sends a body that isn't
        # JSON.
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
            if status == MALFORMED:
                return httpx2.Response(
                    200, content=b"{", headers={"content-type": "application/json"}
                )
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
    settings: ModelSettings = BACKGROUND,
    conversation_id: str | None = None,
    *,
    patched: bool = True,
) -> ActivityModelInput:
    """A model call's input, by default from a run that records the patch."""
    return ActivityModelInput(
        model_name="gpt-4o",
        input="Hi",
        model_settings=settings,
        tracing=ModelTracingInput.DISABLED,
        tools=[],
        handoffs=[],
        conversation_id=conversation_id,
        background_responses=patched,
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
        {"status": "failed", "error_code": "a_code_openai_adds_later"},
        {"status": "incomplete"},
        {"status": "cancelled"},
        {"gone": True},
    ],
    ids=[
        "failed-transiently",
        "failed-with-an-unknown-code",
        "incomplete",
        "cancelled",
        "gone",
    ],
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


async def test_retry_reads_the_earlier_response_again_after_a_transient_error(
    api: FakeResponsesAPI,
) -> None:
    api.add_job(Job(id="resp_old", status="in_progress"))
    api.retrieve_errors = [500]
    env, _ = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert response.response_id == "resp_old"
    assert api.creates == []
    assert api.cancels == []


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


def test_permanent_error_codes_are_openai_error_codes() -> None:
    # A misspelled code would be retried, paying for a new response each time.
    assert _PERMANENT_ERROR_CODES <= set(
        get_args(ResponseError.model_fields["code"].annotation)
    )


@pytest.mark.parametrize(
    "error_code",
    ["server_error", "a_code_openai_adds_later"],
    ids=["transient-code", "unknown-code"],
)
async def test_other_failure_fails_as_in_a_native_run(
    api: FakeResponsesAPI, error_code: str
) -> None:
    api.script = ["failed"]
    api.error_code = error_code
    env, _ = activity_env()
    # Left to the retry policy, as an error without an HTTP status is.
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
    api.retrieve_errors = [MALFORMED]
    env, _ = activity_env(attempt=attempt, maximum_attempts=3)
    with pytest.raises(ValueError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert api.cancels == (["resp_1"] if cancelled else [])


async def test_error_type_the_policy_wont_retry_cancels(api: FakeResponsesAPI) -> None:
    api.script = ["in_progress"]
    api.retrieve_errors = [MALFORMED]
    env, _ = activity_env(
        retry_policy=RetryPolicy(non_retryable_error_types=["JSONDecodeError"])
    )
    with pytest.raises(ValueError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert api.cancels == ["resp_1"]


async def test_read_again_despite_the_policys_non_retryable_types(
    api: FakeResponsesAPI,
) -> None:
    # The policy governs the server's retries of the activity; a read resends nothing.
    api.retrieve_errors = [503]
    env, _ = activity_env(
        retry_policy=RetryPolicy(non_retryable_error_types=["APIStatusError"])
    )
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert response.response_id == "resp_1"
    assert api.cancels == []


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
    api.retrieve_errors = [MALFORMED]
    env, _ = activity_env(
        retry_policy=RetryPolicy(initial_interval=initial_interval),
        schedule_to_close_timeout=schedule_to_close,
    )
    with pytest.raises(ValueError):
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert api.cancels == (["resp_1"] if cancelled else [])


@pytest.mark.parametrize(
    ("delay", "final"),
    [(60, True), (5, False)],
    ids=["retry-past-deadline", "retry-before-deadline"],
)
async def test_provider_retry_delay_decides_the_last_attempt(
    delay: int, final: bool
) -> None:
    # The policy alone would retry before the deadline, so only the failure's own
    # delay can put the next attempt after it.
    env, _ = activity_env(
        retry_policy=RetryPolicy(initial_interval=timedelta(seconds=1)),
        schedule_to_close_timeout=timedelta(seconds=30),
    )
    err = ApplicationError("busy", next_retry_delay=timedelta(seconds=delay))

    async def is_final() -> bool:
        return is_final_failure(err)

    assert await env.run(is_final) is final


@pytest.mark.parametrize(
    ("coefficient", "attempt"),
    [(1.01, 500), (2.0, 5000)],
    ids=["slow-backoff", "overflowing-backoff"],
)
async def test_backoff_after_many_attempts_reaches_the_maximum_interval(
    coefficient: float, attempt: int
) -> None:
    # 1.01 ** 499 is about 144, so the next retry waits the 100 s maximum, past the
    # deadline. 2.0 ** 4999 overflows a float.
    env, _ = activity_env(
        attempt=attempt,
        retry_policy=RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=coefficient,
            maximum_interval=timedelta(seconds=100),
        ),
        schedule_to_close_timeout=timedelta(seconds=50),
    )

    async def is_final() -> bool:
        return is_final_failure(ValueError())

    assert await env.run(is_final)


def retry_after(form: str, seconds: int) -> dict[str, str]:
    """A provider retry delay, in one of the forms the OpenAI client reads."""
    if form == "ms":
        return {"retry-after-ms": str(seconds * 1000)}
    if form == "seconds":
        return {"retry-after": str(seconds)}
    return {"retry-after": email.utils.formatdate(time.time() + seconds, usegmt=True)}


async def test_transient_read_error_on_the_last_attempt_reads_again(
    api: FakeResponsesAPI,
) -> None:
    # Reading resends no request, so the attempt goes on instead of ending.
    api.retrieve_errors = [503, 503]
    env, _ = activity_env(attempt=3, maximum_attempts=3)
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert response.response_id == "resp_1"
    assert len(api.creates) == 1
    assert api.cancels == []


async def test_connection_error_while_waiting_reads_again(
    api: FakeResponsesAPI,
) -> None:
    api.retrieve_errors = [0]
    env, _ = activity_env()
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert response.response_id == "resp_1"
    assert len(api.creates) == 1
    assert api.cancels == []


@pytest.mark.parametrize("form", ["ms", "seconds", "http-date"])
async def test_read_again_waits_for_the_provider_retry_delay(
    api: FakeResponsesAPI, form: str
) -> None:
    api.retrieve_errors = [(429, retry_after(form, 2))]
    env, _ = activity_env()
    started = time.monotonic()
    response = await env.run(ModelActivity().invoke_model_activity, model_input())
    assert response.response_id == "resp_1"
    # An HTTP date has whole seconds.
    assert time.monotonic() - started >= 1
    assert api.cancels == []


async def test_permanent_read_error_ends_the_attempt(api: FakeResponsesAPI) -> None:
    api.retrieve_errors = [400]
    env, _ = activity_env()
    with pytest.raises(ApplicationError) as err:
        await env.run(ModelActivity().invoke_model_activity, model_input())
    assert err.value.non_retryable
    assert len(api.retrieves) == 1
    assert api.cancels == ["resp_1"]


# Cancels


@pytest.mark.parametrize(
    ("details", "attempt", "cancelled"),
    [
        (ActivityCancellationDetails(cancel_requested=True), 1, True),
        # Temporal keeps heartbeat details on reset unless asked to clear them.
        (ActivityCancellationDetails(reset=True), 1, False),
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
        self.describes = 0
        self.data_converter = DataConverter(
            payload_converter_class=OpenAIPayloadConverter
        )

    def get_workflow_handle(self, *_args: Any, **_kwargs: Any) -> Any:
        return self

    async def describe(self) -> Any:
        self.describes += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def description(
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.RUNNING,
    state: Any = PendingActivityState.PENDING_ACTIVITY_STATE_STARTED,
    held: str | None = "resp_1",
    attempt: int = 2,
) -> Any:
    """A workflow description, whose pending activity's details hold the ID ``held``.

    The activity is pending at ``attempt``, by default the retry of the first attempt.
    """
    details = [{_DETAILS_KEY: held}] if held is not None else []
    payloads = OpenAIPayloadConverter().to_payloads(details)
    pending = (
        [
            SimpleNamespace(
                activity_id="test",
                state=state,
                attempt=attempt,
                heartbeat_details=SimpleNamespace(payloads=payloads),
            )
        ]
        if state is not None
        else []
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
        (description(held=None), True),
        (description(held="resp_old"), True),
        (description(state=None), True),
        (description(status=WorkflowExecutionStatus.CANCELED, state=None), True),
        (RPCError("gone", RPCStatusCode.NOT_FOUND, b""), True),
        (RPCError("refused", RPCStatusCode.PERMISSION_DENIED, b""), False),
        (ValueError("could not decode the description"), False),
    ],
    ids=[
        "still-pending",
        "cancel-requested",
        "id-not-held",
        "other-id-held",
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
    # OpenAI answered after the timeout, so the server never got the ID.
    api.create_delay = 0.2
    env, _ = activity_env(
        maximum_attempts=3, client=FakeTemporalClient(description(held=None))
    )
    async with running(env) as task:
        await assert_eventually(sending(api))
        env.cancel(ActivityCancellationDetails(timed_out=True, not_found=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert api.cancels == ["resp_1"]


@pytest.mark.parametrize(
    ("held", "cancelled"),
    [("resp_1", False), (None, True), ("resp_old", True)],
    ids=["id-held", "no-id-held", "earlier-id-held"],
)
async def test_timeout_cancels_a_response_whose_id_the_server_lacks(
    api: FakeResponsesAPI, held: str | None, cancelled: bool
) -> None:
    # The earlier attempt's response is gone, so this attempt starts another. Whether
    # the heartbeat with the new ID reached the server decides what a retry finds.
    api.hold = True
    client = FakeTemporalClient(description(held=held, attempt=3))
    env, _ = activity_env(
        heartbeat_details=inherited("resp_old"),
        attempt=2,
        maximum_attempts=3,
        client=client,
    )

    async def polling_the_new_response() -> None:
        assert api.creates and len(api.retrieves) > 1

    async with running(env) as task:
        await assert_eventually(polling_the_new_response)
        env.cancel(ActivityCancellationDetails(timed_out=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert client.describes == 1
    assert api.cancels == (["resp_1"] if cancelled else [])


@pytest.mark.parametrize(
    ("attempt", "pending_attempt", "held", "cancelled"),
    [
        (2, 3, "resp_old", False),
        # As after update-options raised maximum_attempts during the attempt.
        (3, 4, "resp_old", False),
        # The server hasn't processed the timeout yet, so the policy decides.
        (2, 2, "resp_old", False),
        (3, 3, "resp_old", True),
        # An operator reset the activity and kept its heartbeat details.
        (3, 1, "resp_old", False),
        (2, 3, None, True),
    ],
    ids=[
        "retry",
        "retry-past-the-policy",
        "undecided-retry-left",
        "undecided-last-attempt",
        "reset",
        "details-cleared",
    ],
)
async def test_timeout_follows_the_servers_retry_decision(
    api: FakeResponsesAPI,
    attempt: int,
    pending_attempt: int,
    held: str | None,
    cancelled: bool,
) -> None:
    api.add_job(Job(id="resp_old", status="in_progress", held=True))
    client = FakeTemporalClient(description(held=held, attempt=pending_attempt))
    env, _ = activity_env(
        heartbeat_details=inherited("resp_old"),
        attempt=attempt,
        maximum_attempts=3,
        client=client,
    )
    async with running(env) as task:
        await assert_eventually(polling(api))
        env.cancel(ActivityCancellationDetails(timed_out=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    assert client.describes == 1
    assert api.cancels == (["resp_old"] if cancelled else [])


@pytest.mark.parametrize(
    ("error", "says"),
    [
        (RPCError("refused", RPCStatusCode.PERMISSION_DENIED, b""), "permission"),
        (RPCError("unavailable", RPCStatusCode.UNAVAILABLE, b""), "it failed"),
    ],
    ids=["refused", "failed"],
)
async def test_failed_describe_says_why(
    api: FakeResponsesAPI,
    caplog: pytest.LogCaptureFixture,
    error: RPCError,
    says: str,
) -> None:
    api.hold = True
    env, _ = activity_env(maximum_attempts=3, client=FakeTemporalClient(error))
    async with running(env) as task:
        await assert_eventually(polling(api))
        env.cancel(ActivityCancellationDetails(not_found=True))
        with pytest.raises(asyncio.CancelledError):
            await task
    [warning] = [r for r in caplog.records if "Could not describe" in r.getMessage()]
    assert says in warning.getMessage()
    assert ("permission" in warning.getMessage()) == (says == "permission")


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

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> AsyncIterator[TResponseStreamEvent]:
        self.settings.append(model_settings)
        events: list[TResponseStreamEvent] = []
        for event in events:
            yield event


def streaming_model_input(
    settings: ModelSettings = BACKGROUND, *, patched: bool = True
) -> Any:
    return StreamingActivityModelInput(
        model_name="gpt-4o",
        input="Hi",
        model_settings=settings,
        tracing=ModelTracingInput.DISABLED,
        tools=[],
        handoffs=[],
        streaming_topic="events",
        background_responses=patched,
    )


async def test_other_models_get_background_as_in_a_native_run() -> None:
    model = RecordingModel()
    env, _ = activity_env()
    await env.run(ModelActivity(ProviderOf(model)).invoke_model_activity, model_input())
    assert model.settings[0].extra_args == {"background": True}


class NullStream:
    """A workflow stream client that publishes nowhere."""

    def topic(self, _name: str) -> Any:
        return SimpleNamespace(publish=lambda event: None)

    async def __aenter__(self) -> NullStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


async def test_other_streamed_models_get_background_as_in_a_native_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WorkflowStreamClient,
        "from_within_activity",
        classmethod(lambda cls, **kwargs: NullStream()),
    )
    model = RecordingModel()
    env, _ = activity_env()
    await env.run(
        ModelActivity(ProviderOf(model)).invoke_model_activity_streaming,
        streaming_model_input(),
    )
    assert model.settings[0].extra_args == {"background": True}


@pytest.mark.parametrize(
    ("settings", "custom_provider"),
    [
        (BACKGROUND, False),
        (ModelSettings(extra_body={"background": True}), False),
        (BACKGROUND, True),
    ],
    ids=["extra-args", "extra-body", "custom-provider"],
)
async def test_streamed_background_call_is_refused_before_any_request(
    api: FakeResponsesAPI, settings: ModelSettings, custom_provider: bool
) -> None:
    env, _ = activity_env()
    activity = (
        ModelActivity(
            ProviderOf(OpenAIResponsesModel(model="gpt-4o", openai_client=api.client))
        )
        if custom_provider
        else ModelActivity()
    )
    with pytest.raises(ApplicationError) as err:
        await env.run(
            activity.invoke_model_activity_streaming, streaming_model_input(settings)
        )
    assert err.value.type == UNSUPPORTED_ERROR_TYPE
    assert err.value.non_retryable
    assert "streaming" in str(err.value)
    assert api.creates == []


async def test_call_from_before_the_patch_runs_as_before(
    api: FakeResponsesAPI,
) -> None:
    api.add_job(Job(id="resp_old", status="in_progress"))
    # Even with an ID in its details, which only a patched call would record.
    env, heartbeats = activity_env(heartbeat_details=inherited("resp_old"), attempt=2)
    response = await env.run(
        ModelActivity().invoke_model_activity, model_input(patched=False)
    )
    # The Agents SDK returns the background response as OpenAI first answers it.
    assert response.response_id == "resp_1"
    assert api.creates[0]["background"] is True
    assert api.retrieves == []
    assert api.cancels == []
    assert heartbeats == []


async def test_streamed_call_from_before_the_patch_is_not_refused(
    api: FakeResponsesAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        WorkflowStreamClient,
        "from_within_activity",
        classmethod(lambda cls, **kwargs: NullStream()),
    )
    env, _ = activity_env()
    # The fake answers with JSON rather than a stream, so no events come back.
    await env.run(
        ModelActivity().invoke_model_activity_streaming,
        streaming_model_input(patched=False),
    )
    assert api.creates[0]["stream"] is True
    assert api.creates[0]["background"] is True


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
    background: bool = True


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
            model_settings=BACKGROUND if run.background else ModelSettings(),
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


async def patch_ids(client: Client, workflow_id: str) -> list[str]:
    """The IDs of the patch markers in a workflow's history."""
    ids: list[str] = []
    async for event in client.get_workflow_handle(workflow_id).fetch_history_events():
        if event.event_type == EventType.EVENT_TYPE_MARKER_RECORDED:
            details = event.marker_recorded_event_attributes.details
            if "patch-data" in details:
                for payload in details["patch-data"].payloads:
                    ids.append(json.loads(payload.data)["id"])
    return ids


async def test_workflow_gets_the_background_response(
    client: Client, api: FakeResponsesAPI
) -> None:
    workflow_id = f"background-{uuid.uuid4()}"
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            result = await client.execute_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(),
                id=workflow_id,
                task_queue=worker.task_queue,
            )
    assert result == "Done. 8"
    assert len(api.creates) == 1
    assert api.cancels == []
    assert BACKGROUND_RESPONSES_PATCH in await patch_ids(client, workflow_id)


async def test_run_evaluates_the_patch_before_its_first_model_call(
    client: Client, api: FakeResponsesAPI
) -> None:
    # Even without background, so that a run on an earlier release that hasn't yet
    # reached a background call replays without the marker and keeps the old behavior.
    workflow_id = f"background-{uuid.uuid4()}"
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            await client.execute_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(background=False),
                id=workflow_id,
                task_queue=worker.task_queue,
            )
    events = [
        event.event_type
        async for event in client.get_workflow_handle(
            workflow_id
        ).fetch_history_events()
    ]
    assert events.index(EventType.EVENT_TYPE_MARKER_RECORDED) < events.index(
        EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    )
    assert await patch_ids(client, workflow_id) == [BACKGROUND_RESPONSES_PATCH]
    assert api.creates[0].get("background") is None


async def test_run_from_before_the_patch_keeps_the_earlier_behavior(
    client: Client, api: FakeResponsesAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run that started an agent run on an earlier release replays without the
    # patch marker, as if the gate were off.
    for module in (_openai_runner, _temporal_model_stub):
        monkeypatch.setattr(module, "background_responses_enabled", lambda: False)
    workflow_id = f"background-{uuid.uuid4()}"
    async with agent_environment() as env:
        client = env.applied_on_client(client)
        async with new_worker(client, BackgroundAgentWorkflow) as worker:
            result = await client.execute_workflow(
                BackgroundAgentWorkflow.run,
                BackgroundRun(),
                id=workflow_id,
                task_queue=worker.task_queue,
            )
    # The model call returned the background response unfinished, with no output.
    assert result != "Done. 8"
    assert len(api.creates) == 1
    assert api.retrieves == []
    assert BACKGROUND_RESPONSES_PATCH not in await patch_ids(client, workflow_id)


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
