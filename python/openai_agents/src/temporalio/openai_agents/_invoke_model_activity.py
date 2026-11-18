"""A temporal activity that invokes a LLM model.

Implements mapping of OpenAI datastructures to Pydantic friendly types.
"""

import asyncio
import email.utils
import enum
import time
from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, NoReturn, cast

from agents import (
    AgentOutputSchemaBase,
    CodeInterpreterTool,
    FileSearchTool,
    FunctionTool,
    Handoff,
    HostedMCPTool,
    ImageGenerationTool,
    Model,
    ModelProvider,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    OpenAIProvider,
    OpenAIResponsesModel,
    RunContextWrapper,
    Tool,
    ToolCaller,
    TResponseInputItem,
    UserError,
    WebSearchTool,
    tool_namespace,
)
from agents.items import TResponseStreamEvent
from agents.models import _openai_shared, get_default_model_settings
from agents.tool import (
    ApplyPatchTool,
    CustomTool,
    ShellTool,
    ShellToolEnvironment,
    ToolSearchTool,
)
from openai import (
    APIStatusError,
    AsyncOpenAI,
)
from openai.types.responses import (
    CustomToolParam,
    ResponseCreatedEvent,
    ResponseErrorEvent,
)
from openai.types.responses.tool_param import Mcp
from typing_extensions import Required, TypedDict

from temporalio import activity
from temporalio.contrib.workflow_streams import WorkflowStreamClient
from temporalio.exceptions import ApplicationError
from temporalio.openai_agents._background_responses import (
    UNSUPPORTED_ERROR_TYPE,
    BackgroundCall,
    BackgroundOpenAIClient,
    HeartbeatThrottle,
    attempt_call,
    cancel_reason,
    inherited_response_id,
    is_final_failure,
    requests_background,
)
from temporalio.openai_agents._heartbeat_decorator import (
    auto_heartbeater,
    heartbeating,
    set_heartbeat_details,
)
from temporalio.openai_agents._temporal_worker_env_ref import (
    AllowAllWorkerEnvVars,
    _WorkerEnvRefResolver,
)


@dataclass
class HandoffInput:
    """Data conversion friendly representation of a Handoff. Contains only the fields which are needed by the model
    execution to determine what to handoff to, not the actual handoff invocation, which remains in the workflow context.
    """

    tool_name: str
    tool_description: str
    input_json_schema: dict[str, Any]
    agent_name: str
    strict_json_schema: bool = True


@dataclass
class FunctionToolInput:
    """Data conversion friendly representation of a FunctionTool. Contains only the fields which are needed by the model
    execution to determine what tool to call, not the actual tool invocation, which remains in the workflow context.
    """

    name: str
    description: str
    params_json_schema: dict[str, Any]
    strict_json_schema: bool = True
    # The fields below were added after the first release. They are optional so that
    # activity inputs recorded in older workflow histories still decode.
    defer_loading: bool = False
    namespace: str | None = None
    namespace_description: str | None = None
    allowed_callers: list[ToolCaller] | None = None
    output_json_schema: dict[str, Any] | None = None


@dataclass
class HostedMCPToolInput:
    """Data conversion friendly representation of a HostedMCPTool. Contains only the fields which are needed by the model
    execution to determine what tool to call, not the actual tool invocation, which remains in the workflow context.
    """

    tool_config: Mcp


@dataclass
class ShellToolInput:
    """Data conversion friendly representation of a ShellTool. Contains only the fields which are needed by the model
    execution to determine what tool to call, not the actual tool invocation, which remains in the workflow context.
    """

    environment: ShellToolEnvironment | None
    name: str = "shell"


class _NoopApplyPatchEditor:
    """Satisfies the ApplyPatchEditor protocol for tool reconstruction during model calls."""

    def create_file(self, operation: Any) -> None:  # type: ignore[reportUnusedParameter]
        return None

    def update_file(self, operation: Any) -> None:  # type: ignore[reportUnusedParameter]
        return None

    def delete_file(self, operation: Any) -> None:  # type: ignore[reportUnusedParameter]
        return None


@dataclass
class ApplyPatchToolInput:
    """Data conversion friendly representation of an ApplyPatchTool."""

    name: str = "apply_patch"


@dataclass
class CustomToolInput:
    """Data conversion friendly representation of a CustomTool. Contains only the fields which are needed by the model
    execution to determine what tool to call, not the actual tool invocation, which remains in the workflow context.
    """

    tool_config: CustomToolParam


ToolInput = (
    FunctionToolInput
    | FileSearchTool
    | WebSearchTool
    | ImageGenerationTool
    | CodeInterpreterTool
    | HostedMCPToolInput
    | ShellToolInput
    | ApplyPatchToolInput
    | CustomToolInput
    | ToolSearchTool
)


@dataclass
class AgentOutputSchemaInput(AgentOutputSchemaBase):
    """Data conversion friendly representation of AgentOutputSchema."""

    output_type_name: str | None
    is_wrapped: bool
    output_schema: dict[str, Any] | None
    strict_json_schema: bool

    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""
        return self.output_type_name is None or self.output_type_name == "str"

    def is_strict_json_schema(self) -> bool:
        """Whether the JSON schema is in strict mode."""
        return self.strict_json_schema

    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of the output type."""
        if self.is_plain_text():
            raise UserError("Output type is plain text, so no JSON schema is available")
        if self.output_schema is None:
            raise UserError("Output schema is not defined")
        return self.output_schema

    def validate_json(self, json_str: str) -> Any:
        """Validate the JSON string against the schema."""
        raise NotImplementedError()

    def name(self) -> str:
        """Get the name of the output type."""
        if self.output_type_name is None:
            raise ValueError("output_type_name is None")
        return self.output_type_name


class ModelTracingInput(enum.IntEnum):
    """Conversion friendly representation of ModelTracing.

    Needed as ModelTracing is enum.Enum instead of IntEnum
    """

    DISABLED = 0
    ENABLED = 1
    ENABLED_WITHOUT_DATA = 2


class ActivityModelInput(TypedDict, total=False):
    """Input for the invoke_model_activity activity."""

    model_name: str | None
    system_instructions: str | None
    input: Required[str | list[TResponseInputItem]]
    model_settings: Required[ModelSettings]
    tools: list[ToolInput]
    output_schema: AgentOutputSchemaInput | None
    handoffs: list[HandoffInput]
    tracing: Required[ModelTracingInput]
    previous_response_id: str | None
    conversation_id: str | None
    prompt: Any | None
    # Set by the workflow when neither the agent nor the run named a model: whether
    # the agent's settings are the SDK's implicit defaults, and the run-level overrides.
    implicit_model_settings: bool
    run_model_settings: ModelSettings | None


class StreamingActivityModelInput(ActivityModelInput, total=False):
    """Input for the invoke_model_activity_streaming activity.

    Adds the streaming-only fields on top of :class:`ActivityModelInput`.
    """

    streaming_topic: Required[str]
    streaming_batch_interval: timedelta


async def _empty_on_invoke_tool(_ctx: RunContextWrapper[Any], _input: str) -> str:
    return ""


async def _empty_on_invoke_handoff(_ctx: RunContextWrapper[Any], _input: str) -> Any:
    return None


async def _noop_shell_executor(*_a: Any, **_kw: Any) -> str:
    return ""


def _raise_unknown_tool(tool: Any) -> NoReturn:
    if isinstance(tool, dict):
        raise UserError(
            "Tool input was not reconstructed into a supported tool type; "
            f"got a dict with keys: {sorted(tool)}"
        )
    raise UserError(f"Unknown tool type: {getattr(tool, 'name', type(tool).__name__)}")


def _build_tool(tool: ToolInput, env_refs: _WorkerEnvRefResolver) -> Tool:
    """Reconstruct a Tool from its data-conversion-friendly input form."""
    if isinstance(
        tool,
        (
            FileSearchTool,
            WebSearchTool,
            ImageGenerationTool,
            ToolSearchTool,
        ),
    ):
        return tool
    elif isinstance(tool, CodeInterpreterTool):
        return CodeInterpreterTool(
            tool_config=env_refs.resolve_code_interpreter_tool_config(tool.tool_config)
        )
    elif isinstance(tool, ShellToolInput):
        environment = env_refs.resolve_shell_tool_environment(tool.environment)
        # Only a local environment takes an executor.
        return ShellTool(
            name=tool.name,
            environment=environment,
            executor=_noop_shell_executor if environment["type"] == "local" else None,
        )
    elif isinstance(tool, ApplyPatchToolInput):
        return ApplyPatchTool(name=tool.name, editor=_NoopApplyPatchEditor())
    elif isinstance(tool, HostedMCPToolInput):
        return HostedMCPTool(
            tool_config=env_refs.resolve_mcp_tool_config(tool.tool_config)
        )
    elif isinstance(tool, CustomToolInput):
        return CustomTool(
            name=tool.tool_config["name"],
            description=tool.tool_config.get("description", ""),
            on_invoke_tool=_empty_on_invoke_tool,
            format=tool.tool_config.get("format"),
            defer_loading=tool.tool_config.get("defer_loading", False),
        )
    elif isinstance(tool, FunctionToolInput):
        function_tool = FunctionTool(
            name=tool.name,
            description=tool.description,
            params_json_schema=tool.params_json_schema,
            on_invoke_tool=_empty_on_invoke_tool,
            strict_json_schema=tool.strict_json_schema,
            defer_loading=tool.defer_loading,
            allowed_callers=tool.allowed_callers,
            output_json_schema=tool.output_json_schema,
        )
        if tool.namespace:
            # tool_namespace() is the only public way to attach namespace metadata.
            # The tool was already validated in the workflow, so the fallback
            # description only covers inputs built by hand without one.
            function_tool = tool_namespace(
                name=tool.namespace,
                description=tool.namespace_description or tool.namespace,
                tools=[function_tool],
            )[0]
        return function_tool
    else:
        _raise_unknown_tool(tool)  # type:ignore[reportUnreachable]


def _build_tools_and_handoffs(
    input: ActivityModelInput,
    env_refs: _WorkerEnvRefResolver,
) -> tuple[list[Tool], list[Handoff[Any, Any]]]:
    tools = [_build_tool(x, env_refs) for x in input.get("tools", [])]
    handoffs: list[Handoff[Any, Any]] = [
        Handoff(
            tool_name=x.tool_name,
            tool_description=x.tool_description,
            input_json_schema=x.input_json_schema,
            agent_name=x.agent_name,
            strict_json_schema=x.strict_json_schema,
            on_invoke_handoff=_empty_on_invoke_handoff,
        )
        for x in input.get("handoffs", [])
    ]
    return tools, handoffs


def _align_implicit_model_settings(
    input: ActivityModelInput, model: Model
) -> ModelSettings:
    """Align implicit default model settings with the model the provider resolved.

    An agent that names no model carries the Agents SDK's default settings for the SDK's
    own default model, such as a gpt-5 reasoning effort and ``verbosity``. The SDK swaps
    them for the settings that fit the model name actually used once that name is known.
    Inside a workflow the name is only known here, after this activity's provider resolves
    the model, so the same swap happens here. The workflow records whether the agent's
    settings were those implicit defaults, judged in its own environment, and the
    run-level overrides, which are re-applied after the swap. Explicit agent settings,
    inputs from older workflow workers, and models that do not expose a ``model`` name
    pass through unchanged.
    """
    model_settings = input["model_settings"]
    if input.get("model_name") is not None or not input.get("implicit_model_settings"):
        return model_settings
    resolved_name = getattr(model, "model", None)
    if not isinstance(resolved_name, str):
        return model_settings
    aligned = get_default_model_settings(resolved_name).resolve(
        input.get("run_model_settings")
    )
    # Keep the extra arguments a SandboxAgent's capabilities add in the Workflow, such as
    # the compaction threshold.
    return replace(aligned, extra_args=model_settings.extra_args)


def _retry_after(e: APIStatusError) -> timedelta | None:
    """Read the delay the provider asks for, as the OpenAI client does.

    ``retry-after-ms`` wins over ``retry-after``, which is a number of seconds or an HTTP date.
    Values that cannot be parsed, or that do not fit in a ``timedelta``, are ignored.
    """
    retry_after_ms_header = e.response.headers.get("retry-after-ms")
    if retry_after_ms_header is not None:
        try:
            return timedelta(milliseconds=float(retry_after_ms_header))
        except (OverflowError, ValueError):
            pass

    retry_after_header = e.response.headers.get("retry-after")
    if retry_after_header is None:
        return None
    try:
        return timedelta(seconds=float(retry_after_header))
    except (OverflowError, ValueError):
        pass
    retry_date = email.utils.parsedate_tz(retry_after_header)
    if retry_date is None:
        return None
    try:
        delay = timedelta(seconds=email.utils.mktime_tz(retry_date) - time.time())
    except (OverflowError, ValueError):  # a date outside the supported range
        return None
    return delay if delay > timedelta(0) else None


def _raise_for_openai_status(e: APIStatusError) -> NoReturn:
    """Translate an OpenAI APIStatusError into the right retry posture."""
    retry_after = _retry_after(e)

    should_retry_header = e.response.headers.get("x-should-retry")
    if should_retry_header == "false":
        raise ApplicationError(
            message="Non retryable OpenAI error",
            type=APIStatusError.__name__,
            non_retryable=True,
            next_retry_delay=retry_after,
        ) from e

    # Retry on 408 (Request Timeout), 409 (Conflict / often transient
    # state mismatch), 429 (Too Many Requests / rate-limited), and any
    # 5xx (server-side errors). All other 4xx codes are caller errors
    # that won't recover on retry, unless the server explicitly asks for
    # a retry via x-should-retry.
    retryable = (
        should_retry_header == "true"
        or e.response.status_code in [408, 409, 429]
        or e.response.status_code >= 500
    )
    raise ApplicationError(
        message=(
            f"{'Retryable' if retryable else 'Non retryable'} OpenAI status code: "
            f"{e.response.status_code}"
        ),
        type=APIStatusError.__name__,
        non_retryable=not retryable,
        next_retry_delay=retry_after,
    ) from e


async def _get_response(
    model: Model,
    input: ActivityModelInput,
    model_settings: ModelSettings,
    tools: list[Tool],
    handoffs: list[Handoff[Any, Any]],
) -> ModelResponse:
    """Call ``model.get_response`` with the activity input, translating status errors."""
    try:
        return await model.get_response(
            system_instructions=input.get("system_instructions"),
            input=input["input"],
            model_settings=model_settings,
            tools=tools,
            output_schema=input.get("output_schema"),
            handoffs=handoffs,
            tracing=ModelTracing(input["tracing"]),
            previous_response_id=input.get("previous_response_id"),
            conversation_id=input.get("conversation_id"),
            prompt=input.get("prompt"),
        )
    except APIStatusError as e:
        _raise_for_openai_status(e)


def _cancel_requested() -> bool:
    """Whether the workflow asked to cancel the running activity."""
    details = activity.cancellation_details()
    return details is not None and details.cancel_requested


def _default_openai_client() -> AsyncOpenAI:
    """Return the framework's default client as it is, or a client on its default key.

    A client the application set keeps its own settings, retries included, as in native
    runs. Otherwise the client has retries off, so that Temporal retries model calls.
    """
    client = _openai_shared.get_default_openai_client()
    if client is not None:
        return client
    return AsyncOpenAI(api_key=_openai_shared.get_default_openai_key(), max_retries=0)


class ModelActivity:
    """Class wrapper for model invocation activities to allow model customization. By default, we use an OpenAIProvider with retries disabled.
    Disabling retries in your model of choice is recommended to allow activity retries to define the retry model.
    """

    def __init__(
        self,
        model_provider: ModelProvider | None = None,
        resolvable_worker_env_vars: Collection[str] | AllowAllWorkerEnvVars = (),
        *,
        heartbeat_throttle: HeartbeatThrottle | None = None,
    ):
        """Initialize the activity with a model provider.

        ``heartbeat_throttle`` is the Worker's heartbeat throttling, which background
        model calls need to tell whether a response's ID has reached the server.
        """
        self._heartbeat_throttle = heartbeat_throttle or HeartbeatThrottle()
        # The client of the default provider, which background model calls also use.
        self._background_client: AsyncOpenAI | None = None
        self._background_provider: ModelProvider | None = None
        if model_provider:
            self._model_provider = model_provider
        else:
            client = _default_openai_client()
            self._model_provider = OpenAIProvider(openai_client=client)
            # Only background model calls go through the wrapper, which sends every
            # request with the same client.
            self._background_client = client
            self._background_provider = OpenAIProvider(
                # A duck-typed stand-in, as the Agents SDK's own wrappers are.
                openai_client=cast(
                    AsyncOpenAI, cast(object, BackgroundOpenAIClient(client))
                )
            )
        self._env_refs = _WorkerEnvRefResolver(resolvable_worker_env_vars)

    @activity.defn
    @auto_heartbeater
    async def invoke_model_activity(self, input: ActivityModelInput) -> ModelResponse:
        """Activity that invokes a model with the given input."""
        model = self._model_provider.get_model(input.get("model_name"))
        model_settings = _align_implicit_model_settings(input, model)
        tools, handoffs = _build_tools_and_handoffs(input, self._env_refs)

        if requests_background(model_settings.extra_args, model_settings.extra_body):
            background_model = await self._background_model(input, model)
            if background_model is not None:
                return await self._invoke_in_background(
                    background_model, input, model_settings, tools, handoffs
                )

        return await _get_response(model, input, model_settings, tools, handoffs)

    async def _background_model(
        self, input: ActivityModelInput, model: Model
    ) -> Model | None:
        """The model for a background call, None to call ``model`` as in a native run.

        Raises a non-retryable ``ApplicationError`` before any request for a call that
        would get an unfinished response back. Other models, such as the Responses
        websocket and Chat Completions models, handle ``background`` as in a native run.
        """
        # Exact type: subclasses such as OpenAIResponsesWSModel send requests another way.
        if type(model) is not OpenAIResponsesModel:
            await self._abandon_inherited()
            return None
        if self._background_provider is None:
            await self._refuse_background(
                "background=True needs the plugin's default model provider. With a "
                "custom model provider, the Agents SDK returns the background "
                "response unfinished."
            )
        if activity.info().is_local:
            await self._refuse_background(
                "background=True needs a regular activity. Local activities record no "
                "heartbeats, so a retry could not continue the background response. "
                "Turn off use_local_activity for this call."
            )
        if input.get("conversation_id") is not None:
            await self._refuse_background(
                "background=True does not support conversation_id yet. Use "
                "previous_response_id to chain responses."
            )
        assert self._background_provider is not None
        background_model = self._background_provider.get_model(input.get("model_name"))
        if type(background_model) is not OpenAIResponsesModel:
            await self._abandon_inherited()
            return None
        return background_model

    async def _abandon_inherited(self) -> None:
        """Cancel an earlier attempt's background response this attempt won't continue.

        An attempt on another worker, one that runs the call in the background, may
        have started it.
        """
        response_id = inherited_response_id()
        if response_id is None:
            return
        if self._background_client is not None:
            await BackgroundCall(
                self._background_client, response_id, seed=False
            ).cancel("this worker does not run the call in the background")
            # So that a later attempt doesn't find it again.
            set_heartbeat_details()
        else:
            activity.logger.warning(
                "Background response %s from an earlier attempt can't be cancelled "
                "by this worker; it may run to the end and be billed",
                response_id,
            )

    async def _refuse_background(self, message: str) -> NoReturn:
        await self._abandon_inherited()
        raise ApplicationError(message, type=UNSUPPORTED_ERROR_TYPE, non_retryable=True)

    async def _invoke_in_background(
        self,
        model: Model,
        input: ActivityModelInput,
        model_settings: ModelSettings,
        tools: list[Tool],
        handoffs: list[Handoff[Any, Any]],
    ) -> ModelResponse:
        """Run a background call that a retried attempt continues instead of repeating."""
        assert self._background_client is not None
        call = BackgroundCall(
            self._background_client,
            inherited_response_id(),
            throttle=self._heartbeat_throttle,
        )
        with attempt_call(call):
            try:
                response = await _get_response(
                    model, input, model_settings, tools, handoffs
                )
            except asyncio.CancelledError:
                try:
                    reason = await cancel_reason(call)
                except Exception:
                    # Report the cancel, not a failure deciding what to do with it.
                    activity.logger.warning(
                        "Could not tell whether another attempt will continue the "
                        "background response, so it is left running",
                        exc_info=True,
                    )
                    reason = None
                if reason is not None:
                    await call.cancel(reason)
                raise
            except Exception as e:
                if is_final_failure(e):
                    await call.cancel("the call failed with no retry to follow")
                raise
        if not call.intercepted:
            # The Agents SDK sent the request some other way, and its response may be
            # unfinished.
            if response.response_id is not None:
                call.adopt(response.response_id)
                await call.cancel("the plugin could not wait for it")
            raise ApplicationError(
                "background=True is not supported with this version of the Agents "
                "SDK: the plugin could not wait for the background response.",
                type=UNSUPPORTED_ERROR_TYPE,
                non_retryable=True,
            )
        return response

    @activity.defn
    async def invoke_model_activity_streaming(
        self, input: StreamingActivityModelInput
    ) -> list[TResponseStreamEvent]:
        """Streaming-aware model activity.

        .. warning::
            Streaming support is experimental and may change in future
            versions.

        Calls ``model.stream_response()`` and returns the collected list
        of native OpenAI stream events. The workflow's
        ``Model.stream_response`` stub yields these to the agents
        framework, which builds the final ``ModelResponse`` from the
        terminal ``ResponseCompletedEvent``.

        Each event is also published to the workflow's stream on
        ``streaming_topic`` so external consumers (UIs, tracing,
        etc.) can observe events as they arrive.

        When the workflow cancels it, for example for a tripped input
        guardrail, ``RunResultStreaming.cancel()`` or a
        ``ModelSettings.timeout``, it publishes a ``ResponseErrorEvent``
        with the code ``cancelled`` after its events and stops. Its
        ``param`` is the ID from the call's ``response.created`` event, if
        the model sent one. That event ends this model call, not the run: a
        ``ModelSettings.retry`` or the next model call can publish after it.

        Heartbeats run on a background task via ``heartbeating`` so
        long initial-token latency or long pauses between chunks do not
        trip ``heartbeat_timeout``.
        """
        model = self._model_provider.get_model(input.get("model_name"))
        model_settings = _align_implicit_model_settings(input, model)
        tools, handoffs = _build_tools_and_handoffs(input, self._env_refs)

        topic = input["streaming_topic"]
        batch_interval = input.get(
            "streaming_batch_interval", timedelta(milliseconds=100)
        )
        events: list[TResponseStreamEvent] = []
        # From the call's response.created event, to name the response a cancel ends.
        response_id: str | None = None

        stream = WorkflowStreamClient.from_within_activity(
            batch_interval=batch_interval
        )
        # TResponseStreamEvent is a typing.Annotated[Union[...]] — a typing
        # special form, not a class — so it cannot be passed as type[T].
        # Leave the topic untyped (default Any); subscribers that want
        # typed decode can pass result_type=TResponseStreamEvent on
        # their own subscribe call.
        events_topic = stream.topic(topic)
        try:
            async with heartbeating():
                async with stream:
                    try:
                        async for event in model.stream_response(
                            system_instructions=input.get("system_instructions"),
                            input=input["input"],
                            model_settings=model_settings,
                            tools=tools,
                            output_schema=input.get("output_schema"),
                            handoffs=handoffs,
                            tracing=ModelTracing(input["tracing"]),
                            previous_response_id=input.get("previous_response_id"),
                            conversation_id=input.get("conversation_id"),
                            prompt=input.get("prompt"),
                        ):
                            # OpenAI models set defer_build=True, so an event's pydantic
                            # schema may still be an unbuilt placeholder.
                            type(event).model_rebuild()
                            if isinstance(event, ResponseCreatedEvent):
                                response_id = event.response.id
                            events.append(event)
                            events_topic.publish(event)
                    except APIStatusError as e:
                        _raise_for_openai_status(e)
            # Leaving the context managers swallows a cancel that arrives while they
            # stop the stream's background flusher or the heartbeats, so raise that
            # cancel here.
            if _cancel_requested():
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            # When the workflow cancelled the call, for example because a parallel
            # input guardrail tripped, tell subscribers the response ends here. The
            # cancel can arrive while the model streams or while leaving the context
            # managers flushes the events, so publish after both, behind the events
            # still buffered. Other cancels, such as a Worker shutdown or a heartbeat
            # timeout, can be retried, so publish nothing and let the retry answer.
            if _cancel_requested():
                # Best effort: if the publish fails, still report the cancel.
                try:
                    async with heartbeating():
                        events_topic.publish(
                            ResponseErrorEvent(
                                type="error",
                                code="cancelled",
                                message="The model call was cancelled.",
                                param=response_id,
                                sequence_number=len(events),
                            )
                        )
                        await stream.flush()
                except Exception:
                    activity.logger.debug(
                        "Could not publish the cancelled event", exc_info=True
                    )
            raise

        return events
