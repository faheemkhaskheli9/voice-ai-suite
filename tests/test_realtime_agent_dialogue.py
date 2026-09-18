import pytest

from realtime_agent.dialogue import (
    FinalResponse,
    ToolCall,
    ToolCallStep,
    ToolNotFound,
    ToolRegistry,
    ToolResult,
)


def test_tool_registry_register_and_get():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)

    fn = registry.get("add")

    assert fn(2, 3) == 5
    assert registry.names() == ("add",)
    assert "add" in registry


def test_tool_registry_unknown_tool_raises_tool_not_found():
    registry = ToolRegistry()
    registry.register("add", lambda a, b: a + b)

    with pytest.raises(ToolNotFound, match="unknown_tool"):
        registry.get("unknown_tool")


def test_tool_result_ok_reflects_error_field():
    call = ToolCall(name="add", arguments={"a": 1, "b": 2})

    success = ToolResult(call=call, output=3)
    failure = ToolResult(call=call, error="boom")

    assert success.ok is True
    assert failure.ok is False


def test_tool_call_step_and_final_response_are_distinguishable():
    step = ToolCallStep(calls=(ToolCall(name="add", arguments={}),))
    final = FinalResponse(text="done")

    assert isinstance(step, ToolCallStep)
    assert not isinstance(step, FinalResponse)
    assert isinstance(final, FinalResponse)
    assert final.text == "done"
