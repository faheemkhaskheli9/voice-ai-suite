import pytest

from realtime_agent.memory import ConversationMemory, Turn


def test_new_session_has_empty_history():
    memory = ConversationMemory()

    assert memory.history("caller-a") == ()


def test_add_turn_appends_to_history_oldest_first():
    memory = ConversationMemory()

    memory.add_turn("caller-a", "hi", "hello")
    memory.add_turn("caller-a", "how are you?", "doing well")

    assert memory.history("caller-a") == (
        Turn(user_text="hi", agent_text="hello"),
        Turn(user_text="how are you?", agent_text="doing well"),
    )


def test_sessions_do_not_leak_into_each_other():
    memory = ConversationMemory()

    memory.add_turn("caller-a", "hi from a", "hello a")
    memory.add_turn("caller-b", "hi from b", "hello b")

    assert [t.user_text for t in memory.history("caller-a")] == ["hi from a"]
    assert [t.user_text for t in memory.history("caller-b")] == ["hi from b"]


def test_history_is_bounded_by_max_turns_sliding_window():
    memory = ConversationMemory(max_turns=2)

    memory.add_turn("caller-a", "one", "1")
    memory.add_turn("caller-a", "two", "2")
    memory.add_turn("caller-a", "three", "3")

    assert [t.user_text for t in memory.history("caller-a")] == ["two", "three"]


def test_clear_drops_a_sessions_history():
    memory = ConversationMemory()
    memory.add_turn("caller-a", "hi", "hello")

    memory.clear("caller-a")

    assert memory.history("caller-a") == ()


def test_clear_of_unknown_session_is_a_noop():
    memory = ConversationMemory()

    memory.clear("never-seen")  # must not raise


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError, match="max_turns"):
        ConversationMemory(max_turns=0)
