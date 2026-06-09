from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import app
from assistant import MAX_USER_MESSAGE_CHARS
from schemas import (
    BookingRequest,
    CalClientError,
    CancelRequest,
    PendingAction,
    RescheduleRequest,
)


# ---------------------------------------------------------------------------
# App helpers
# ---------------------------------------------------------------------------


_T = datetime(2050, 9, 5, 14, 0, tzinfo=timezone.utc)


def test_not_in_confirmation_when_no_pending():
    assert not app._in_confirmation_phase(None)


def test_not_in_confirmation_when_pending_has_no_request():
    pending = PendingAction(action_type="book")
    assert not app._in_confirmation_phase(pending)


def test_in_confirmation_with_booking_request():
    req = BookingRequest(
        attendee_name="Jane",
        attendee_email="jane@example.com",
        start_time=_T,
        duration_minutes=30,
        timezone="UTC",
        event_type_id=42,
    )
    pending = PendingAction(action_type="book", booking_request=req)
    assert app._in_confirmation_phase(pending)


def test_in_confirmation_with_cancel_request():
    pending = PendingAction(
        action_type="cancel", cancel_request=CancelRequest(booking_uid="uid-123")
    )
    assert app._in_confirmation_phase(pending)


def test_in_confirmation_with_reschedule_request():
    pending = PendingAction(
        action_type="reschedule",
        reschedule_request=RescheduleRequest(booking_uid="uid-123", new_start_time=_T),
    )
    assert app._in_confirmation_phase(pending)


def test_missing_required_env_reports_blank_values(monkeypatch):
    monkeypatch.setenv("CAL_API_KEY", "test-cal-key")
    monkeypatch.setenv("CAL_USERNAME", "testuser")
    monkeypatch.setenv("OPENAI_API_KEY", " ")
    monkeypatch.setenv("LLM_MODEL", "")

    missing = app._missing_required_env()

    assert "OPENAI_API_KEY" in missing
    assert "LLM_MODEL" in missing
    assert "CAL_API_KEY" not in missing
    assert "CAL_USERNAME" not in missing


def test_dispatch_error_reply_formats_cal_400():
    reply = app._dispatch_error_reply(
        CalClientError("Email verification code is required", 400)
    )

    assert "rejected" in reply.lower()
    assert "Email verification code is required" in reply


def test_dispatch_catches_cal_client_error(monkeypatch):
    state = {"messages": [], "pending_action": None, "available_slots": []}
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(
        app,
        "handle_message",
        MagicMock(side_effect=CalClientError("Bad request", 400)),
    )

    app._dispatch("yes", MagicMock())

    assert state["messages"][0] == {"role": "user", "content": "yes"}
    assert state["messages"][1]["role"] == "assistant"
    assert "rejected" in state["messages"][1]["content"].lower()


def test_overlong_input_returns_length_message(monkeypatch):
    state = {"messages": [], "pending_action": None, "available_slots": []}
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "handle_message", MagicMock(return_value="unused"))

    app._dispatch("x" * (MAX_USER_MESSAGE_CHARS + 1), MagicMock())

    assert state["messages"][-1]["role"] == "assistant"
    assert state["messages"][-1]["content"] == (
        "That message is a bit long — can you shorten it?"
    )


def test_overlong_input_does_not_call_handle_message(monkeypatch):
    state = {"messages": [], "pending_action": None, "available_slots": []}
    handle = MagicMock(return_value="unused")
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "handle_message", handle)

    app._dispatch("x" * (MAX_USER_MESSAGE_CHARS + 1), MagicMock())

    handle.assert_not_called()


def test_overlong_input_does_not_store_raw_text(monkeypatch):
    state = {"messages": [], "pending_action": None, "available_slots": []}
    raw = "x" * (MAX_USER_MESSAGE_CHARS + 1)
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "handle_message", MagicMock(return_value="unused"))

    app._dispatch(raw, MagicMock())

    assert all(raw not in message["content"] for message in state["messages"])


def test_empty_input_ignored(monkeypatch):
    state = {"messages": [], "pending_action": None, "available_slots": []}
    handle = MagicMock(return_value="unused")
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "handle_message", handle)

    app._dispatch("", MagicMock())

    assert state["messages"] == []
    handle.assert_not_called()


def test_whitespace_input_ignored(monkeypatch):
    state = {"messages": [], "pending_action": None, "available_slots": []}
    handle = MagicMock(return_value="unused")
    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "handle_message", handle)

    app._dispatch("   ", MagicMock())

    assert state["messages"] == []
    handle.assert_not_called()


# ===========================================================================
# TestOptimisticRendering
# ===========================================================================
# These tests cover the _dispatch (button) path and the assistant-level
# _history_override parameter.  The Streamlit chat-input optimistic path is
# exercised by the integration smoke test because it requires a full Streamlit
# rendering context.


class TestOptimisticRendering:
    def test_dispatch_button_path_appends_user_then_assistant(self, monkeypatch):
        """Button-path _dispatch appends user message then assistant message in order."""
        state = {"messages": [], "pending_action": None, "available_slots": []}
        monkeypatch.setattr(app.st, "session_state", state)
        monkeypatch.setattr(app, "handle_message", MagicMock(return_value="All good."))

        app._dispatch("hello", MagicMock())

        assert len(state["messages"]) == 2
        assert state["messages"][0] == {"role": "user", "content": "hello"}
        assert state["messages"][1] == {"role": "assistant", "content": "All good."}

    def test_dispatch_button_path_does_not_pass_history_override(self, monkeypatch):
        """Button-path _dispatch passes no _history_override so the handler uses full history."""
        state = {"messages": [], "pending_action": None, "available_slots": []}
        monkeypatch.setattr(app.st, "session_state", state)
        handle = MagicMock(return_value="reply")
        monkeypatch.setattr(app, "handle_message", handle)

        app._dispatch("hello", MagicMock())

        _, call_kwargs = handle.call_args
        assert "_history_override" not in call_kwargs

    def test_dispatch_redacts_api_key_before_storing(self, monkeypatch):
        """A message containing a fake API key must not appear verbatim in session messages."""
        state = {"messages": [], "pending_action": None, "available_slots": []}
        monkeypatch.setattr(app.st, "session_state", state)
        monkeypatch.setattr(app, "handle_message", MagicMock(return_value="ok"))

        raw = "my secret sk-1234567890abcdefghijklmnopqrs"
        app._dispatch(raw, MagicMock())

        stored_user = state["messages"][0]["content"]
        assert "sk-1234567890abcdefghijklmnopqrs" not in stored_user

    def test_dispatch_exception_restores_pending_and_slots(self, monkeypatch):
        """If handle_message raises, pending_action and available_slots are rolled back."""
        from schemas import CancelRequest

        original_pending = PendingAction(
            action_type="cancel",
            cancel_request=CancelRequest(booking_uid="uid-1"),
        )
        state = {
            "messages": [],
            "pending_action": original_pending,
            "available_slots": ["slot-a"],
        }
        monkeypatch.setattr(app.st, "session_state", state)
        monkeypatch.setattr(
            app,
            "handle_message",
            MagicMock(side_effect=RuntimeError("unexpected")),
        )

        app._dispatch("yes", MagicMock())

        assert state["pending_action"] is original_pending
        assert state["available_slots"] == ["slot-a"]
        # An error message should still be appended
        assert len(state["messages"]) == 2
        assert state["messages"][1]["role"] == "assistant"

    def test_handle_message_accepts_history_override_param(self):
        """handle_message signature accepts _history_override without crashing."""
        from assistant import handle_message
        import inspect

        sig = inspect.signature(handle_message)
        assert "_history_override" in sig.parameters

    def test_repeated_dispatch_appends_correct_count(self, monkeypatch):
        """Each _dispatch call appends exactly one user + one assistant message."""
        state = {"messages": [], "pending_action": None, "available_slots": []}
        monkeypatch.setattr(app.st, "session_state", state)
        monkeypatch.setattr(app, "handle_message", MagicMock(return_value="ok"))
        cal = MagicMock()

        app._dispatch("first", cal)
        app._dispatch("second", cal)

        assert len(state["messages"]) == 4
        assert state["messages"][0]["role"] == "user"
        assert state["messages"][1]["role"] == "assistant"
        assert state["messages"][2]["role"] == "user"
        assert state["messages"][3]["role"] == "assistant"
