from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import app
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
