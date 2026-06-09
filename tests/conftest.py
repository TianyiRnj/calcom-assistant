from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from cal_client import CalClient
from schemas import Attendee, Booking, Slot


# ---------------------------------------------------------------------------
# Environment — no real credentials needed
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAL_API_KEY", "test-cal-key")
    monkeypatch.setenv("CAL_API_BASE_URL", "https://api.cal.com/v2")
    monkeypatch.setenv("CAL_USERNAME", "testuser")
    monkeypatch.setenv("CAL_TIMEZONE", "America/New_York")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_MODEL", "test-openai-model")


# ---------------------------------------------------------------------------
# Sample data (far-future dates so "past date" checks never fire)
# ---------------------------------------------------------------------------

_T0 = datetime(2050, 9, 5, 14, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2050, 9, 5, 14, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_booking() -> Booking:
    return Booking(
        uid="uid-123",
        title="Intro Call",
        start=_T0,
        end=_T1,
        attendees=[Attendee(name="Jane", email="jane@example.com")],
        status="accepted",
    )


@pytest.fixture
def sample_cancelled_booking() -> Booking:
    return Booking(
        uid="uid-456",
        title="Cancelled Call",
        start=_T0,
        end=_T1,
        attendees=[Attendee(name="Jane", email="jane@example.com")],
        status="cancelled",
    )


@pytest.fixture
def sample_slot() -> Slot:
    return Slot(start=_T0, end=_T1)


# ---------------------------------------------------------------------------
# CalClient with injected mock httpx client
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_http() -> MagicMock:
    return MagicMock()


@pytest.fixture
def cal_client(mock_http: MagicMock) -> CalClient:
    return CalClient(
        api_key="test-cal-key",
        base_url="https://api.cal.com/v2",
        event_type_id=42,
        username="testuser",
        timezone="America/New_York",
        _client=mock_http,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@pytest.fixture
def session_state() -> dict:
    return {"messages": [], "pending_action": None, "available_slots": []}


# ---------------------------------------------------------------------------
# Module-level helpers (not fixtures)
# ---------------------------------------------------------------------------


def make_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.is_success = 200 <= status_code < 300
    r.json.return_value = json_data if json_data is not None else {}
    return r


def openai_response(intent_type: str, **fields: Any) -> MagicMock:
    """Build a mock LLM response in ExtractedIntent format.

    Accepts new-style ExtractedIntent fields directly, or old-style UserIntent fields
    (attendee_name, attendee_email, start_time, end_time, duration_minutes, source_end_time,
    time_preference, time_granularity) which are mapped automatically for backward compat.
    """
    data: dict[str, Any] = {"intent_type": intent_type}

    # attendee_name / attendee_email → attendees list
    name = fields.pop("attendee_name", None)
    email = fields.pop("attendee_email", None)
    if name is not None or email is not None:
        attendee: dict[str, Any] = {}
        if name is not None:
            attendee["name"] = name
        if email is not None:
            attendee["email"] = email
        if "attendees" not in fields:
            data["attendees"] = [attendee]

    # start_time → appropriate new field based on intent
    if "start_time" in fields:
        st = fields.pop("start_time")
        if intent_type == "list":
            data.setdefault("date_range_start", st)
        elif intent_type == "book":
            data.setdefault("target_start_time", st)
        elif intent_type == "cancel":
            data.setdefault("source_start_time", st)
        elif intent_type == "reschedule":
            data.setdefault("source_start_time", st)

    # end_time → date_range_end (list) or dropped (cancel/reschedule source_end_time gone)
    if "end_time" in fields:
        et = fields.pop("end_time")
        if intent_type == "list":
            data.setdefault("date_range_end", et)
        # For other intents, end_time is no longer used — duration is used instead.

    # duration_minutes → target_duration_minutes (book/reschedule) or source_duration_minutes
    if "duration_minutes" in fields:
        dm = fields.pop("duration_minutes")
        if intent_type in ("book", "reschedule"):
            data.setdefault("target_duration_minutes", dm)
        else:
            data.setdefault("source_duration_minutes", dm)

    # source_end_time no longer exists in ExtractedIntent — drop it
    fields.pop("source_end_time", None)
    # time_preference and time_granularity no longer in ExtractedIntent — drop them
    fields.pop("time_preference", None)
    fields.pop("time_granularity", None)

    # Remaining fields pass through directly (search_text, booking_uid, timezone,
    # relative_time_qualifier, event_name, source_start_time, source_duration_minutes,
    # target_start_time, target_duration_minutes, date_range_start, date_range_end, attendees)
    data.update(fields)

    text = json.dumps(data)
    msg = MagicMock()
    msg.output_text = text
    return msg


def booking_dict(
    uid: str = "uid-123",
    title: str = "Intro Call",
    status: str = "accepted",
) -> dict:
    return {
        "uid": uid,
        "title": title,
        "start": _T0.isoformat(),
        "end": _T1.isoformat(),
        "status": status,
        "eventTypeId": 42,
        "attendees": [{"name": "Jane", "email": "jane@example.com"}],
    }
