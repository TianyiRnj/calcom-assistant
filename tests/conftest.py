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
    monkeypatch.setenv("CAL_EVENT_TYPE_ID", "42")
    monkeypatch.setenv("CAL_USERNAME", "testuser")
    monkeypatch.setenv("CAL_TIMEZONE", "America/New_York")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4-nano")


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
    text = json.dumps({"intent_type": intent_type, **fields})
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
        "attendees": [{"name": "Jane", "email": "jane@example.com"}],
    }
