"""Tests for assistant.py — intent extraction, error handling, and field collection."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from assistant import extract_intent, handle_message
from cal_client import CalClient
from schemas import (
    AssistantError,
    Attendee,
    Booking,
    BookingDraft,
    CalClientError,
    IntentType,
    IntentValidationError,
    PendingAction,
    UserIntent,
)

from tests.conftest import openai_response, booking_dict

_T0 = datetime(2050, 9, 5, 14, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2050, 9, 5, 14, 30, 0, tzinfo=timezone.utc)


def _mock_openai(return_value: MagicMock):
    return patch(
        "assistant._create_openai_client",
        return_value=MagicMock(
            responses=MagicMock(create=MagicMock(return_value=return_value))
        ),
    )


# ===========================================================================
# TestIntentExtraction
# ===========================================================================


class TestIntentExtraction:
    def test_list_intent(self) -> None:
        resp = openai_response("list", start_time=_T0.isoformat(), end_time=_T1.isoformat())
        with _mock_openai(resp):
            intent = extract_intent("What's on my calendar tomorrow?", [])
        assert intent.intent_type == IntentType.list

    def test_book_intent_extracts_duration_and_name(self) -> None:
        resp = openai_response("book", attendee_name="Jane", duration_minutes=30)
        with _mock_openai(resp):
            intent = extract_intent("Book a 30-minute intro with Jane", [])
        assert intent.intent_type == IntentType.book
        assert intent.attendee_name == "Jane"
        assert intent.duration_minutes == 30

    def test_cancel_intent_extracts_search_text(self) -> None:
        resp = openai_response("cancel", search_text="Jane")
        with _mock_openai(resp):
            intent = extract_intent("Cancel my call with Jane", [])
        assert intent.intent_type == IntentType.cancel
        assert intent.search_text is not None
        assert "Jane" in intent.search_text

    def test_reschedule_intent_extracts_time_preference(self) -> None:
        resp = openai_response("reschedule", time_preference="later today")
        with _mock_openai(resp):
            intent = extract_intent("Move my 3pm to later today", [])
        assert intent.intent_type == IntentType.reschedule
        assert intent.time_preference is not None

    def test_unknown_intent(self) -> None:
        resp = openai_response("unknown")
        with _mock_openai(resp):
            intent = extract_intent("Send Jane the notes", [])
        assert intent.intent_type == IntentType.unknown


# ===========================================================================
# TestAssistantErrors — LLM/JSON failures
# ===========================================================================


class TestAssistantErrors:
    def test_llm_api_failure_raises_assistant_error_reason_llm_failure(self) -> None:
        with patch("assistant._create_openai_client") as mock_create_client:
            mock_create_client.return_value.responses.create.side_effect = Exception("network error")
            with pytest.raises(AssistantError) as exc_info:
                extract_intent("Book something", [])
        assert exc_info.value.reason == "llm_failure"

    def test_bad_json_raises_assistant_error_reason_bad_json(self) -> None:
        bad_msg = MagicMock()
        bad_msg.output_text = "not valid json {{{{"
        with _mock_openai(bad_msg):
            with pytest.raises(AssistantError) as exc_info:
                extract_intent("Book something", [])
        assert exc_info.value.reason == "bad_json"

    def test_missing_intent_field_raises_assistant_error_reason_missing_field(self) -> None:
        empty_msg = MagicMock()
        empty_msg.output_text = "{}"
        with _mock_openai(empty_msg):
            with pytest.raises(AssistantError) as exc_info:
                extract_intent("Book something", [])
        assert exc_info.value.reason == "missing_field"

    def test_assistant_error_is_not_cal_client_error(self) -> None:
        with patch("assistant._create_openai_client") as mock_create_client:
            mock_create_client.return_value.responses.create.side_effect = Exception("boom")
            with pytest.raises(Exception) as exc_info:
                extract_intent("Book something", [])
        assert not isinstance(exc_info.value, CalClientError)


# ===========================================================================
# TestIntentValidationErrors — user data failures
# ===========================================================================


class TestIntentValidationErrors:
    def test_impossible_date_string_raises_intent_validation_error_invalid_date(self) -> None:
        """Valid JSON but impossible date value → IntentValidationError(reason='invalid_date')."""
        bad_date_msg = MagicMock()
        bad_date_msg.output_text = '{"intent_type": "book", "start_time": "2050-02-30T14:00:00+00:00"}'
        with _mock_openai(bad_date_msg):
            with pytest.raises(IntentValidationError) as exc_info:
                extract_intent("Book on February 30", [])
        assert exc_info.value.reason == "invalid_date"

    def test_intent_validation_error_is_not_assistant_error(self) -> None:
        bad_date_msg = MagicMock()
        bad_date_msg.output_text = '{"intent_type": "book", "start_time": "2050-02-30T14:00:00+00:00"}'
        with _mock_openai(bad_date_msg):
            with pytest.raises(Exception) as exc_info:
                extract_intent("Book on February 30", [])
        assert not isinstance(exc_info.value, AssistantError)

    def test_intent_validation_error_is_not_cal_client_error(self) -> None:
        bad_date_msg = MagicMock()
        bad_date_msg.output_text = '{"intent_type": "book", "start_time": "2050-02-30T14:00:00+00:00"}'
        with _mock_openai(bad_date_msg):
            with pytest.raises(Exception) as exc_info:
                extract_intent("Book on February 30", [])
        assert not isinstance(exc_info.value, CalClientError)


# ===========================================================================
# TestHandleMessageErrorRecovery
# ===========================================================================


class TestHandleMessageErrorRecovery:
    def _state(self) -> dict:
        return {"messages": [], "pending_action": None, "available_slots": []}

    def _cal(self) -> MagicMock:
        return MagicMock(spec=CalClient)

    def test_handle_message_bad_json_returns_rephrase_prompt(self) -> None:
        with patch("assistant.extract_intent", side_effect=AssistantError("bad json", reason="bad_json")):
            reply = handle_message("asdfasdf", self._state(), self._cal())
        assert "rephrase" in reply.lower() or "catch" in reply.lower() or "understand" in reply.lower()
        # Must NOT contain date-error language
        assert "date" not in reply.lower()

    def test_handle_message_llm_failure_returns_retry_prompt(self) -> None:
        with patch("assistant.extract_intent", side_effect=AssistantError("down", reason="llm_failure")):
            reply = handle_message("Book something", self._state(), self._cal())
        assert "try again" in reply.lower() or "trouble" in reply.lower()
        # Must NOT contain date-error language
        assert "date" not in reply.lower()

    def test_handle_message_invalid_date_returns_date_prompt(self) -> None:
        with patch(
            "assistant.extract_intent",
            side_effect=IntentValidationError("bad date", reason="invalid_date"),
        ):
            reply = handle_message("Book on February 30", self._state(), self._cal())
        assert "date" in reply.lower()
        # Must NOT contain "rephrase" (that's the bad_json message)
        assert "rephrase" not in reply.lower()

    def test_handle_message_past_date_returns_future_only_message(self) -> None:
        with patch(
            "assistant.extract_intent",
            side_effect=IntentValidationError("past", reason="past_date"),
        ):
            reply = handle_message("Book something yesterday", self._state(), self._cal())
        assert "future" in reply.lower() or "past" in reply.lower()

    def test_handle_message_invalid_email_returns_email_prompt(self) -> None:
        with patch(
            "assistant.extract_intent",
            side_effect=IntentValidationError("bad email", reason="invalid_email"),
        ):
            reply = handle_message("Book with notanemail", self._state(), self._cal())
        assert "email" in reply.lower()


# ===========================================================================
# TestMissingFields
# ===========================================================================


class TestMissingFields:
    def _state(self) -> dict:
        return {"messages": [], "pending_action": None, "available_slots": []}

    def _cal(self) -> MagicMock:
        m = MagicMock(spec=CalClient)
        m.find_slots.return_value = []
        return m

    def test_asks_for_missing_email(self) -> None:
        intent = UserIntent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            duration_minutes=30,
            time_preference="Thursday afternoon",
        )
        with patch("assistant.extract_intent", return_value=intent):
            state = self._state()
            reply = handle_message("Book a call with Jane", state, self._cal())
        assert "email" in reply.lower()
        assert state["pending_action"] is not None
        assert state["pending_action"].booking_draft is not None
        assert state["pending_action"].booking_draft.attendee_name == "Jane"

    def test_asks_for_missing_name(self) -> None:
        intent = UserIntent(
            intent_type=IntentType.book,
            attendee_email="jane@example.com",
            duration_minutes=30,
            time_preference="Thursday afternoon",
        )
        with patch("assistant.extract_intent", return_value=intent):
            state = self._state()
            reply = handle_message("Book a call", state, self._cal())
        assert "name" in reply.lower()

    def test_asks_one_field_at_a_time(self) -> None:
        """When both name and email are missing, only one question is asked."""
        intent = UserIntent(
            intent_type=IntentType.book,
            duration_minutes=30,
            time_preference="Thursday afternoon",
        )
        with patch("assistant.extract_intent", return_value=intent):
            state = self._state()
            reply = handle_message("Book a call", state, self._cal())
        # Should ask for exactly one thing — not both at once
        email_ask = "email" in reply.lower()
        name_ask = "name" in reply.lower()
        assert email_ask ^ name_ask, f"Expected exactly one field asked, got: {reply!r}"

    def test_draft_merges_on_follow_up(self) -> None:
        """Second turn merges email into the existing draft.

        After the merge, start_time is still None (only time_preference="Thursday"
        was provided), so find_slots is NOT called — the assistant asks for a
        specific date/time instead.
        """
        intent1 = UserIntent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            duration_minutes=30,
            time_preference="Thursday",
        )
        intent2 = UserIntent(
            intent_type=IntentType.book,
            attendee_email="jane@example.com",
        )
        state = self._state()
        cal = self._cal()
        with patch("assistant.extract_intent", side_effect=[intent1, intent2]):
            handle_message("Book with Jane on Thursday", state, cal)  # asks for email
            handle_message("jane@example.com", state, cal)  # provides email; asks for specific time

        draft = state["pending_action"].booking_draft if state["pending_action"] else None
        # Draft should exist with both name and email merged in
        assert draft is not None
        assert draft.attendee_name == "Jane"
        assert draft.attendee_email == "jane@example.com"
        # start_time is still None → find_slots NOT called
        cal.find_slots.assert_not_called()


# ===========================================================================
# TestTimezoneAwareness
# ===========================================================================


class TestTimezoneAwareness:
    def test_slot_rejects_naive_datetime(self) -> None:
        from schemas import Slot

        with pytest.raises(ValidationError):
            Slot(start=datetime(2050, 9, 5, 14, 0, 0), end=datetime(2050, 9, 5, 14, 30, 0))

    def test_booking_rejects_naive_start(self) -> None:
        with pytest.raises(ValidationError):
            Booking(
                uid="x",
                title="T",
                start=datetime(2050, 9, 5, 14, 0, 0),  # naive
                end=_T1,
            )

    def test_naive_datetime_in_flow_returns_friendly_error(self) -> None:
        """If validation error somehow surfaces in handle_message, it returns a friendly reply."""
        # Simulate an IntentValidationError from parsing an impossible date
        with patch(
            "assistant.extract_intent",
            side_effect=IntentValidationError("naive datetime", reason="invalid_date"),
        ):
            state = {"messages": [], "pending_action": None, "available_slots": []}
            reply = handle_message("Book something", state, MagicMock(spec=CalClient))
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert "date" in reply.lower() or "time" in reply.lower()


# ===========================================================================
# TestSchemaIntegrity
# ===========================================================================


class TestSchemaIntegrity:
    def test_pending_action_matching_bookings_not_shared(self) -> None:
        """Two PendingAction instances must not share the same matching_bookings list."""
        a = PendingAction(action_type="cancel")
        b = PendingAction(action_type="cancel")
        a.matching_bookings.append(
            Booking(uid="x", title="T", start=_T0, end=_T1)
        )
        assert len(b.matching_bookings) == 0, (
            "PendingAction instances share matching_bookings list — use Field(default_factory=list)"
        )

    def test_booking_draft_missing_fields_reports_correctly(self) -> None:
        draft = BookingDraft(attendee_name="Jane")
        missing = draft.missing_fields()
        assert "attendee_email" in missing
        assert "attendee_name" not in missing

    def test_booking_draft_is_ready_when_complete(self) -> None:
        draft = BookingDraft(
            attendee_name="Jane",
            attendee_email="jane@example.com",
            time_preference="Thursday afternoon",
        )
        assert draft.is_ready()

    def test_booking_draft_not_ready_without_time(self) -> None:
        draft = BookingDraft(
            attendee_name="Jane",
            attendee_email="jane@example.com",
        )
        assert not draft.is_ready()


# ===========================================================================
# TestTimePreferenceHandling
# ===========================================================================


class TestTimePreferenceHandling:
    def _state(self) -> dict:
        return {"messages": [], "pending_action": None, "available_slots": []}

    def _cal(self) -> MagicMock:
        m = MagicMock(spec=CalClient)
        m.find_slots.return_value = []
        return m

    def test_unresolved_time_preference_asks_follow_up_and_no_find_slots(self) -> None:
        """name+email+time_preference but no start_time → asks for specific time; no find_slots."""
        intent = UserIntent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="jane@example.com",
            time_preference="Thursday afternoon",
        )
        state = self._state()
        cal = self._cal()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book with Jane Thursday afternoon", state, cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when", "thursday", "2pm"))

    def test_resolved_afternoon_preference_passes_afternoon_range_to_find_slots(self) -> None:
        """start_time in early afternoon with no end_time → derived end ≤ 17:00 same day."""
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
        # 12:30pm ET
        noon_start = datetime(2050, 9, 12, 12, 30, 0, tzinfo=tz)
        intent = UserIntent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=noon_start,
            time_preference="afternoon",
            duration_minutes=30,
        )
        state = self._state()
        cal = self._cal()
        cal.find_slots.return_value = []
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("Book Jane Thursday afternoon", state, cal)
        cal.find_slots.assert_called_once()
        _, kwargs = cal.find_slots.call_args
        end_arg = kwargs.get("end") or (cal.find_slots.call_args[0][1] if cal.find_slots.call_args[0] else None)
        assert end_arg is not None
        end_local = end_arg.astimezone(tz)
        assert end_local.hour <= 17

    def test_resolved_exact_time_passes_narrow_range_to_find_slots(self) -> None:
        """Exact start_time with duration_minutes → end close to start + duration."""
        intent = UserIntent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=_T0,
            duration_minutes=30,
        )
        state = self._state()
        cal = self._cal()
        cal.find_slots.return_value = []
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("Book with Jane at 2pm", state, cal)
        cal.find_slots.assert_called_once()
        _, kwargs = cal.find_slots.call_args
        end_arg = kwargs.get("end") or cal.find_slots.call_args[0][1]
        assert end_arg is not None
        # End should be within reasonable window — not 7 days out
        assert end_arg <= _T0 + timedelta(days=1)

    def test_reschedule_unresolved_later_today_asks_follow_up(self) -> None:
        """Reschedule with time_preference but no start_time → asks for specific time."""
        from unittest.mock import MagicMock
        from schemas import Attendee
        booking = Booking(
            uid="uid-123", title="Intro Call", start=_T0, end=_T1,
            attendees=[Attendee(name="Jane", email="jane@example.com")],
        )
        cal = MagicMock(spec=CalClient)
        cal.list_bookings.return_value = [booking]
        intent = UserIntent(
            intent_type=IntentType.reschedule,
            search_text="Intro Call",
            time_preference="later today",
        )
        state = self._state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move my Intro Call to later today", state, cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when", "reschedule"))
