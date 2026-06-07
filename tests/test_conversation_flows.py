"""End-to-end conversation flow tests — all Cal.com calls and LLM calls are mocked."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from assistant import handle_message
from cal_client import CalClient
from schemas import (
    AssistantError,
    Attendee,
    Booking,
    BookingDraft,
    BookingRequest,
    CalClientError,
    CancelRequest,
    EventType,
    IntentType,
    IntentValidationError,
    PendingAction,
    RescheduleRequest,
    Slot,
    UserIntent,
)

_T0 = datetime(2050, 9, 5, 14, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2050, 9, 5, 14, 30, 0, tzinfo=timezone.utc)
_T2 = datetime(2050, 9, 6, 10, 0, 0, tzinfo=timezone.utc)
_T3 = datetime(2050, 9, 6, 10, 30, 0, tzinfo=timezone.utc)


def _sample_booking(**kwargs) -> Booking:
    defaults = dict(
        uid="uid-123",
        title="Intro Call",
        start=_T0,
        end=_T1,
        attendees=[Attendee(name="Jane", email="jane@example.com")],
        status="accepted",
        event_type_id=42,
    )
    defaults.update(kwargs)
    return Booking(**defaults)


def _sample_slot(start=None, end=None) -> Slot:
    return Slot(start=start or _T0, end=end or _T1)


def _mock_cal(**overrides) -> MagicMock:
    cal = MagicMock(spec=CalClient)
    cal.list_bookings.return_value = []
    cal.find_slots.return_value = []
    cal.list_event_types.return_value = [
        EventType(id=41, title="15 min meeting", slug="15min", lengthInMinutes=15),
        EventType(id=42, title="30 min meeting", slug="30min", lengthInMinutes=30),
    ]
    cal.create_booking.return_value = _sample_booking()
    cal.cancel_booking.return_value = None
    cal.reschedule_booking.return_value = _sample_booking(start=_T2, end=_T3)
    for k, v in overrides.items():
        setattr(cal, k, v)
    return cal


def _state() -> dict:
    return {"messages": [], "pending_action": None, "available_slots": []}


def _intent(**kwargs) -> UserIntent:
    return UserIntent(**kwargs)


# ===========================================================================
# TestListingFlow
# ===========================================================================


class TestListingFlow:
    def test_listing_returns_agenda_when_bookings_exist(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [
            _sample_booking(title="Intro Call"),
            _sample_booking(uid="uid-456", title="Follow-up"),
        ]
        intent = _intent(intent_type=IntentType.list)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("What's on my calendar?", _state(), cal)
        assert "Intro Call" in reply or "Follow-up" in reply

    def test_listing_returns_friendly_empty_state_message(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        intent = _intent(intent_type=IntentType.list)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("What's on my calendar?", _state(), cal)
        assert any(word in reply.lower() for word in ("nothing", "no events", "empty", "scheduled"))


# ===========================================================================
# TestBookingFlow
# ===========================================================================


class TestBookingFlow:
    def test_booking_turn1_missing_email_asks_for_email(self) -> None:
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            duration_minutes=30,
            time_preference="Thursday afternoon",
        )
        state = _state()
        cal = _mock_cal()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book a call with Jane", state, cal)
        assert "email" in reply.lower()
        assert state["pending_action"] is not None
        assert state["pending_action"].booking_draft is not None
        cal.create_booking.assert_not_called()

    def test_booking_turn2_email_provided_fetches_slots(self) -> None:
        intent1 = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            duration_minutes=30,
            time_preference="Thursday",
        )
        # intent2 includes start_time — simulates LLM resolving the date with updated prompt
        intent2 = _intent(
            intent_type=IntentType.book,
            attendee_email="jane@example.com",
            start_time=_T0,
        )
        state = _state()
        cal = _mock_cal()
        cal.find_slots.return_value = [_sample_slot()]
        with patch("assistant.extract_intent", side_effect=[intent1, intent2]):
            handle_message("Book with Jane Thursday", state, cal)   # → asks for email
            reply = handle_message("jane@example.com", state, cal)  # → fetches slots
        cal.find_slots.assert_called_once()
        cal.create_booking.assert_not_called()
        # Reply should present slot options
        assert "2050" in reply or "Sep" in reply or "slot" in reply.lower() or ":" in reply

    def test_booking_slot_chosen_asks_confirmation(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_draft=BookingDraft(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
                time_preference="Thursday",
            ),
        )
        state["available_slots"] = [_sample_slot()]
        cal = _mock_cal()
        reply = handle_message("1", state, cal)
        assert "yes" in reply.lower() or "confirm" in reply.lower() or "no" in reply.lower()
        assert state["pending_action"].booking_request is not None
        cal.create_booking.assert_not_called()

    def test_booking_creates_only_after_explicit_yes(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        reply = handle_message("yes", state, cal)
        cal.create_booking.assert_called_once()
        assert state["pending_action"] is None

    def test_booking_does_not_create_when_user_declines(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        handle_message("no", state, cal)
        cal.create_booking.assert_not_called()
        assert state["pending_action"] is None

    def test_booking_asks_for_new_slot_when_create_fails(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("Slot no longer available", None, reason="slot_unavailable")
        reply = handle_message("yes", state, cal)
        assert "slot" in reply.lower() or "available" in reply.lower() or "choose" in reply.lower()

    def test_booking_follow_up_without_concrete_start_time_does_not_fetch_slots(self) -> None:
        intent1 = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            duration_minutes=30,
            time_preference="Thursday",
        )
        intent2 = _intent(
            intent_type=IntentType.book,
            attendee_email="jane@example.com",
        )
        state = _state()
        cal = _mock_cal()
        cal.find_slots.return_value = [_sample_slot()]
        with patch("assistant.extract_intent", side_effect=[intent1, intent2]):
            handle_message("Book with Jane", state, cal)
            reply = handle_message("jane@example.com", state, cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when"))


# ===========================================================================
# TestCancelFlow
# ===========================================================================


class TestCancelFlow:
    def test_cancel_searches_upcoming_bookings_when_no_uid_given(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        intent = _intent(intent_type=IntentType.cancel, search_text="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("Cancel my call with Jane", _state(), cal)
        cal.list_bookings.assert_called()

    def test_cancel_asks_confirmation_before_canceling(self) -> None:
        state = _state()
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        intent = _intent(intent_type=IntentType.cancel, search_text="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Cancel my call with Jane", state, cal)
        assert "yes" in reply.lower() or "confirm" in reply.lower() or "cancel" in reply.lower()
        cal.cancel_booking.assert_not_called()

    def test_cancel_cancels_after_explicit_yes(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            cancel_request=CancelRequest(booking_uid="uid-123"),
        )
        cal = _mock_cal()
        reply = handle_message("yes", state, cal)
        cal.cancel_booking.assert_called_once_with("uid-123")
        assert state["pending_action"] is None

    def test_cancel_does_not_cancel_when_declined(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            cancel_request=CancelRequest(booking_uid="uid-123"),
        )
        cal = _mock_cal()
        handle_message("no", state, cal)
        cal.cancel_booking.assert_not_called()
        assert state["pending_action"] is None

    def test_cancel_no_matching_booking_returns_not_found(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        intent = _intent(intent_type=IntentType.cancel, search_text="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Cancel my call with Jane", _state(), cal)
        assert "not found" in reply.lower() or "couldn't find" in reply.lower() or "no matching" in reply.lower()
        cal.cancel_booking.assert_not_called()


# ===========================================================================
# TestRescheduleFlow
# ===========================================================================


class TestRescheduleFlow:
    def test_reschedule_searches_upcoming_bookings_when_no_uid_given(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        cal.find_slots.return_value = [_sample_slot()]
        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="Intro Call",
            time_preference="later today",
        )
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("Move my Intro Call to later", _state(), cal)
        cal.list_bookings.assert_called()

    def test_reschedule_without_concrete_start_time_does_not_fetch_slots(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking(uid="uid-123")]
        cal.find_slots.return_value = [_sample_slot()]
        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="Intro Call",
            time_preference="later today",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move my Intro Call to later", _state(), cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when"))

    def test_reschedule_without_concrete_start_time_asks_follow_up(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        cal.find_slots.return_value = [_sample_slot(_T2, _T3)]
        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="Intro",
            time_preference="tomorrow",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move my Intro Call to tomorrow", _state(), cal)
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when"))
        cal.find_slots.assert_not_called()
        cal.reschedule_booking.assert_not_called()

    def test_reschedule_reschedules_after_explicit_yes(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            reschedule_request=RescheduleRequest(
                booking_uid="uid-123", new_start_time=_T2
            ),
        )
        cal = _mock_cal()
        reply = handle_message("yes", state, cal)
        cal.reschedule_booking.assert_called_once_with("uid-123", _T2)
        assert state["pending_action"] is None

    def test_reschedule_no_slots_asks_for_different_time(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        cal.find_slots.return_value = []
        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="Intro",
            start_time=_T0,  # concrete — exercises the actual no-slots path
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move my call to tomorrow", _state(), cal)
        cal.find_slots.assert_called_once()
        assert any(word in reply.lower() for word in ("no", "available", "window", "time"))
        cal.reschedule_booking.assert_not_called()

    def test_reschedule_no_slots_retry_uses_stored_booking_uid(self) -> None:
        """After no-slots, the next time offer reuses the stored uid instead of a fresh search."""
        booking = _sample_booking(uid="uid-123")
        state = _state()
        state["pending_action"] = PendingAction(action_type="reschedule")
        state["_reschedule_booking_uid"] = "uid-123"
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking]
        cal.find_slots.return_value = [_sample_slot(_T2, _T3)]

        intent = _intent(
            intent_type=IntentType.reschedule,
            start_time=_T2,
        )
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("How about Saturday?", state, cal)

        cal.find_slots.assert_called_once()
        _, kwargs = cal.find_slots.call_args
        assert kwargs.get("booking_uid_to_reschedule") == "uid-123"


# ===========================================================================
# TestInvalidActions
# ===========================================================================


class TestInvalidActions:
    def test_booking_in_past_returns_friendly_reply_without_api_call(self) -> None:
        past_time = datetime(2000, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=past_time,
            duration_minutes=30,
        )
        state = _state()
        cal = _mock_cal()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book yesterday", state, cal)
        assert "future" in reply.lower() or "past" in reply.lower() or "only book" in reply.lower()
        cal.create_booking.assert_not_called()

    def test_invalid_email_returns_friendly_reply_without_api_call(self) -> None:
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="not-an-email",
            duration_minutes=30,
            time_preference="Thursday",
        )
        state = _state()
        cal = _mock_cal()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book with Jane, email is not-an-email", state, cal)
        assert "email" in reply.lower()
        cal.create_booking.assert_not_called()

    def test_impossible_date_returns_valid_date_prompt(self) -> None:
        """LLM returned an unparseable date (invalid_date) → asks for valid date."""
        state = _state()
        cal = _mock_cal()
        with patch(
            "assistant.extract_intent",
            side_effect=IntentValidationError("impossible date", reason="invalid_date"),
        ):
            reply = handle_message("Book on February 30", state, cal)
        assert "date" in reply.lower()
        assert "rephrase" not in reply.lower()

    def test_llm_failure_returns_retry_message_not_date_prompt(self) -> None:
        state = _state()
        cal = _mock_cal()
        with patch(
            "assistant.extract_intent",
            side_effect=AssistantError("down", reason="llm_failure"),
        ):
            reply = handle_message("Book something", state, cal)
        assert "try again" in reply.lower() or "trouble" in reply.lower()
        assert "date" not in reply.lower()

    def test_unsupported_action_explains_scope(self) -> None:
        intent = _intent(intent_type=IntentType.unknown)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Send Jane the notes", _state(), _mock_cal())
        assert any(word in reply.lower() for word in ("scheduling", "booking", "cancel", "reschedule"))

    def test_cancel_nonexistent_event_returns_not_found(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        intent = _intent(intent_type=IntentType.cancel, search_text="Nonexistent")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Cancel my nonexistent meeting", _state(), cal)
        assert "not found" in reply.lower() or "couldn't find" in reply.lower() or "no matching" in reply.lower()

    def test_already_cancelled_booking_skips_api(self) -> None:
        cal = _mock_cal()
        # upcoming returns empty; cancelled returns the booking — matches real API behavior
        def _list_side(status="upcoming", **kw):
            if status == "cancelled":
                return [_sample_booking(status="cancelled")]
            return []
        cal.list_bookings.side_effect = _list_side
        intent = _intent(intent_type=IntentType.cancel, search_text="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Cancel my call with Jane", _state(), cal)
        assert "already" in reply.lower() or "cancelled" in reply.lower()
        cal.cancel_booking.assert_not_called()

    def test_stale_slot_triggers_new_slot_selection(self) -> None:
        """Slot becomes unavailable at confirmation → asks user to pick again."""
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("Slot unavailable", None, reason="slot_unavailable")
        reply = handle_message("yes", state, cal)
        assert "slot" in reply.lower() or "available" in reply.lower() or "choose" in reply.lower()
        # Must not claim success
        assert "confirmed" not in reply.lower()
        assert "booked" not in reply.lower()


# ===========================================================================
# TestCornerCases
# ===========================================================================


class TestCornerCases:
    def test_multiple_cancel_matches_asks_to_pick(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [
            _sample_booking(uid="uid-1", title="Call A"),
            _sample_booking(uid="uid-2", title="Call B"),
        ]
        intent = _intent(intent_type=IntentType.cancel, search_text="call")
        with patch("assistant.extract_intent", return_value=intent):
            state = _state()
            reply = handle_message("Cancel my call", state, cal)
        assert "2" in reply or "two" in reply.lower() or "which" in reply.lower() or "pick" in reply.lower()
        assert state["pending_action"].matching_bookings is not None
        assert len(state["pending_action"].matching_bookings) == 2
        cal.cancel_booking.assert_not_called()

    def test_multiple_reschedule_matches_asks_to_pick(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.return_value = [
            _sample_booking(uid="uid-1", title="Call A"),
            _sample_booking(uid="uid-2", title="Call B"),
        ]
        cal.find_slots.return_value = [_sample_slot()]
        intent = _intent(intent_type=IntentType.reschedule, search_text="call")
        with patch("assistant.extract_intent", return_value=intent):
            state = _state()
            reply = handle_message("Reschedule my call", state, cal)
        assert "2" in reply or "which" in reply.lower() or "pick" in reply.lower()
        cal.reschedule_booking.assert_not_called()

    def test_mid_flow_change_resets_pending(self) -> None:
        """User starts booking then asks to list events — pending is cleared and list is served."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking(title="Existing Meeting")]
        state = _state()

        book_intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            time_preference="Thursday",
        )
        list_intent = _intent(intent_type=IntentType.list)

        with patch("assistant.extract_intent", side_effect=[book_intent, list_intent]):
            handle_message("Book a call with Jane", state, cal)  # sets pending
            reply = handle_message("What's on tomorrow?", state, cal)  # should reset pending

        assert "Existing Meeting" in reply or any(w in reply.lower() for w in ("coming up", "scheduled"))
        # No pending booking action
        if state["pending_action"] is not None:
            assert state["pending_action"].action_type != "book"

    def test_yes_with_no_pending_asks_what_they_want(self) -> None:
        state = _state()  # pending_action is None
        cal = _mock_cal()
        reply = handle_message("yes", state, cal)
        assert any(word in reply.lower() for word in ("what", "help", "book", "cancel", "reschedule"))

    def test_cancel_keyword_during_booking_flow_clears_pending_not_calendar(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_draft=BookingDraft(
                attendee_name="Jane",
                duration_minutes=30,
                time_preference="Thursday",
                event_type_id=42,
                timezone="UTC",
            ),
        )
        cal = _mock_cal()
        # LLM interprets "cancel" as a cancel intent with no specific event
        cancel_intent = _intent(intent_type=IntentType.cancel)  # no search_text or uid
        with patch("assistant.extract_intent", return_value=cancel_intent):
            reply = handle_message("cancel", state, cal)
        assert state["pending_action"] is None
        cal.cancel_booking.assert_not_called()

    def test_different_timezone_shown_in_reply(self) -> None:
        """Confirmation message should reflect the attendee's timezone."""
        state = _state()
        from zoneinfo import ZoneInfo
        la_start = datetime(2050, 9, 5, 7, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=la_start,
                duration_minutes=30,
                timezone="America/Los_Angeles",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        # Booking response uses the LA time
        cal.create_booking.return_value = _sample_booking(start=la_start, end=datetime(2050, 9, 5, 7, 30, 0, tzinfo=ZoneInfo("America/Los_Angeles")))
        reply = handle_message("yes", state, cal)
        cal.create_booking.assert_called_once()
        assert isinstance(reply, str)

    def test_rate_limit_shows_retry_message(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.side_effect = CalClientError("Rate limited", 429)
        intent = _intent(intent_type=IntentType.list)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("What's on my calendar?", _state(), cal)
        assert "try again" in reply.lower() or "busy" in reply.lower() or "moment" in reply.lower()

    def test_auth_error_shows_configuration_message(self) -> None:
        cal = _mock_cal()
        cal.list_bookings.side_effect = CalClientError("Unauthorized", 401)
        intent = _intent(intent_type=IntentType.list)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("What's on my calendar?", _state(), cal)
        assert "api key" in reply.lower() or "configuration" in reply.lower() or "authentication" in reply.lower()

    def test_network_timeout_does_not_claim_success(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("Request timed out", None, reason="timeout")
        reply = handle_message("yes", state, cal)
        # Must not claim the booking was made
        assert "confirmed" not in reply.lower()
        assert "booked" not in reply.lower()
        assert "done" not in reply.lower()
        assert "timed out" in reply.lower() or "try again" in reply.lower()

    def test_booking_succeeds_with_empty_attendees_list_still_shows_start_time(self) -> None:

        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        cal = _mock_cal()
        # Response has no attendees (optional field)
        cal.create_booking.return_value = Booking(
            uid="uid-999",
            title="Intro Call",
            start=_T0,
            end=_T1,
            attendees=[],  # empty
        )
        reply = handle_message("yes", state, cal)
        # Must show the confirmed start time without crashing
        assert "2050" in reply or "Sep" in reply or "14:00" in reply or "booked" in reply.lower() or "done" in reply.lower()


# ===========================================================================
# TestRescheduleSlotSelection
# ===========================================================================


class TestRescheduleSlotSelection:
    def test_reschedule_slot_selection_creates_reschedule_request(self) -> None:
        """User picks slot 1 → reschedule_request is set; reschedule_booking NOT called yet."""
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            booking_draft=BookingDraft(
                attendee_name="Jane",
                event_type_id=42,
                timezone="UTC",
            ),
        )
        state["available_slots"] = [_sample_slot(_T2, _T3)]
        state["_reschedule_booking_uid"] = "uid-123"
        cal = _mock_cal()
        reply = handle_message("1", state, cal)
        assert "yes" in reply.lower() or "no" in reply.lower()
        assert state["pending_action"].reschedule_request is not None
        cal.reschedule_booking.assert_not_called()

    def test_reschedule_slot_selection_asks_yes_or_no(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            booking_draft=BookingDraft(event_type_id=42, timezone="UTC"),
        )
        state["available_slots"] = [_sample_slot(_T2, _T3)]
        state["_reschedule_booking_uid"] = "uid-123"
        cal = _mock_cal()
        reply = handle_message("first", state, cal)
        assert "yes" in reply.lower() or "no" in reply.lower()

    def test_reschedule_invalid_slot_re_lists_options(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            booking_draft=BookingDraft(event_type_id=42, timezone="UTC"),
        )
        state["available_slots"] = [_sample_slot(_T2, _T3)]
        state["_reschedule_booking_uid"] = "uid-123"
        cal = _mock_cal()
        reply = handle_message("banana", state, cal)
        assert "slot" in reply.lower() or "1." in reply or "Sep" in reply
        cal.reschedule_booking.assert_not_called()


# ===========================================================================
# TestMultipleMatchSelection
# ===========================================================================


class TestMultipleMatchSelection:
    def test_cancel_multiple_matches_number_selects_correct_booking(self) -> None:
        """User says '1' → first booking's uid used for cancel_request; cancel NOT called."""
        booking1 = _sample_booking(uid="uid-1", title="Call A")
        booking2 = _sample_booking(uid="uid-2", title="Call B")
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            matching_bookings=[booking1, booking2],
        )
        cal = _mock_cal()
        reply = handle_message("1", state, cal)
        assert state["pending_action"].cancel_request is not None
        assert state["pending_action"].cancel_request.booking_uid == "uid-1"
        cal.cancel_booking.assert_not_called()

    def test_reschedule_multiple_matches_number_asks_for_new_time(self) -> None:
        """User says '2' without a new time → asks for the reschedule target time."""
        booking1 = _sample_booking(uid="uid-1", title="Call A")
        booking2 = _sample_booking(uid="uid-2", title="Call B")
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            matching_bookings=[booking1, booking2],
        )
        cal = _mock_cal()
        cal.find_slots.return_value = [_sample_slot()]
        reply = handle_message("2", state, cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when"))

    def test_cancel_invalid_match_selection_shows_list_again(self) -> None:
        """User says 'banana' → re-lists matches; cancel NOT called; matching_bookings unchanged."""
        booking1 = _sample_booking(uid="uid-1", title="Call A")
        booking2 = _sample_booking(uid="uid-2", title="Call B")
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            matching_bookings=[booking1, booking2],
        )
        cal = _mock_cal()
        reply = handle_message("banana", state, cal)
        assert "Call A" in reply or "Call B" in reply or "1" in reply or "2" in reply
        cal.cancel_booking.assert_not_called()
        assert len(state["pending_action"].matching_bookings) == 2

    def test_multiple_match_selection_does_not_call_extract_intent(self) -> None:
        """Selecting from a match list bypasses LLM entirely."""
        booking1 = _sample_booking(uid="uid-1", title="Call A")
        booking2 = _sample_booking(uid="uid-2", title="Call B")
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            matching_bookings=[booking1, booking2],
        )
        cal = _mock_cal()
        with patch("assistant.extract_intent") as mock_ei:
            handle_message("1", state, cal)
        mock_ei.assert_not_called()

    def test_reschedule_multiple_match_preserves_original_time(self) -> None:
        """Picking a booking from a multiple-match list uses the time from the original request."""
        booking1 = _sample_booking(uid="uid-1", title="Call A")
        booking2 = _sample_booking(uid="uid-2", title="Call B")
        state = _state()
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking1, booking2]
        cal.find_slots.return_value = [_sample_slot()]

        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="call",
            start_time=_T0,
        )
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("Move my call to Friday 2pm", state, cal)

        handle_message("2", state, cal)
        cal.find_slots.assert_called_once()
        _, kwargs = cal.find_slots.call_args
        assert kwargs.get("start") == _T0
        assert kwargs.get("booking_uid_to_reschedule") == "uid-2"

    def test_reschedule_multiple_match_full_happy_path(self) -> None:
        """Multi-turn: provide time upfront → multiple matches → pick → see slots → slot selected → confirm prompt."""
        booking1 = _sample_booking(uid="uid-1", title="Call A")
        booking2 = _sample_booking(uid="uid-2", title="Call B")
        slot = _sample_slot(_T2, _T3)
        state = _state()
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking1, booking2]
        cal.find_slots.return_value = [slot]

        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="call",
            start_time=_T0,
        )
        # Turn 1: multiple matches — asks which booking
        with patch("assistant.extract_intent", return_value=intent):
            reply1 = handle_message("Move my call to Friday 2pm", state, cal)
        assert "Call A" in reply1 or "Call B" in reply1

        # Turn 2: pick booking 2 (bypasses LLM) — shows slot list
        reply2 = handle_message("2", state, cal)
        cal.find_slots.assert_called_once()
        _, kwargs = cal.find_slots.call_args
        assert kwargs.get("booking_uid_to_reschedule") == "uid-2"
        assert "slot" in reply2.lower() or "Sep" in reply2

        # Turn 3: pick slot 1 (bypasses LLM) — asks yes/no
        reply3 = handle_message("1", state, cal)
        assert "yes" in reply3.lower() or "no" in reply3.lower()
        rr = state["pending_action"].reschedule_request
        assert rr is not None
        assert rr.booking_uid == "uid-2"
        assert rr.new_start_time == slot.start
        cal.reschedule_booking.assert_not_called()


# ===========================================================================
# TestCancelledBookingDetection
# ===========================================================================


class TestCancelledBookingDetection:
    def test_cancel_finds_cancelled_only_in_cancelled_status(self) -> None:
        """Upcoming search returns empty; cancelled search returns the booking."""
        cal = _mock_cal()

        def _list_side(status="upcoming", **kw):
            if status == "cancelled":
                return [_sample_booking(status="cancelled")]
            return []

        cal.list_bookings.side_effect = _list_side
        intent = _intent(intent_type=IntentType.cancel, search_text="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Cancel my call with Jane", _state(), cal)
        assert "already" in reply.lower() or "cancelled" in reply.lower()
        cal.cancel_booking.assert_not_called()

    def test_reschedule_finds_cancelled_only_in_cancelled_status(self) -> None:
        """Upcoming search returns empty; cancelled search returns the booking."""
        cal = _mock_cal()

        def _list_side(status="upcoming", **kw):
            if status == "cancelled":
                return [_sample_booking(status="cancelled")]
            return []

        cal.list_bookings.side_effect = _list_side
        intent = _intent(intent_type=IntentType.reschedule, search_text="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Reschedule my call with Jane", _state(), cal)
        assert "cancelled" in reply.lower()
        cal.reschedule_booking.assert_not_called()


# ===========================================================================
# TestErrorClassification
# ===========================================================================


class TestErrorClassification:
    def _booking_pending_state(self) -> dict:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_request=BookingRequest(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                start_time=_T0,
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        return state

    def _cancel_pending_state(self) -> dict:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            cancel_request=CancelRequest(booking_uid="uid-123"),
        )
        return state

    def _reschedule_pending_state(self) -> dict:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            reschedule_request=RescheduleRequest(booking_uid="uid-123", new_start_time=_T2),
        )
        return state

    def test_booking_confirmation_timeout_shows_retry_message(self) -> None:
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("timed out", None, reason="timeout")
        reply = handle_message("yes", self._booking_pending_state(), cal)
        assert "timed out" in reply.lower() or "could not be reached" in reply.lower()
        assert "confirmed" not in reply.lower()

    def test_booking_confirmation_network_error_shows_retry_message(self) -> None:
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("network error", None, reason="network")
        reply = handle_message("yes", self._booking_pending_state(), cal)
        assert "timed out" in reply.lower() or "could not be reached" in reply.lower()
        assert "confirmed" not in reply.lower()

    def test_booking_confirmation_malformed_shows_unexpected_response(self) -> None:
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("bad json", None, reason="malformed")
        reply = handle_message("yes", self._booking_pending_state(), cal)
        assert "unexpected" in reply.lower() or "response" in reply.lower()
        assert "confirmed" not in reply.lower()

    def test_booking_confirmation_http_400_shows_rejected_request(self) -> None:
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError(
            "Email verification code is required", 400
        )
        reply = handle_message("yes", self._booking_pending_state(), cal)

        assert "rejected" in reply.lower()
        assert "Email verification code is required" in reply
        assert "confirmed" not in reply.lower()

    def test_booking_slot_unavailable_asks_to_choose_again(self) -> None:
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError("slot gone", None, reason="slot_unavailable")
        reply = handle_message("yes", self._booking_pending_state(), cal)
        assert "slot" in reply.lower() or "available" in reply.lower() or "choose" in reply.lower()
        assert "confirmed" not in reply.lower()

    def test_cancel_timeout_shows_retry_message(self) -> None:
        cal = _mock_cal()
        cal.cancel_booking.side_effect = CalClientError("timed out", None, reason="timeout")
        reply = handle_message("yes", self._cancel_pending_state(), cal)
        assert "timed out" in reply.lower() or "could not be reached" in reply.lower()

    def test_reschedule_timeout_shows_retry_message(self) -> None:
        cal = _mock_cal()
        cal.reschedule_booking.side_effect = CalClientError("timed out", None, reason="timeout")
        reply = handle_message("yes", self._reschedule_pending_state(), cal)
        assert "timed out" in reply.lower() or "could not be reached" in reply.lower()


# ===========================================================================
# TestTimePreferenceResolution
# ===========================================================================


class TestTimePreferenceResolution:
    def test_unresolved_time_preference_asks_for_specific_time(self) -> None:
        """time_preference set but start_time still None → ask for specific date/time."""
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="jane@example.com",
            time_preference="Thursday afternoon",
        )
        state = _state()
        cal = _mock_cal()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book with Jane Thursday afternoon", state, cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when", "2pm", "thursday"))

    def test_reschedule_unresolved_later_today_asks_for_specific_time(self) -> None:
        """Reschedule with only time_preference and no start_time → ask for specific time."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="Intro Call",
            time_preference="later today",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move my Intro Call to later today", _state(), cal)
        cal.find_slots.assert_not_called()
        assert any(word in reply.lower() for word in ("specific", "date", "time", "day", "when", "reschedule"))
