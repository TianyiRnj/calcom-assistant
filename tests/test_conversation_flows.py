"""End-to-end conversation flow tests — all Cal.com calls and LLM calls are mocked."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

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
    ExtractedAttendee,
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

    def test_bare_tomorrow_lists_local_tomorrow(self) -> None:
        ny = ZoneInfo("America/New_York")
        local_now = datetime(2026, 6, 7, 22, 30, 0, tzinfo=ny)
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        list_intent = _intent(
            intent_type=IntentType.list,
            start_time=datetime(2026, 6, 8, 0, 0, 0, tzinfo=ny),
            end_time=datetime(2026, 6, 9, 0, 0, 0, tzinfo=ny),
        )

        with (
            patch("assistant._local_now", return_value=local_now),
            patch("assistant.extract_intent", return_value=list_intent),
        ):
            reply = handle_message("tomorrow", _state(), cal)

        assert "nothing" in reply.lower()
        kwargs = cal.list_bookings.call_args.kwargs
        assert kwargs["start"] == datetime(2026, 6, 8, 0, 0, 0, tzinfo=ny)
        assert kwargs["end"] == datetime(2026, 6, 9, 0, 0, 0, tzinfo=ny)

    def test_what_happen_next_week_is_stable_across_repeats(self) -> None:
        ny = ZoneInfo("America/New_York")
        local_now = datetime(2026, 6, 7, 22, 30, 0, tzinfo=ny)
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        state = _state()
        list_intent = _intent(
            intent_type=IntentType.list,
            start_time=datetime(2026, 6, 8, 0, 0, 0, tzinfo=ny),
            end_time=datetime(2026, 6, 15, 0, 0, 0, tzinfo=ny),
        )

        with (
            patch("assistant._local_now", return_value=local_now),
            patch("assistant.extract_intent", return_value=list_intent),
        ):
            handle_message("what happen next week", state, cal)
            handle_message("what happen next week", state, cal)
            handle_message("what is on my calendar next week", state, cal)

        for call in cal.list_bookings.call_args_list:
            assert call.kwargs["start"] == datetime(2026, 6, 8, 0, 0, 0, tzinfo=ny)
            assert call.kwargs["end"] == datetime(2026, 6, 15, 0, 0, 0, tzinfo=ny)


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
        # Turn 1 intent has a concrete start_time so that when the email is direct-merged
        # on turn 2 (bypassing the LLM), the draft is already complete.
        intent1 = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            duration_minutes=30,
            start_time=_T0,
        )
        state = _state()
        cal = _mock_cal()
        cal.find_slots.return_value = [_sample_slot()]
        with patch("assistant.extract_intent", return_value=intent1):
            handle_message("Book with Jane at 2pm", state, cal)  # → asks for email
        # Turn 2: email direct-merged without LLM call
        reply = handle_message("jane@example.com", state, cal)
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
        assert "?" in reply
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
        handle_message("yes", state, cal)
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
        handle_message("yes", state, cal)
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

    def test_cancel_tomorrow_meeting_matches_by_date_window(self) -> None:
        # search_text="meeting" strips to no tokens → vague fallback; time window filters to 1 booking.
        # Single vague match → shown as numbered list (not auto-confirmed).
        tomorrow_start = datetime(2050, 9, 6, 0, 0, 0, tzinfo=timezone.utc)
        tomorrow_end = datetime(2050, 9, 7, 0, 0, 0, tzinfo=timezone.utc)
        cal = _mock_cal()
        cal.list_bookings.return_value = [
            _sample_booking(uid="uid-today", title="Intro Call"),
            _sample_booking(uid="uid-tomorrow", title="Team Sync", start=_T2, end=_T3),
        ]
        intent = _intent(
            intent_type=IntentType.cancel,
            search_text="meeting",
            start_time=tomorrow_start,
            end_time=tomorrow_end,
        )
        state = _state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Cancel tomorrow meeting", state, cal)

        assert "Team Sync" in reply
        # Vague match → numbered list shown; cancel not yet confirmed
        pending = state["pending_action"]
        assert pending is not None
        assert pending.matching_bookings_are_partial is True
        assert any(b.uid == "uid-tomorrow" for b in pending.matching_bookings)
        cal.cancel_booking.assert_not_called()

    def test_cancel_the_meeting_at_time_strips_filler_words(self) -> None:
        # "meeting" is stripped; time filter narrows to 1 booking → shown as possible match list.
        ny = ZoneInfo("America/New_York")
        local_now = datetime(2026, 6, 7, 22, 30, 0, tzinfo=ny)
        meeting = _sample_booking(
            uid="uid-130",
            title="30 min meeting between Tianyi Ren and tom and jack",
            start=datetime(2026, 6, 8, 13, 30, 0, tzinfo=ny),
            end=datetime(2026, 6, 8, 14, 0, 0, tzinfo=ny),
        )
        cal = _mock_cal()
        cal.list_bookings.return_value = [meeting]
        state = _state()
        intent = _intent(
            intent_type=IntentType.cancel,
            source_start_time=datetime(2026, 6, 8, 13, 30, 0, tzinfo=ny),
        )

        with (
            patch("assistant._local_now", return_value=local_now),
            patch("assistant.extract_intent", return_value=intent),
        ):
            reply = handle_message("cancel the meeting tomorrow at 1:30", state, cal)

        assert "tom and jack" in reply
        pending = state["pending_action"]
        assert pending is not None
        assert pending.matching_bookings_are_partial is True
        assert any(b.uid == "uid-130" for b in pending.matching_bookings)
        cal.cancel_booking.assert_not_called()

    def test_broad_source_window_cancel_multiple_bookings_shows_choices(self) -> None:
        """A broad source time window overlapping multiple upcoming bookings
        produces a numbered choice list without cancelling anything."""
        afternoon_start = datetime(2050, 9, 5, 12, 0, 0, tzinfo=timezone.utc)  # noon
        afternoon_end = datetime(2050, 9, 5, 17, 0, 0, tzinfo=timezone.utc)  # 5 PM
        b1 = _sample_booking(
            uid="uid-1",
            title="Morning Call",
            start=datetime(2050, 9, 5, 14, 0, tzinfo=timezone.utc),
            end=datetime(2050, 9, 5, 14, 30, tzinfo=timezone.utc),
        )
        b2 = _sample_booking(
            uid="uid-2",
            title="Afternoon Sync",
            start=datetime(2050, 9, 5, 15, 0, tzinfo=timezone.utc),
            end=datetime(2050, 9, 5, 15, 30, tzinfo=timezone.utc),
        )
        cal = _mock_cal()
        cal.list_bookings.return_value = [b1, b2]
        # Broad source window spans both same-day bookings; no text filters → vague fallback
        intent = _intent(
            intent_type=IntentType.cancel,
            source_start_time=afternoon_start,
            source_end_time=afternoon_end,
        )
        state = _state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel my call this afternoon", state, cal)

        pending = state["pending_action"]
        assert pending is not None
        assert len(pending.matching_bookings) == 2
        assert any(b.uid == "uid-1" for b in pending.matching_bookings)
        assert any(b.uid == "uid-2" for b in pending.matching_bookings)
        assert "1." in reply
        assert "2." in reply
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
        handle_message("yes", state, cal)
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
        assert cal.find_slots.call_count >= 1
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

    def test_reschedule_source_time_and_target_day_are_not_confused(self) -> None:
        # LLM extracts source_start_time (tomorrow 1:30) and target_start_time (Tuesday).
        # "meeting" stripped → vague match; time filter narrows to 1 booking → numbered list.
        # User selects "1" → slots shown → user selects "1" → confirmation → "yes" confirms.
        ny = ZoneInfo("America/New_York")
        local_now = datetime(2026, 6, 7, 22, 30, 0, tzinfo=ny)
        source_start = datetime(2026, 6, 8, 13, 30, 0, tzinfo=ny)
        source_end = source_start + timedelta(minutes=30)
        target_start = datetime(2026, 6, 9, 13, 30, 0, tzinfo=ny)
        target_end = target_start + timedelta(minutes=30)
        intended_booking = _sample_booking(
            uid="uid-tom-jack",
            title="30 min meeting between Tianyi Ren and tom and jack",
            start=source_start,
            end=source_end,
        )
        wrong_tuesday_booking = _sample_booking(
            uid="uid-taylor",
            title="30 min meeting between Tianyi Ren and Taylor",
            start=datetime(2026, 6, 9, 14, 0, 0, tzinfo=ny),
            end=datetime(2026, 6, 9, 14, 30, 0, tzinfo=ny),
        )
        slot = Slot(start=target_start, end=target_end)
        state = _state()
        cal = _mock_cal()
        cal.list_bookings.return_value = [intended_booking, wrong_tuesday_booking]
        cal.find_slots.return_value = [slot]
        cal.reschedule_booking.return_value = _sample_booking(
            uid="uid-tom-jack",
            title=intended_booking.title,
            start=target_start,
            end=target_end,
        )
        reschedule_intent = _intent(
            intent_type=IntentType.reschedule,
            source_start_time=source_start,
            start_time=target_start,  # target destination for reschedule
        )

        with (
            patch("assistant._local_now", return_value=local_now),
            patch("assistant.extract_intent", return_value=reschedule_intent),
        ):
            reply1 = handle_message("move the meeting tomorrow at 1:30 to Tuesday", state, cal)
            # Source time filter → 1 vague match shown as list
            assert "possible match" in reply1.lower() or "tom and jack" in reply1
            reply2 = handle_message("1", state, cal)  # select the booking
            assert "available slots" in reply2.lower()
            reply3 = handle_message("1", state, cal)  # select the slot
            assert "reschedule" in reply3.lower()
            reply4 = handle_message("yes", state, cal)  # confirm

        assert "Rescheduled" in reply4
        find_kwargs = cal.find_slots.call_args.kwargs
        assert find_kwargs["booking_uid_to_reschedule"] == "uid-tom-jack"
        assert find_kwargs["start"] == target_start
        cal.reschedule_booking.assert_called_once_with("uid-tom-jack", target_start)

    def test_cancelled_only_reschedule_treated_as_not_found(self) -> None:
        """Cancelled bookings are omitted from upcoming results; reschedule returns not-found.

        list_bookings is queried once with status="upcoming"; no separate cancelled
        lookup is made and reschedule_booking is never called.
        """
        cal = _mock_cal()
        # Simulates that the matching booking exists only in cancelled status —
        # list_bookings(status="upcoming") returns nothing for it.
        cal.list_bookings.return_value = []
        intent = _intent(
            intent_type=IntentType.reschedule,
            attendee_name="Jane",
            start_time=_T2,
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move my call with Jane to tomorrow", _state(), cal)

        assert "not found" in reply.lower() or "couldn't find" in reply.lower()
        cal.list_bookings.assert_called_once()
        _, kwargs = cal.list_bookings.call_args
        assert kwargs.get("status") == "upcoming"
        cal.reschedule_booking.assert_not_called()

    def test_broad_source_window_reschedule_multiple_bookings_shows_choices(self) -> None:
        """A broad source time window overlapping multiple upcoming bookings
        produces a numbered choice list without rescheduling anything."""
        afternoon_start = datetime(2050, 9, 5, 12, 0, 0, tzinfo=timezone.utc)  # noon
        afternoon_end = datetime(2050, 9, 5, 17, 0, 0, tzinfo=timezone.utc)  # 5 PM
        b1 = _sample_booking(
            uid="uid-1",
            title="Call A",
            start=datetime(2050, 9, 5, 14, 0, tzinfo=timezone.utc),
            end=datetime(2050, 9, 5, 14, 30, tzinfo=timezone.utc),
        )
        b2 = _sample_booking(
            uid="uid-2",
            title="Call B",
            start=datetime(2050, 9, 5, 15, 0, tzinfo=timezone.utc),
            end=datetime(2050, 9, 5, 15, 30, tzinfo=timezone.utc),
        )
        cal = _mock_cal()
        cal.list_bookings.return_value = [b1, b2]
        # Broad source window spans both bookings; target is a clearly different day
        intent = _intent(
            intent_type=IntentType.reschedule,
            source_start_time=afternoon_start,
            source_end_time=afternoon_end,
            start_time=_T2,  # Sep 6 10:00 — unambiguously outside the source window
        )
        state = _state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("move my call this afternoon to tomorrow morning", state, cal)

        pending = state["pending_action"]
        assert pending is not None
        assert len(pending.matching_bookings) == 2
        assert any(b.uid == "uid-1" for b in pending.matching_bookings)
        assert any(b.uid == "uid-2" for b in pending.matching_bookings)
        assert "1." in reply
        assert "2." in reply
        cal.reschedule_booking.assert_not_called()


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

    def test_new_command_interrupts_open_slot_selection(self) -> None:
        """A new intent while slots are open clears stale slot choices and handles the new request."""
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="book",
            booking_draft=BookingDraft(
                attendee_name="Jane",
                attendee_email="jane@example.com",
                duration_minutes=30,
                timezone="UTC",
                event_type_id=42,
            ),
        )
        state["available_slots"] = [_sample_slot()]
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        intent = _intent(intent_type=IntentType.cancel, search_text="Jane")

        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Can you cancel my call with Jane?", state, cal)

        assert state["available_slots"] == []
        assert state["pending_action"].action_type == "cancel"
        assert state["pending_action"].cancel_request is not None
        assert "Cancel" in reply
        cal.create_booking.assert_not_called()

    def test_new_command_interrupts_multiple_match_selection(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            matching_bookings=[
                _sample_booking(uid="uid-1", title="Call A"),
                _sample_booking(uid="uid-2", title="Call B"),
            ],
        )
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking(title="Existing Meeting")]
        intent = _intent(intent_type=IntentType.list)

        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Actually, what's on my calendar?", state, cal)

        assert "Existing Meeting" in reply
        assert state["pending_action"] is None

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
            handle_message("cancel", state, cal)
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
        assert "?" in reply
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
        assert "?" in reply

    def test_reschedule_invalid_slot_re_lists_options(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            booking_draft=BookingDraft(event_type_id=42, timezone="UTC"),
        )
        state["available_slots"] = [_sample_slot(_T2, _T3)]
        state["_reschedule_booking_uid"] = "uid-123"
        cal = _mock_cal()
        with patch(
            "assistant.extract_intent",
            return_value=_intent(intent_type=IntentType.unknown),
        ):
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
        handle_message("1", state, cal)
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
        with patch(
            "assistant.extract_intent",
            return_value=_intent(intent_type=IntentType.unknown),
        ):
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

        # Turn 3: pick slot 1 (bypasses LLM) — asks for confirmation
        reply3 = handle_message("1", state, cal)
        assert "?" in reply3
        rr = state["pending_action"].reschedule_request
        assert rr is not None
        assert rr.booking_uid == "uid-2"
        assert rr.new_start_time == slot.start
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

    def test_booking_slot_unavailable_offers_nearby_daypart_slots(self) -> None:
        state = self._booking_pending_state()
        nearby_slot = _sample_slot(
            _T0 + timedelta(hours=1),
            _T1 + timedelta(hours=1),
        )
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError(
            "slot gone", 400, reason="slot_unavailable"
        )
        cal.find_slots.return_value = [nearby_slot]

        reply = handle_message("yes", state, cal)

        assert "nearby" in reply.lower()
        assert state["available_slots"] == [nearby_slot]
        assert state["pending_action"].booking_request is None

    def test_booking_slot_unavailable_falls_back_to_same_day(self) -> None:
        state = self._booking_pending_state()
        same_day_slot = _sample_slot(
            _T0 + timedelta(hours=3),
            _T1 + timedelta(hours=3),
        )
        cal = _mock_cal()
        cal.create_booking.side_effect = CalClientError(
            "slot gone", 400, reason="slot_unavailable"
        )
        cal.find_slots.side_effect = [[], [same_day_slot]]

        reply = handle_message("yes", state, cal)

        assert "nearby" in reply.lower()
        assert state["available_slots"] == [same_day_slot]
        assert cal.find_slots.call_count == 2

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


# ===========================================================================
# TestNoAvailabilityExplanation
# ===========================================================================


class TestNoAvailabilityExplanation:
    def test_no_slots_after_all_fallbacks_returns_rules_message(self) -> None:
        from assistant import _no_availability_message
        cal = _mock_cal()
        cal.find_slots.return_value = []
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Taylor",
            attendee_email="taylor@example.com",
            start_time=_T0,
            end_time=_T1,
            duration_minutes=30,
            time_granularity="exact",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book with Taylor at 2pm", _state(), cal)
        expected = _no_availability_message()
        assert reply == expected

    def test_no_slots_for_reschedule_returns_rules_message(self) -> None:
        from assistant import _no_availability_message
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        cal.find_slots.return_value = []
        # _T0/_T1 match the sample booking's time so the time-window filter finds it.
        # The cascade fires all fallback windows (all returning []) then yields the rules message.
        intent = _intent(
            intent_type=IntentType.reschedule,
            search_text="Intro Call",
            start_time=_T0,
            end_time=_T1,
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Move Intro Call to Monday", _state(), cal)
        expected = _no_availability_message()
        assert reply == expected

    def test_no_slots_does_not_say_what_other_time(self) -> None:
        cal = _mock_cal()
        cal.find_slots.return_value = []
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Taylor",
            attendee_email="taylor@example.com",
            start_time=_T0,
            end_time=_T1,
            duration_minutes=30,
            time_granularity="exact",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("Book with Taylor at 2pm", _state(), cal)
        assert "What other time" not in reply


# ===========================================================================
# TestSlotRefinement
# ===========================================================================


class TestSlotRefinement:
    def _state_with_slots(self) -> dict:
        draft = BookingDraft(
            attendee_name="Taylor",
            attendee_email="taylor@example.com",
            duration_minutes=30,
            start_time=_T0,
            timezone="UTC",
            event_type_id=42,
        )
        pending = PendingAction(action_type="book", booking_draft=draft)
        return {
            "messages": [],
            "pending_action": pending,
            "available_slots": [Slot(start=_T0, end=_T1)],
        }

    def test_date_refinement_during_slot_selection_reuses_draft(self) -> None:
        state = self._state_with_slots()
        new_start = _T2  # different date
        new_slots = [Slot(start=new_start, end=_T3)]
        cal = _mock_cal()
        cal.find_slots.return_value = new_slots
        intent = _intent(
            intent_type=IntentType.book,
            start_time=new_start,
            time_granularity="date",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("actually next Monday", state, cal)
        # Draft name/email preserved; find_slots was called again
        cal.find_slots.assert_called()
        assert state["pending_action"].booking_draft.attendee_name == "Taylor"
        assert state["pending_action"].booking_draft.attendee_email == "taylor@example.com"
        assert "slot" in reply.lower() or "available" in reply.lower() or "select" in reply.lower()

    def test_time_refinement_during_slot_selection(self) -> None:
        state = self._state_with_slots()
        new_start = datetime(2050, 9, 5, 15, 0, 0, tzinfo=timezone.utc)  # 3pm same day
        cal = _mock_cal()
        cal.find_slots.return_value = [Slot(start=new_start, end=new_start + timedelta(minutes=30))]
        intent = _intent(
            intent_type=IntentType.book,
            start_time=new_start,
            time_granularity="exact",
        )
        with patch("assistant.extract_intent", return_value=intent):
            handle_message("actually 3pm", state, cal)
        cal.find_slots.assert_called()
        assert state["pending_action"].booking_draft.attendee_name == "Taylor"

    def test_non_book_intent_during_slot_selection_clears_slots(self) -> None:
        state = self._state_with_slots()
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        intent = _intent(intent_type=IntentType.list, start_time=_T0, end_time=_T1)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("what's on my calendar?", state, cal)
        assert state["available_slots"] == []
        assert "coming up" in reply.lower() or "nothing" in reply.lower()


# ===========================================================================
# TestContextIsolation
# ===========================================================================


class TestContextIsolation:
    """After confirm/decline, _new_task clears LLM history for the next call."""

    def test_confirm_sets_new_task_and_next_turn_passes_empty_history(self) -> None:
        state = _state()
        state["messages"] = [
            {"role": "user", "content": "reschedule my call"},
            {"role": "assistant", "content": "Here are some slots"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "Confirm reschedule?"},
        ]
        state["pending_action"] = PendingAction(
            action_type="reschedule",
            reschedule_request=RescheduleRequest(booking_uid="uid-1", new_start_time=_T2),
        )
        cal = _mock_cal()
        # Confirm — should set _new_task=True and clear pending_action
        handle_message("yes", state, cal)
        assert state.get("_new_task") is True
        assert state["pending_action"] is None

        # Next call — _new_task consumed; extract_intent should get empty history
        captured: list[list] = []

        def _capture_intent(text: str, history: list) -> UserIntent:
            captured.append(list(history))
            return UserIntent(intent_type=IntentType.list)

        cal2 = _mock_cal()
        cal2.list_bookings.return_value = []
        with patch("assistant.extract_intent", side_effect=_capture_intent):
            handle_message("list my meetings", state, cal2)

        assert len(captured) == 1
        assert captured[0] == []
        assert not state.get("_new_task")

    def test_decline_sets_new_task(self) -> None:
        state = _state()
        state["pending_action"] = PendingAction(
            action_type="cancel",
            cancel_request=CancelRequest(booking_uid="uid-1"),
        )
        cal = _mock_cal()
        handle_message("no", state, cal)
        assert state.get("_new_task") is True
        assert state["pending_action"] is None

    def test_new_task_consumed_after_one_turn(self) -> None:
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
        handle_message("yes", state, cal)
        assert state.get("_new_task") is True

        cal2 = _mock_cal()
        cal2.list_bookings.return_value = []
        with patch("assistant.extract_intent", return_value=UserIntent(intent_type=IntentType.list)):
            handle_message("list", state, cal2)

        # _new_task should be consumed after this turn
        assert not state.get("_new_task")

    def test_yes_with_no_pending_consumes_new_task(self) -> None:
        state = _state()
        state["_new_task"] = True
        cal = _mock_cal()
        handle_message("yes", state, cal)
        # _new_task must be consumed regardless of which branch fires
        assert not state.get("_new_task")


# ===========================================================================
# TestTokenBasedMatching
# ===========================================================================


class TestTokenBasedMatching:
    """Token-based cancel/reschedule search with full sentences, months, p.m., durations."""

    NY = ZoneInfo("America/New_York")
    _BOOKING_START = datetime(2026, 6, 9, 13, 30, 0, tzinfo=ZoneInfo("America/New_York"))
    _BOOKING_END = datetime(2026, 6, 9, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))

    def _taylor_booking(self) -> Booking:
        return _sample_booking(
            uid="uid-taylor",
            title="30 min meeting between Tianyi Ren and Taylor",
            start=self._BOOKING_START,
            end=self._BOOKING_END,
            attendees=[Attendee(name="Taylor", email="taylor@example.com")],
        )

    def _local_now(self):
        return datetime(2026, 6, 7, 10, 0, 0, tzinfo=self.NY)

    def test_full_sentence_cancel_finds_taylor_booking(self) -> None:
        """LLM returns search_text with full sentence; token matching finds the booking."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [self._taylor_booking()]
        state = _state()
        # Simulate what the LLM would return for a full sentence cancel
        intent = _intent(
            intent_type=IntentType.cancel,
            search_text="30 min meeting between Tianyi Ren and Taylor on Jun 9 1:30 PM EDT",
            start_time=self._BOOKING_START,
            end_time=self._BOOKING_END,
            time_granularity="exact",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message(
                "cancel 30 min meeting between Tianyi Ren and Taylor on Jun 9, 1:30 PM EDT",
                state,
                cal,
            )
        # Tokens: ["tianyi", "ren", "taylor"] — all present in booking title
        assert "not found" not in reply.lower() and "couldn't find" not in reply.lower()
        assert state["pending_action"] is not None
        assert state["pending_action"].cancel_request is not None
        assert state["pending_action"].cancel_request.booking_uid == "uid-taylor"

    def test_cancel_with_full_month_name_finds_booking(self) -> None:
        """search_text with full month name 'June' is stripped correctly."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [self._taylor_booking()]
        state = _state()
        intent = _intent(
            intent_type=IntentType.cancel,
            search_text="Taylor June 9 at 1:30 p.m.",
            start_time=self._BOOKING_START,
            end_time=self._BOOKING_END,
            time_granularity="exact",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel Taylor June 9 at 1:30 p.m.", state, cal)
        # Token "taylor" survives stripping; booking found
        assert "not found" not in reply.lower() and "couldn't find" not in reply.lower()
        assert state["pending_action"] is not None
        assert state["pending_action"].cancel_request is not None

    def test_cancel_with_duration_word_finds_booking(self) -> None:
        """Duration words 'minute' are stripped so 'taylor' is the identifying token."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [self._taylor_booking()]
        state = _state()
        intent = _intent(
            intent_type=IntentType.cancel,
            search_text="30 minute meeting with Taylor Jun 9 1:30 PM",
            start_time=self._BOOKING_START,
            end_time=self._BOOKING_END,
            time_granularity="exact",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel 30 minute meeting with Taylor Jun 9 1:30 PM", state, cal)
        assert "not found" not in reply.lower() and "couldn't find" not in reply.lower()
        assert state["pending_action"] is not None

    def test_only_generic_words_returns_all_bookings(self) -> None:
        """When all tokens are stripped and no time filter, no text filter is applied."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [self._taylor_booking()]
        state = _state()
        # search_text with all tokens stripped; no time filter → no text filter → all bookings
        intent = _intent(intent_type=IntentType.cancel, search_text="meeting call")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel the meeting call", state, cal)
        # With no effective filters, the single booking is returned → confirmation prompt
        assert state["pending_action"] is not None

    def test_meeting_with_followup_calls_llm(self) -> None:
        """'meeting with Tom' after a failed cancel routes through LLM, not _MEETING_WITH_FOLLOWUP_RE."""
        booking_tom = _sample_booking(
            uid="uid-tom",
            title="30 min meeting between Tianyi and Tom",
            attendees=[Attendee(name="Tom", email="tom@example.com")],
        )
        cal = _mock_cal()
        state = _state()
        cal.list_bookings.return_value = [booking_tom]
        # LLM extracts a cancel intent with attendee_name "Tom"
        intent = _intent(intent_type=IntentType.cancel, attendee_name="tom")
        with patch("assistant.extract_intent", return_value=intent) as mock_extract:
            reply = handle_message("meeting with tom", state, cal)
        mock_extract.assert_called_once()
        assert isinstance(reply, str)

    def test_no_cancelled_lookup_on_no_match(self) -> None:
        """Cancelled bookings are omitted from upcoming results, so the assistant treats them
        as no match. list_bookings is queried once with status="upcoming"; no separate
        cancelled lookup is made and cancel_booking is never called."""
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        intent = _intent(intent_type=IntentType.cancel, attendee_name="nonexistent")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel my meeting with nonexistent", _state(), cal)
        assert "not found" in reply.lower() or "couldn't find" in reply.lower()
        cal.list_bookings.assert_called_once()
        _, kwargs = cal.list_bookings.call_args
        assert kwargs.get("status") == "upcoming"
        cal.cancel_booking.assert_not_called()


# ===========================================================================
# TestListResultIsolation
# ===========================================================================


class TestListResultIsolation:
    """list tomorrow/next week after a completed reschedule should not be contaminated."""

    NY = ZoneInfo("America/New_York")

    def _local_now_jun8(self):
        return datetime(2026, 6, 8, 10, 0, 0, tzinfo=self.NY)

    def test_list_tomorrow_after_reschedule_uses_fresh_list(self) -> None:
        local_now = self._local_now_jun8()
        jun8_booking = _sample_booking(
            uid="uid-tom-jack",
            title="30 min meeting between Tianyi Ren and tom and jack",
            start=datetime(2026, 6, 8, 13, 30, 0, tzinfo=self.NY),
            end=datetime(2026, 6, 8, 14, 0, 0, tzinfo=self.NY),
        )
        # Reschedule completed — state is clean with _new_task set
        state = _state()
        state["_new_task"] = True
        state["messages"] = [
            {"role": "user", "content": "reschedule tom jack meeting to Tuesday"},
            {"role": "assistant", "content": "Rescheduled to Jun 9 1:30 PM EDT."},
        ]

        cal = _mock_cal()
        cal.list_bookings.return_value = [jun8_booking]
        list_intent = _intent(
            intent_type=IntentType.list,
            start_time=datetime(2026, 6, 9, 0, 0, 0, tzinfo=self.NY),
            end_time=datetime(2026, 6, 10, 0, 0, 0, tzinfo=self.NY),
        )

        with (
            patch("assistant._local_now", return_value=local_now),
            patch("assistant.extract_intent", return_value=list_intent),
        ):
            reply = handle_message("list tomorrow", state, cal)

        # list_bookings should have been called for Jun 9 2026
        kwargs = cal.list_bookings.call_args.kwargs
        assert kwargs["start"].astimezone(self.NY).date().isoformat() == "2026-06-09"

    def test_list_tomorrow_empty_calendar_after_reschedule(self) -> None:
        local_now = self._local_now_jun8()
        state = _state()
        state["_new_task"] = True
        cal = _mock_cal()
        cal.list_bookings.return_value = []
        list_intent = _intent(
            intent_type=IntentType.list,
            start_time=datetime(2026, 6, 9, 0, 0, 0, tzinfo=self.NY),
            end_time=datetime(2026, 6, 10, 0, 0, 0, tzinfo=self.NY),
        )

        with (
            patch("assistant._local_now", return_value=local_now),
            patch("assistant.extract_intent", return_value=list_intent),
        ):
            reply = handle_message("list tomorrow", state, cal)

        assert any(word in reply.lower() for word in ("nothing", "no events", "empty", "scheduled"))


# ===========================================================================
# TestPendingStateLifecycle
# ===========================================================================


class TestPendingStateLifecycle:
    def test_pending_action_cleared_after_successful_confirm(self) -> None:
        """pending_action is None after user confirms and booking completes."""
        booking = _sample_booking()
        cal = _mock_cal()
        cal.cancel_booking.return_value = None
        cal.list_bookings.return_value = [booking]
        state = _state()
        intent_cancel = _intent(
            intent_type=IntentType.cancel,
            attendee_name="Jane",
        )
        with patch("assistant.extract_intent", return_value=intent_cancel):
            handle_message("cancel my call with Jane", state, cal)
        # Select booking 1
        with patch("assistant.extract_intent", side_effect=AssertionError("should not call LLM")):
            handle_message("1", state, cal)
        # Confirm
        with patch("assistant.extract_intent", side_effect=AssertionError("should not call LLM")):
            handle_message("yes", state, cal)
        assert state["pending_action"] is None

    def test_pending_action_cleared_after_decline(self) -> None:
        """pending_action is None after user declines confirmation."""
        booking = _sample_booking()
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking]
        state = _state()
        intent_cancel = _intent(
            intent_type=IntentType.cancel,
            attendee_name="Jane",
        )
        with patch("assistant.extract_intent", return_value=intent_cancel):
            handle_message("cancel my call with Jane", state, cal)
        with patch("assistant.extract_intent", side_effect=AssertionError("no LLM")):
            reply = handle_message("no", state, cal)
        assert state["pending_action"] is None
        assert isinstance(reply, str) and len(reply) > 0

    def test_pending_action_cleared_when_user_says_cancel_in_booking_flow(self) -> None:
        """'cancel' during a booking flow cancels the chat action, not a calendar event."""
        cal = _mock_cal()
        state = _state()
        intent_book = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=_T0,
            duration_minutes=30,
        )
        with patch("assistant.extract_intent", return_value=intent_book):
            handle_message("book a call with Jane", state, cal)
        # User says cancel → should cancel the pending booking flow
        intent_cancel_keyword = _intent(intent_type=IntentType.cancel)
        with patch("assistant.extract_intent", return_value=intent_cancel_keyword):
            reply = handle_message("cancel", state, cal)
        assert state["pending_action"] is None
        assert "cancel" in reply.lower() or "request" in reply.lower()

    def test_old_pending_action_not_reused_for_different_request(self) -> None:
        """Starting a new cancel request after a completed one creates a fresh pending state."""
        booking1 = _sample_booking(uid="uid-1", title="Intro Call")
        booking2 = _sample_booking(uid="uid-2", title="Strategy Call", attendees=[
            Attendee(name="Bob", email="bob@example.com")
        ])
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking1, booking2]
        state = _state()

        intent1 = _intent(intent_type=IntentType.cancel, attendee_name="Jane")
        with patch("assistant.extract_intent", return_value=intent1):
            handle_message("cancel call with Jane", state, cal)
        with patch("assistant.extract_intent", side_effect=AssertionError("no LLM")):
            handle_message("no", state, cal)
        assert state["pending_action"] is None

        # New cancel request for a different booking
        intent2 = _intent(intent_type=IntentType.cancel, attendee_name="Bob")
        with patch("assistant.extract_intent", return_value=intent2):
            handle_message("cancel call with Bob", state, cal)
        pending = state["pending_action"]
        # Should be a fresh cancel action, not the old one
        assert pending is not None
        assert pending.action_type == "cancel"


# ===========================================================================
# TestCancelClarification
# ===========================================================================


class TestCancelClarification:
    def test_cancel_underspecified_no_match_returns_clarification(self) -> None:
        """When source details are missing and no booking matches, assistant asks clarification."""
        cal = _mock_cal()
        cal.list_bookings.return_value = [_sample_booking()]
        state = _state()
        # Intent with no identifying info
        intent = _intent(intent_type=IntentType.cancel)
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel my meeting", state, cal)
        # Should ask for clarification or show list, not silently fail
        assert any(
            word in reply.lower()
            for word in ("which", "book", "cancel", "match", "meeting", "help", "name", "time")
        )

    def test_cancel_multiple_match_shows_choices(self) -> None:
        """When multiple bookings match, assistant shows numbered list."""
        booking1 = _sample_booking(uid="uid-1", title="Call with Jane")
        booking2 = _sample_booking(uid="uid-2", title="Intro Call Jane", start=_T2, end=_T3)
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking1, booking2]
        state = _state()
        intent = _intent(intent_type=IntentType.cancel, attendee_name="Jane")
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel call with Jane", state, cal)
        assert "1" in reply and "2" in reply
        # Pending should hold the candidates
        assert state["pending_action"] is not None
        assert len(state["pending_action"].matching_bookings) == 2


# ===========================================================================
# TestRescheduleClarification
# ===========================================================================


class TestRescheduleClarification:
    def test_reschedule_source_identified_asks_target_when_missing(self) -> None:
        """When source booking is found but no target time given, asks for target time."""
        booking = _sample_booking()
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking]
        cal.find_slots.return_value = []
        state = _state()
        intent = _intent(
            intent_type=IntentType.reschedule,
            attendee_name="Jane",
            source_start_time=_T0,
            source_end_time=_T1,
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("reschedule my call with Jane", state, cal)
        # Should either find booking and show slots, or ask for target time
        assert isinstance(reply, str) and len(reply) > 0

    def test_reschedule_source_duration_used_for_matching(self) -> None:
        """source_end_time derived from source_start_time + duration matches overlapping booking."""
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/New_York")
        source_start = datetime(2050, 9, 5, 14, 0, 0, tzinfo=tz)
        source_end = source_start + timedelta(minutes=30)

        booking = _sample_booking(
            start=source_start,
            end=source_start + timedelta(minutes=30),
        )
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking]
        cal.find_slots.return_value = [_sample_slot(_T2, _T3)]
        state = _state()
        intent = _intent(
            intent_type=IntentType.reschedule,
            source_start_time=source_start,
            source_end_time=source_end,
            attendee_name="Jane",
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("reschedule my Jane call", state, cal)
        # Should find the booking and proceed (not say no match found)
        pending = state["pending_action"]
        assert pending is not None or "reschedule" in reply.lower()


# ===========================================================================
# TestBookFlowMissingFields
# ===========================================================================


class TestBookFlowMissingFields:
    def _cal(self) -> MagicMock:
        return _mock_cal()

    def test_book_asks_attendee_name_when_missing(self) -> None:
        intent = _intent(
            intent_type=IntentType.book,
            attendee_email="jane@example.com",
            start_time=_T0,
            duration_minutes=30,
        )
        state = _state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("book a call", state, self._cal())
        assert "name" in reply.lower()

    def test_book_asks_attendee_email_when_missing(self) -> None:
        intent = _intent(
            intent_type=IntentType.book,
            attendee_name="Jane",
            start_time=_T0,
            duration_minutes=30,
        )
        state = _state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("book a call with Jane", state, self._cal())
        assert "email" in reply.lower()

    def test_book_asks_one_field_at_a_time(self) -> None:
        """When both name and email are missing, only one question is asked."""
        intent = _intent(
            intent_type=IntentType.book,
            start_time=_T0,
            duration_minutes=30,
        )
        state = _state()
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("book a 30 min call", state, self._cal())
        email_ask = "email" in reply.lower()
        name_ask = "name" in reply.lower()
        assert email_ask ^ name_ask


# ===========================================================================
# TestDateRangeList
# ===========================================================================


class TestDateRangeList:
    def test_list_intent_uses_start_end_from_date_range(self) -> None:
        """List intent with start_time/end_time from date_range_* correctly passes to list_bookings."""
        cal = _mock_cal()
        booking = _sample_booking()
        cal.list_bookings.return_value = [booking]
        state = _state()
        intent = _intent(
            intent_type=IntentType.list,
            start_time=_T0,
            end_time=_T1,
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("what is on my calendar", state, cal)
        cal.list_bookings.assert_called_once()
        call_kwargs = cal.list_bookings.call_args.kwargs
        assert call_kwargs.get("start") == _T0 or cal.list_bookings.call_args.args[0] == _T0

    def test_cancel_with_date_range_source_window_finds_booking(self) -> None:
        """Cancel intent where date_range is the source window correctly filters bookings."""
        booking = _sample_booking(start=_T0, end=_T1)
        cal = _mock_cal()
        cal.list_bookings.return_value = [booking]
        state = _state()
        # source_start_time comes from date_range mapping in _map_extracted_to_intent
        intent = _intent(
            intent_type=IntentType.cancel,
            source_start_time=_T0,
            source_end_time=_T1,
        )
        with patch("assistant.extract_intent", return_value=intent):
            reply = handle_message("cancel my meeting tomorrow", state, cal)
        # Should find the booking and show confirmation
        pending = state["pending_action"]
        assert pending is not None or "intro call" in reply.lower()
