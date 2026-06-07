from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from cal_client import CalClient
from schemas import (
    AssistantError,
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

_AFFIRMATIVES = {"yes", "y", "confirm", "sure", "ok", "okay", "yep", "yeah", "do it"}
_NEGATIVES = {"no", "n", "nope", "nah", "never mind", "nevermind", "don't", "dont"}

_INTENT_PROMPT_BASE = """\
Extract the scheduling intent from the user's message and return JSON only.

Current date and time: {now_iso} (timezone: {default_tz})

Required field: intent_type — one of: list, book, cancel, reschedule, unknown.

Optional fields:
- search_text: relevant names/keywords
- attendee_name: person's name
- attendee_email: email address
- duration_minutes: integer
- start_time: ISO 8601 with timezone offset — resolve relative expressions like \
"Thursday afternoon", "later today", "next week" to concrete datetimes using the \
current date above. Set to null if the expression genuinely cannot be resolved.
- end_time: ISO 8601 with timezone offset — search window end. If you set start_time, \
also derive end_time based on the time-of-day preference (morning=08:00-12:00, \
afternoon=12:00-17:00, evening=17:00-21:00, exact time=start+duration or +30min).
- timezone: IANA timezone string
- booking_uid: booking UID if mentioned
- time_preference: set ONLY when the time expression truly cannot be resolved to \
a concrete start_time (e.g. "sometime next week" with no specific day)

Return ONLY valid JSON. No explanation."""


def _build_intent_prompt(now: datetime, default_tz: str) -> str:
    return _INTENT_PROMPT_BASE.format(
        now_iso=now.isoformat(),
        default_tz=default_tz,
    )


def _create_openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AssistantError(
            "OpenAI SDK is not installed. Run `pip install -r requirements.txt`.",
            reason="llm_failure",
        ) from exc
    return OpenAI(api_key=api_key)


def _openai_messages(messages: list[dict]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    allowed_roles = {"user", "assistant", "system", "developer"}

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if content is None:
            continue
        if role not in allowed_roles:
            role = "user"
        normalized.append({"role": role, "content": str(content)})

    return normalized


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    for item in getattr(response, "output", []) or []:
        content = getattr(item, "content", None)
        if isinstance(item, dict):
            content = item.get("content", content)
        for part in content or []:
            text = getattr(part, "text", None)
            if isinstance(part, dict):
                text = part.get("text", text)
            if isinstance(text, str) and text.strip():
                return text

    raise AssistantError("LLM response did not include text output", reason="bad_json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_intent(user_text: str, conversation_history: list[dict]) -> UserIntent:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise AssistantError("Missing OPENAI_API_KEY", reason="llm_failure")
    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise AssistantError("Missing LLM_MODEL", reason="llm_failure")

    client = _create_openai_client(api_key)
    default_tz = os.environ.get("CAL_TIMEZONE", "America/New_York")
    now = datetime.now(timezone.utc)

    messages = [
        {"role": "developer", "content": "Return JSON only."},
        *conversation_history,
        {"role": "user", "content": user_text},
    ]

    try:
        response = client.responses.create(
            model=model,
            instructions=_build_intent_prompt(now, default_tz),
            input=_openai_messages(messages),
            max_output_tokens=512,
            text={"format": {"type": "json_object"}, "verbosity": "low"},
        )
    except Exception as exc:
        raise AssistantError(f"LLM API error: {exc}", reason="llm_failure") from exc

    raw = _extract_response_text(response)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise AssistantError(
            f"LLM returned invalid JSON: {exc}", reason="bad_json"
        ) from exc

    if "intent_type" not in data:
        raise AssistantError(
            "LLM response missing intent_type field", reason="missing_field"
        )

    try:
        return UserIntent(**data)
    except Exception as exc:
        raise IntentValidationError(
            f"Invalid date or field in intent: {exc}", reason="invalid_date"
        ) from exc


def handle_message(
    user_text: str, session_state: dict, cal_client: CalClient
) -> str:
    pending: Optional[PendingAction] = session_state.get("pending_action")
    available_slots: list[Slot] = session_state.get("available_slots", [])

    # --- "yes" with no pending action ---
    if pending is None and _is_affirmative(user_text):
        return "What would you like to do? I can help with booking, canceling, or rescheduling."

    # --- Confirmation phase: pending action waiting for yes/no ---
    if pending is not None and _in_confirmation_phase(pending):
        return _handle_confirmation(user_text, pending, session_state, cal_client)

    # --- Slot selection phase: slots shown, pick one ---
    if available_slots and pending is not None:
        if _is_cancel_word(user_text):
            session_state["pending_action"] = None
            session_state["available_slots"] = []
            _clear_reschedule_state(session_state)
            return "Request cancelled. What would you like to do?"
        if pending.action_type == "book":
            return _handle_slot_selection(
                user_text, pending, available_slots, session_state, cal_client
            )
        if pending.action_type == "reschedule":
            return _handle_slot_selection_for_reschedule(
                user_text, session_state, available_slots
            )

    # --- Multiple-match selection: bypass LLM entirely ---
    if (
        pending is not None
        and pending.matching_bookings
        and pending.action_type in ("cancel", "reschedule")
    ):
        return _handle_match_selection(user_text, pending, session_state, cal_client)

    # --- Short duration follow-up: "15", "30 min", etc. ---
    duration = _parse_duration_minutes(user_text)
    if (
        pending is not None
        and pending.action_type == "book"
        and pending.booking_draft is not None
        and pending.booking_draft.duration_minutes is None
        and duration is not None
    ):
        intent = UserIntent(intent_type=IntentType.book, duration_minutes=duration)
        return _handle_book(intent, session_state, cal_client)

    # --- LLM intent extraction ---
    try:
        history = session_state.get("messages", [])
        intent = extract_intent(user_text, history)
    except AssistantError as exc:
        return _handle_assistant_error(exc)
    except IntentValidationError as exc:
        return _handle_validation_error(exc)

    # --- "cancel" keyword during booking draft collection ---
    if (
        pending is not None
        and pending.action_type == "book"
        and intent.intent_type == IntentType.cancel
        and not intent.search_text
        and not intent.booking_uid
    ):
        session_state["pending_action"] = None
        return "Booking request cancelled. What would you like to do?"

    # --- Intent switch resets unrelated pending action ---
    if pending is not None and pending.action_type != intent.intent_type.value:
        if intent.intent_type not in (IntentType.unknown,):
            session_state["pending_action"] = None
            _clear_reschedule_state(session_state)
            pending = None

    try:
        if intent.intent_type == IntentType.list:
            return _handle_list(intent, session_state, cal_client)
        elif intent.intent_type == IntentType.book:
            return _handle_book(intent, session_state, cal_client)
        elif intent.intent_type == IntentType.cancel:
            return _handle_cancel(intent, session_state, cal_client)
        elif intent.intent_type == IntentType.reschedule:
            return _handle_reschedule(intent, session_state, cal_client)
        else:
            return (
                "I can only help with scheduling — booking, canceling, or rescheduling events."
            )
    except CalClientError as exc:
        return _format_cal_error(exc)
    except IntentValidationError as exc:
        return _handle_validation_error(exc)


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------


def _handle_list(
    intent: UserIntent, session_state: dict, cal_client: CalClient
) -> str:
    bookings = cal_client.list_bookings(start=intent.start_time, end=intent.end_time)
    if not bookings:
        return "You have nothing scheduled for that period."
    lines = ["Here's what's coming up:"]
    for b in bookings:
        tz_label = b.start.strftime("%Z") or ""
        lines.append(f"• {b.title} — {b.start.strftime('%b %-d, %I:%M %p')} {tz_label}".strip())
    return "\n".join(lines)


def _handle_book(
    intent: UserIntent, session_state: dict, cal_client: CalClient
) -> str:
    pending: Optional[PendingAction] = session_state.get("pending_action")

    # Merge into existing draft or create new one
    if pending and pending.action_type == "book" and pending.booking_draft is not None:
        draft = pending.booking_draft
        _merge_intent_into_draft(draft, intent)
    else:
        default_tz = os.environ.get("CAL_TIMEZONE", "America/New_York")
        draft = BookingDraft(
            attendee_name=intent.attendee_name,
            attendee_email=intent.attendee_email,
            duration_minutes=intent.duration_minutes,
            start_time=intent.start_time,
            end_time=intent.end_time,
            timezone=intent.timezone or default_tz,
            time_preference=intent.time_preference,
        )
        pending = PendingAction(action_type="book", booking_draft=draft)
        session_state["pending_action"] = pending

    # Validate email if present
    if draft.attendee_email and not _is_valid_email(draft.attendee_email):
        raise IntentValidationError("Invalid email address", reason="invalid_email")

    # Validate start_time not in the past
    if draft.start_time is not None:
        now = datetime.now(tz=draft.start_time.tzinfo)
        if draft.start_time < now:
            raise IntentValidationError(
                "Requested time is in the past", reason="past_date"
            )

    # Ask for next missing field (name, email, or time)
    missing = draft.missing_fields()
    if missing:
        return _ask_for_field(missing[0])

    # All structural fields present — but we need a concrete start_time before fetching slots
    if draft.start_time is None:
        return "What specific date and time works? For example, 'Thursday at 2pm'."

    event_type_prompt = _ensure_booking_event_type(draft, cal_client)
    if event_type_prompt is not None:
        return event_type_prompt

    # All fields present with concrete start time — fetch slots
    return _fetch_and_show_slots(draft, session_state, cal_client)


def _handle_cancel(
    intent: UserIntent, session_state: dict, cal_client: CalClient
) -> str:
    pending: Optional[PendingAction] = session_state.get("pending_action")

    # User is picking from previously listed matches
    if (
        pending is not None
        and pending.action_type == "cancel"
        and pending.matching_bookings
    ):
        selected = _pick_booking(intent, pending.matching_bookings, user_text=None)
        if selected:
            pending.cancel_request = CancelRequest(booking_uid=selected.uid)
            pending.matching_bookings = []
            session_state["pending_action"] = pending
            return _cancel_confirmation_text(selected)

    # Search upcoming bookings first
    upcoming = cal_client.list_bookings(status="upcoming")
    matches = _filter_bookings(upcoming, intent)

    if not matches:
        # Check if the booking exists but is already cancelled
        cancelled_bookings = cal_client.list_bookings(status="cancelled")
        if _filter_bookings(cancelled_bookings, intent):
            return "That booking is already cancelled."
        return "I couldn't find a matching booking."

    if len(matches) > 1:
        pending = PendingAction(action_type="cancel", matching_bookings=matches)
        session_state["pending_action"] = pending
        return _multiple_matches_text(matches, "cancel")

    booking = matches[0]
    pending = PendingAction(
        action_type="cancel",
        cancel_request=CancelRequest(booking_uid=booking.uid),
    )
    session_state["pending_action"] = pending
    return _cancel_confirmation_text(booking)


def _handle_reschedule(
    intent: UserIntent, session_state: dict, cal_client: CalClient
) -> str:
    pending: Optional[PendingAction] = session_state.get("pending_action")
    available_slots: list[Slot] = session_state.get("available_slots", [])

    # User is picking from previously listed matches
    if (
        pending is not None
        and pending.action_type == "reschedule"
        and pending.matching_bookings
    ):
        selected = _pick_booking(intent, pending.matching_bookings, user_text=None)
        if selected:
            pending.matching_bookings = []
            session_state["pending_action"] = pending
            return _fetch_reschedule_slots(selected, intent, session_state, cal_client)

    # Continue a reschedule after "no slots" — bypass full search when uid is known
    stored_uid = session_state.get("_reschedule_booking_uid")
    if (
        stored_uid
        and pending is not None
        and pending.action_type == "reschedule"
        and not pending.matching_bookings
        and intent.start_time is not None
    ):
        upcoming = cal_client.list_bookings(status="upcoming")
        uid_match = next((b for b in upcoming if b.uid == stored_uid), None)
        if uid_match:
            return _fetch_reschedule_slots(uid_match, intent, session_state, cal_client)

    # Search upcoming bookings first
    upcoming = cal_client.list_bookings(status="upcoming")
    matches = _filter_bookings(upcoming, intent)

    if not matches:
        # Check if the booking exists but is already cancelled
        cancelled_bookings = cal_client.list_bookings(status="cancelled")
        if _filter_bookings(cancelled_bookings, intent):
            return "That booking is already cancelled and can't be rescheduled."
        return "I couldn't find a matching booking. Could you provide more details?"

    if len(matches) > 1:
        pending = PendingAction(action_type="reschedule", matching_bookings=matches)
        session_state["pending_action"] = pending
        session_state["_reschedule_original_intent"] = intent
        return _multiple_matches_text(matches, "reschedule")

    booking = matches[0]
    return _fetch_reschedule_slots(booking, intent, session_state, cal_client)


# ---------------------------------------------------------------------------
# Confirmation handling
# ---------------------------------------------------------------------------


def _in_confirmation_phase(pending: PendingAction) -> bool:
    return (
        pending.booking_request is not None
        or pending.cancel_request is not None
        or pending.reschedule_request is not None
    )


def _handle_confirmation(
    user_text: str,
    pending: PendingAction,
    session_state: dict,
    cal_client: CalClient,
) -> str:
    if _is_affirmative(user_text):
        try:
            return _execute_confirmed_action(pending, session_state, cal_client)
        except CalClientError as exc:
            if exc.status_code is None:
                if exc.reason in ("timeout", "network"):
                    return "Cal.com timed out or could not be reached. Please try again."
                if exc.reason == "malformed":
                    return "Cal.com returned an unexpected response. Please try again."
                if exc.reason == "slot_unavailable" and pending.booking_request is not None:
                    session_state["available_slots"] = []
                    pending.booking_request = None
                    session_state["pending_action"] = pending
                    return (
                        "That slot is no longer available. "
                        "Please choose another time."
                    )
                return "Something went wrong with Cal.com. Please try again."
            return _format_cal_error(exc)
    elif _is_negative(user_text) or _is_cancel_word(user_text):
        session_state["pending_action"] = None
        _clear_reschedule_state(session_state)
        return "Got it, no changes made."
    else:
        return _restate_confirmation(pending)


def _execute_confirmed_action(
    pending: PendingAction, session_state: dict, cal_client: CalClient
) -> str:
    if pending.booking_request is not None:
        booking = cal_client.create_booking(pending.booking_request)
        session_state["pending_action"] = None
        tz = booking.start.strftime("%Z") or ""
        return (
            f"Done! '{booking.title}' is booked for "
            f"{booking.start.strftime('%b %-d, %I:%M %p')} {tz}."
        ).strip()

    if pending.cancel_request is not None:
        cal_client.cancel_booking(pending.cancel_request.booking_uid)
        session_state["pending_action"] = None
        return "Booking cancelled."

    if pending.reschedule_request is not None:
        booking = cal_client.reschedule_booking(
            pending.reschedule_request.booking_uid,
            pending.reschedule_request.new_start_time,
        )
        session_state["pending_action"] = None
        _clear_reschedule_state(session_state)
        tz = booking.start.strftime("%Z") or ""
        return (
            f"Rescheduled! '{booking.title}' is now at "
            f"{booking.start.strftime('%b %-d, %I:%M %p')} {tz}."
        ).strip()

    return "Nothing to confirm."


def _restate_confirmation(pending: PendingAction) -> str:
    if pending.booking_request is not None:
        return "Say yes to confirm or no to cancel."
    if pending.cancel_request is not None:
        return "Say yes to cancel this booking, or no to keep it."
    if pending.reschedule_request is not None:
        return "Say yes to reschedule, or no to keep the original time."
    return "Say yes to confirm or no to cancel."


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------


def _handle_slot_selection(
    user_text: str,
    pending: PendingAction,
    available_slots: list[Slot],
    session_state: dict,
    cal_client: CalClient,
) -> str:
    slot = _pick_slot(user_text, available_slots)
    if slot is None:
        return _format_slot_list(available_slots) + "\n\nWhich slot works for you?"

    draft = pending.booking_draft
    if draft is None or draft.event_type_id is None:
        return "Something went wrong. Let's start over — what would you like to book?"

    request = BookingRequest(
        attendee_name=draft.attendee_name or "",
        attendee_email=draft.attendee_email or "",
        start_time=slot.start,
        duration_minutes=draft.duration_minutes or 30,
        timezone=draft.timezone or "UTC",
        event_type_id=draft.event_type_id,
        include_length_in_minutes=getattr(draft, "include_length_in_minutes", False),
    )
    pending.booking_request = request
    pending.selected_slot = slot
    session_state["pending_action"] = pending
    session_state["available_slots"] = []

    tz = slot.start.strftime("%Z") or ""
    return (
        f"Book with {request.attendee_name} on "
        f"{slot.start.strftime('%b %-d at %I:%M %p')} {tz}? "
        "Say yes or no."
    ).strip()


def _pick_slot(user_text: str, slots: list[Slot]) -> Optional[Slot]:
    text = user_text.strip().lower()
    ordinal_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    for word, idx in ordinal_map.items():
        if word in text and idx < len(slots):
            return slots[idx]
    m = re.search(r"\b([1-9])\b", text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(slots):
            return slots[idx]
    return None


def _derive_end_time(
    start: datetime,
    time_preference: Optional[str],
    duration_minutes: Optional[int],
    tz: str,
) -> datetime:
    """Derive a bounded search window end from the start time and intent context."""
    pref = (time_preference or "").lower()

    # Named time-of-day windows use local timezone boundaries
    try:
        local_tz = ZoneInfo(tz)
    except Exception:
        local_tz = timezone.utc

    local_start = start.astimezone(local_tz)
    local_date = local_start.date()

    if "morning" in pref:
        end_local = datetime(local_date.year, local_date.month, local_date.day, 12, 0, 0, tzinfo=local_tz)
        return end_local.astimezone(timezone.utc)
    if "afternoon" in pref:
        end_local = datetime(local_date.year, local_date.month, local_date.day, 17, 0, 0, tzinfo=local_tz)
        return end_local.astimezone(timezone.utc)
    if "evening" in pref:
        end_local = datetime(local_date.year, local_date.month, local_date.day, 21, 0, 0, tzinfo=local_tz)
        return end_local.astimezone(timezone.utc)

    # Exact time: narrow window based on duration or default 30min
    minutes = max(duration_minutes or 30, 30)
    return start + timedelta(minutes=minutes)


def _ensure_booking_event_type(
    draft: BookingDraft, cal_client: CalClient
) -> Optional[str]:
    if draft.event_type_id is not None:
        return None

    event_types = cal_client.list_event_types()
    if not event_types:
        return "I couldn't find any Cal.com event types. Please create one in Cal.com first."

    usable_event_types = _prefer_visible_event_types(event_types)
    if draft.duration_minutes is None:
        durations = _available_durations(usable_event_types)
        if len(durations) == 1:
            draft.duration_minutes = durations[0]
        else:
            return f"How long should it be — {_format_duration_options(durations)} minutes?"

    selected = _select_event_type_for_duration(usable_event_types, draft.duration_minutes)
    if selected is None:
        durations = _available_durations(usable_event_types)
        return (
            f"I don't see a {draft.duration_minutes}-minute event type. "
            f"Available options are {_format_duration_options(durations)} minutes."
        )

    draft.event_type_id = selected.id
    draft.include_length_in_minutes = _has_multiple_durations(selected)
    if draft.duration_minutes is None:
        draft.duration_minutes = selected.length_minutes
    return None


def _event_type_id_for_booking(
    booking: Booking, cal_client: CalClient
) -> Optional[int]:
    if booking.event_type_id is not None:
        return booking.event_type_id

    duration = round((booking.end - booking.start).total_seconds() / 60)
    try:
        event_types = cal_client.list_event_types()
    except CalClientError:
        raise
    except Exception:
        return None
    selected = _select_event_type_for_duration(
        _prefer_visible_event_types(event_types), duration
    )
    return selected.id if selected else None


def _prefer_visible_event_types(event_types: list[EventType]) -> list[EventType]:
    visible = [event_type for event_type in event_types if not event_type.hidden]
    return visible or event_types


def _available_durations(event_types: list[EventType]) -> list[int]:
    durations: set[int] = set()
    for event_type in event_types:
        durations.update(event_type.supported_durations())
    return sorted(durations)


def _has_multiple_durations(event_type: EventType) -> bool:
    return len(event_type.supported_durations()) > 1


def _select_event_type_for_duration(
    event_types: list[EventType], duration_minutes: Optional[int]
) -> Optional[EventType]:
    if duration_minutes is None:
        return event_types[0] if len(event_types) == 1 else None

    candidates = [
        event_type
        for event_type in event_types
        if duration_minutes in event_type.supported_durations()
    ]
    if not candidates:
        return None

    exact_length = [
        event_type
        for event_type in candidates
        if event_type.length_minutes == duration_minutes
    ]
    preferred = exact_length or candidates
    return sorted(preferred, key=lambda event_type: event_type.title.lower())[0]


def _format_duration_options(durations: list[int]) -> str:
    if not durations:
        return "an available duration"
    labels = [str(duration) for duration in durations]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return f"{', '.join(labels[:-1])}, or {labels[-1]}"


def _fetch_and_show_slots(
    draft: BookingDraft,
    session_state: dict,
    cal_client: CalClient,
    booking_uid_to_reschedule: Optional[str] = None,
) -> str:
    # start_time must be set before calling this function
    start = draft.start_time
    if start is None:
        return "What specific date and time works? For example, 'Thursday at 2pm'."

    end = draft.end_time or _derive_end_time(
        start,
        draft.time_preference,
        draft.duration_minutes,
        draft.timezone or "UTC",
    )

    slots = cal_client.find_slots(
        start=start,
        end=end,
        duration_minutes=draft.duration_minutes,
        timezone=draft.timezone,
        booking_uid_to_reschedule=booking_uid_to_reschedule,
        event_type_id=draft.event_type_id,
    )

    if not slots:
        return "No available slots in that window. What other time works for you?"

    session_state["available_slots"] = slots
    return _format_slot_list(slots) + "\n\nWhich slot works for you?"


def _fetch_reschedule_slots(
    booking: Booking,
    intent: UserIntent,
    session_state: dict,
    cal_client: CalClient,
) -> str:
    # Require a concrete start_time — if missing, ask for it
    if intent.start_time is None:
        return "What specific day and time would you like to reschedule to?"

    default_tz = os.environ.get("CAL_TIMEZONE", "America/New_York")
    tz = intent.timezone or default_tz

    start = intent.start_time
    end = intent.end_time or _derive_end_time(
        start,
        intent.time_preference,
        None,
        tz,
    )
    event_type_id = _event_type_id_for_booking(booking, cal_client)
    if event_type_id is None:
        return "I couldn't identify the event type for that booking."

    slots = cal_client.find_slots(
        start=start,
        end=end,
        timezone=tz,
        booking_uid_to_reschedule=booking.uid,
        event_type_id=event_type_id,
    )

    if not slots:
        session_state["_reschedule_booking_uid"] = booking.uid
        session_state["pending_action"] = PendingAction(action_type="reschedule")
        return "No available slots in that window. What other time works for you?"

    # Store the booking we're rescheduling in a temporary draft
    draft = BookingDraft(
        attendee_name=booking.attendees[0].name if booking.attendees else None,
        event_type_id=event_type_id,
        timezone=tz,
    )
    pending = PendingAction(
        action_type="reschedule",
        booking_draft=draft,
        reschedule_request=None,
    )
    session_state["pending_action"] = pending
    session_state["_reschedule_booking_uid"] = booking.uid
    session_state["available_slots"] = slots

    return _format_slot_list(slots) + "\n\nWhich slot works for you?"


def _handle_slot_selection_for_reschedule(
    user_text: str,
    session_state: dict,
    slots: list[Slot],
) -> str:
    slot = _pick_slot(user_text, slots)
    if slot is None:
        return _format_slot_list(slots) + "\n\nWhich slot works for you?"

    booking_uid = session_state.get("_reschedule_booking_uid", "")
    pending = session_state.get("pending_action")
    if pending is None:
        pending = PendingAction(action_type="reschedule")

    pending.reschedule_request = RescheduleRequest(
        booking_uid=booking_uid, new_start_time=slot.start
    )
    pending.selected_slot = slot
    session_state["pending_action"] = pending
    session_state["available_slots"] = []

    tz = slot.start.strftime("%Z") or ""
    return (
        f"Reschedule to {slot.start.strftime('%b %-d at %I:%M %p')} {tz}? "
        "Say yes or no."
    ).strip()


# ---------------------------------------------------------------------------
# Matching and filtering helpers
# ---------------------------------------------------------------------------


def _handle_match_selection(
    user_text: str, pending: PendingAction, session_state: dict, cal_client: CalClient
) -> str:
    selected = _pick_booking_by_text(user_text, pending.matching_bookings)
    if selected is None:
        return (
            _multiple_matches_text(pending.matching_bookings, pending.action_type)
            + "\n\nPlease reply with a number."
        )
    pending.matching_bookings = []
    session_state["pending_action"] = pending
    if pending.action_type == "cancel":
        pending.cancel_request = CancelRequest(booking_uid=selected.uid)
        return _cancel_confirmation_text(selected)
    effective_intent = session_state.pop("_reschedule_original_intent", None) \
        or UserIntent(intent_type=IntentType.reschedule)
    return _fetch_reschedule_slots(selected, effective_intent, session_state, cal_client)


def _pick_booking_by_text(user_text: str, bookings: list[Booking]) -> Optional[Booking]:
    text = user_text.strip().lower()
    ordinal_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    for word, idx in ordinal_map.items():
        if word in text and idx < len(bookings):
            return bookings[idx]
    m = re.search(r"\b([1-9])\b", text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(bookings):
            return bookings[idx]
    for b in bookings:
        if b.uid == user_text.strip():
            return b
    return None


def _filter_bookings(bookings: list[Booking], intent: UserIntent) -> list[Booking]:
    search = (intent.search_text or "").lower()
    name = (intent.attendee_name or "").lower()
    email = (intent.attendee_email or "").lower()

    results: list[Booking] = []
    for b in bookings:
        if intent.booking_uid and b.uid == intent.booking_uid:
            results.append(b)
            continue
        title_match = search and search in b.title.lower()
        search_attendee_match = search and any(
            search in a.name.lower() or search in a.email.lower()
            for a in b.attendees
        )
        attendee_name_match = name and any(
            name in a.name.lower() for a in b.attendees
        )
        attendee_email_match = email and any(
            email in a.email.lower() for a in b.attendees
        )
        if title_match or search_attendee_match or attendee_name_match or attendee_email_match:
            results.append(b)

    # If no filters provided, return all
    if not search and not name and not email and not intent.booking_uid:
        return bookings
    return results


def _pick_booking(
    intent: UserIntent, bookings: list[Booking], user_text: Optional[str]
) -> Optional[Booking]:
    if intent.booking_uid:
        for b in bookings:
            if b.uid == intent.booking_uid:
                return b
    if user_text:
        text = user_text.lower()
        m = re.search(r"\b([1-9])\b", text)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(bookings):
                return bookings[idx]
    return None


def _merge_intent_into_draft(draft: BookingDraft, intent: UserIntent) -> None:
    if intent.attendee_name:
        draft.attendee_name = intent.attendee_name
    if intent.attendee_email:
        draft.attendee_email = intent.attendee_email
    if intent.duration_minutes:
        draft.duration_minutes = intent.duration_minutes
    if intent.start_time:
        draft.start_time = intent.start_time
    if intent.end_time:
        draft.end_time = intent.end_time
    if intent.timezone:
        draft.timezone = intent.timezone
    if intent.time_preference:
        draft.time_preference = intent.time_preference


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _ask_for_field(field: str) -> str:
    prompts = {
        "attendee_name": "What's their name?",
        "attendee_email": "What's their email?",
        "time": "What time works for them?",
    }
    return prompts.get(field, f"What's the {field.replace('_', ' ')}?")


def _format_slot_list(slots: list[Slot]) -> str:
    lines = ["Here are the available slots:"]
    for i, slot in enumerate(slots[:5], 1):
        tz = slot.start.strftime("%Z") or ""
        lines.append(f"{i}. {slot.start.strftime('%b %-d, %I:%M %p')} {tz}".strip())
    return "\n".join(lines)


def _cancel_confirmation_text(booking: Booking) -> str:
    tz = booking.start.strftime("%Z") or ""
    return (
        f"Cancel '{booking.title}' on "
        f"{booking.start.strftime('%b %-d at %I:%M %p')} {tz}? "
        "Say yes or no."
    ).strip()


def _multiple_matches_text(bookings: list[Booking], action: str) -> str:
    lines = [f"I found {len(bookings)} matching bookings. Which one do you mean?"]
    for i, b in enumerate(bookings, 1):
        tz = b.start.strftime("%Z") or ""
        lines.append(f"{i}. {b.title} — {b.start.strftime('%b %-d, %I:%M %p')} {tz}".strip())
    return "\n".join(lines)


def _format_cal_error(exc: CalClientError) -> str:
    if exc.status_code == 401:
        return "There's an issue with the Cal.com API key. Please check your configuration."
    if exc.status_code == 400:
        return f"Cal.com rejected the booking request: {exc.message}"
    if exc.status_code == 429:
        return "Cal.com is busy right now. Please try again in a moment."
    return "Something went wrong with Cal.com. Please try again."


def _handle_assistant_error(exc: AssistantError) -> str:
    if exc.reason == "llm_failure":
        return "I'm having trouble right now. Please try again."
    if exc.reason == "bad_json":
        return "I didn't catch that — could you rephrase?"
    return "I couldn't understand that request."


def _handle_validation_error(exc: IntentValidationError) -> str:
    if exc.reason == "invalid_date":
        return "That date doesn't look right. What date did you mean?"
    if exc.reason == "past_date":
        return "I can only book future times."
    if exc.reason == "invalid_email":
        return "That email doesn't look right. What's their email?"
    return f"I couldn't process that: {exc.message}"


# ---------------------------------------------------------------------------
# String utils
# ---------------------------------------------------------------------------


def _is_affirmative(text: str) -> bool:
    return text.strip().lower() in _AFFIRMATIVES


def _is_negative(text: str) -> bool:
    return text.strip().lower() in _NEGATIVES


def _is_cancel_word(text: str) -> bool:
    return text.strip().lower() == "cancel"


def _parse_duration_minutes(text: str) -> Optional[int]:
    normalized = text.strip().lower()
    match = re.fullmatch(r"(\d{1,3})(?:\s*(?:m|min|mins|minute|minutes))?", normalized)
    if match is None:
        return None
    return int(match.group(1))


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def _clear_reschedule_state(session_state: dict) -> None:
    session_state.pop("_reschedule_booking_uid", None)
    session_state.pop("_reschedule_original_intent", None)
