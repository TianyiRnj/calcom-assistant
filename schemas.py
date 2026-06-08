from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


def _require_tz_aware(v: datetime | None) -> datetime | None:
    if v is not None and v.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return v


class IntentType(str, Enum):
    list = "list"
    book = "book"
    cancel = "cancel"
    reschedule = "reschedule"
    unknown = "unknown"


class Attendee(BaseModel):
    name: str
    email: str


class ExtractedAttendee(BaseModel):
    """Extraction-only attendee shape from LLM output. Both fields are optional
    because the user may mention only a name or only an email."""

    name: Optional[str] = None
    email: Optional[str] = None


class ExtractedIntent(BaseModel):
    """Internal model for raw LLM extraction output only.
    Never stored in session state or sent to Cal.com.
    Must be mapped to UserIntent via _map_extracted_to_intent() before use."""

    intent_type: IntentType
    attendees: list[ExtractedAttendee] = Field(default_factory=list)
    event_name: Optional[str] = None
    search_text: Optional[str] = None
    booking_uid: Optional[str] = None
    source_start_time: Optional[datetime] = None
    source_duration_minutes: Optional[int] = None
    target_start_time: Optional[datetime] = None
    target_duration_minutes: Optional[int] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    relative_time_qualifier: Optional[Literal["earlier", "mid", "later"]] = None
    timezone: Optional[str] = None

    @field_validator(
        "source_start_time",
        "target_start_time",
        "date_range_start",
        "date_range_end",
        mode="after",
    )
    @classmethod
    def must_be_tz_aware(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v)

    @model_validator(mode="after")
    def apply_defaults_and_validate(self) -> "ExtractedIntent":
        # date_range_start and date_range_end must both be set or both null
        if (self.date_range_start is None) != (self.date_range_end is None):
            raise ValueError(
                "date_range_start and date_range_end must both be set or both null"
            )
        # Default source_duration_minutes to 30 when source_start_time is present
        if self.source_start_time is not None and self.source_duration_minutes is None:
            self.source_duration_minutes = 30
        # Default target_duration_minutes when target_start_time is present
        if self.target_start_time is not None and self.target_duration_minutes is None:
            self.target_duration_minutes = self.source_duration_minutes or 30
        # Clamp durations to valid range
        if self.source_duration_minutes is not None and not (
            5 <= self.source_duration_minutes <= 480
        ):
            raise ValueError(
                f"source_duration_minutes {self.source_duration_minutes} out of range [5, 480]"
            )
        if self.target_duration_minutes is not None and not (
            5 <= self.target_duration_minutes <= 480
        ):
            raise ValueError(
                f"target_duration_minutes {self.target_duration_minutes} out of range [5, 480]"
            )
        return self


class Slot(BaseModel):
    start: datetime
    end: datetime

    @field_validator("start", "end", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v


class Booking(BaseModel):
    uid: str
    title: str
    start: datetime
    end: datetime
    attendees: list[Attendee] = Field(default_factory=list)
    status: str = "accepted"
    event_type_id: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("eventTypeId", "event_type_id")
    )

    @field_validator("start", "end", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v


class EventType(BaseModel):
    id: int
    title: str
    slug: str
    length_minutes: int = Field(
        validation_alias=AliasChoices("lengthInMinutes", "length_minutes")
    )
    length_minutes_options: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("lengthInMinutesOptions", "length_minutes_options"),
    )
    hidden: bool = False

    def supported_durations(self) -> list[int]:
        durations = set(self.length_minutes_options)
        durations.add(self.length_minutes)
        return sorted(durations)


class UserIntent(BaseModel):
    intent_type: IntentType
    search_text: Optional[str] = None
    attendee_name: Optional[str] = None
    attendee_email: Optional[str] = None
    # Multi-attendee list from structured extraction; attendee_name/email kept for backward compat
    attendees: list[ExtractedAttendee] = Field(default_factory=list)
    # Event/title keywords for title-only matching; kept separate from search_text fallback
    event_name: Optional[str] = None
    duration_minutes: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source_start_time: Optional[datetime] = None
    source_end_time: Optional[datetime] = None
    timezone: Optional[str] = None
    booking_uid: Optional[str] = None
    time_preference: Optional[str] = None
    # "date"=date only, "daypart"=date+daypart, "exact"=date+explicit time, "none"=no time info
    time_granularity: Optional[Literal["none", "date", "daypart", "exact"]] = None
    # day-level relative qualifier: "earlier" (8-11 AM), "mid" (11 AM-2 PM), "later" (2-6 PM)
    relative_time_qualifier: Optional[Literal["earlier", "mid", "later"]] = None

    @field_validator("start_time", "end_time", "source_start_time", "source_end_time", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v)


class BookingDraft(BaseModel):
    """Partial booking state accumulated across conversation turns."""

    attendee_name: Optional[str] = None
    attendee_email: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    timezone: Optional[str] = None
    event_type_id: Optional[int] = None
    include_length_in_minutes: bool = False
    time_preference: Optional[str] = None
    relative_time_qualifier: Optional[Literal["earlier", "mid", "later"]] = None

    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime | None) -> datetime | None:
        return _require_tz_aware(v)

    def is_ready(self) -> bool:
        return (
            self.attendee_name is not None
            and self.attendee_email is not None
            and (self.start_time is not None or self.time_preference is not None)
        )

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if self.attendee_name is None:
            missing.append("attendee_name")
        if self.attendee_email is None:
            missing.append("attendee_email")
        if self.start_time is None and self.time_preference is None:
            missing.append("time")
        return missing


class BookingRequest(BaseModel):
    attendee_name: str
    attendee_email: str
    start_time: datetime
    duration_minutes: int
    timezone: str
    event_type_id: int
    include_length_in_minutes: bool = False
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("start_time", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v


class CancelRequest(BaseModel):
    booking_uid: str


class RescheduleRequest(BaseModel):
    booking_uid: str
    new_start_time: datetime

    @field_validator("new_start_time", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v


class PendingAction(BaseModel):
    action_type: str  # "book" | "cancel" | "reschedule"
    booking_draft: Optional[BookingDraft] = None
    booking_request: Optional[BookingRequest] = None
    cancel_request: Optional[CancelRequest] = None
    reschedule_request: Optional[RescheduleRequest] = None
    selected_slot: Optional[Slot] = None
    matching_bookings: list[Booking] = Field(default_factory=list)
    waiting_for_field: Optional[str] = None  # "attendee_name" | "attendee_email"
    matching_bookings_are_partial: bool = False  # True when candidates came from Tier 4 or vague fallback


class CalClientError(Exception):
    """Raised for Cal.com API, HTTP, and network failures."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        reason: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        # "timeout" | "network" | "malformed" | "slot_unavailable" | "pagination_limit" | ""
        self.reason = reason


class AssistantError(Exception):
    """Raised for LLM API failures or unparseable LLM responses."""

    def __init__(self, message: str, reason: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason  # "llm_failure" | "bad_json" | "missing_field"


class IntentValidationError(Exception):
    """Raised for user-supplied data that fails semantic validation."""

    def __init__(self, message: str, reason: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.reason = reason  # "invalid_date" | "invalid_email" | "past_date" | "unsupported_action"
