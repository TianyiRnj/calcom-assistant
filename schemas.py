from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("start", "end", mode="after")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return v


class UserIntent(BaseModel):
    intent_type: IntentType
    search_text: Optional[str] = None
    attendee_name: Optional[str] = None
    attendee_email: Optional[str] = None
    duration_minutes: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timezone: Optional[str] = None
    booking_uid: Optional[str] = None
    time_preference: Optional[str] = None

    @field_validator("start_time", "end_time", mode="after")
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
    time_preference: Optional[str] = None

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
