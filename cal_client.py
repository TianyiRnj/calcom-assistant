from __future__ import annotations

import os
from datetime import datetime, timezone as _utc_zone
from typing import Optional

import httpx

from schemas import Booking, BookingRequest, CalClientError, EventType, Slot

_VERSION_EVENT_TYPES = "2024-06-14"
_VERSION_BOOKINGS_READ = "2026-05-01"
_VERSION_BOOKINGS_WRITE = "2026-02-25"
_VERSION_SLOTS = "2024-09-04"

_PAGE_LIMIT = 20

_SLOT_UNAVAILABLE_HINTS = ("slot", "unavailable", "no longer", "already booked")


def _message_from_error_value(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("message", "error", "details"):
            message = _message_from_error_value(value.get(key))
            if message:
                return message
        for nested in value.values():
            message = _message_from_error_value(nested)
            if message:
                return message
    if isinstance(value, list):
        parts = [_message_from_error_value(item) for item in value]
        return "; ".join(part for part in parts if part)
    return ""


def _response_error_message(response: httpx.Response) -> str:
    fallback = f"Cal.com API error: {response.status_code}"
    try:
        data = response.json()
    except Exception:
        return fallback

    if not isinstance(data, dict):
        return fallback

    for key in ("message", "error", "details"):
        message = _message_from_error_value(data.get(key))
        if message:
            return message
    return fallback


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise CalClientError("datetime must be timezone-aware", None)
    return dt.astimezone(_utc_zone.utc).isoformat()


def _parse_booking_data(data: dict) -> Booking:
    booking_data = data.get("data")
    if not isinstance(booking_data, dict):
        raise CalClientError("Malformed booking response: missing 'data'", None, reason="malformed")
    return Booking(**booking_data)


class CalClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        event_type_id: Optional[int] = None,
        username: str = "",
        timezone: str = "America/New_York",
        _client: Optional[httpx.Client] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.event_type_id = event_type_id
        self.username = username
        self.timezone = timezone
        self._client = _client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_bookings(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        status: Optional[str] = "upcoming",
    ) -> list[Booking]:
        params: dict = {}
        if status is not None:
            params["status"] = status
        if start is not None:
            params["afterStart"] = _to_utc_iso(start)
        if end is not None:
            params["beforeEnd"] = _to_utc_iso(end)

        all_bookings: list[Booking] = []
        for _ in range(_PAGE_LIMIT):
            data = self._get_raw("/bookings", params=params, version=_VERSION_BOOKINGS_READ)
            for b in data.get("data", []):
                try:
                    all_bookings.append(Booking(**b))
                except Exception as exc:
                    raise CalClientError(
                        f"Malformed booking data: {exc}", None, reason="malformed"
                    ) from exc
            pagination = data.get("pagination", {})
            if not pagination.get("hasMore"):
                break
            cursor = pagination.get("nextCursor")
            if not cursor:
                break
            params = {**params, "cursor": cursor}
        else:
            raise CalClientError(
                "Pagination safety limit reached; too many booking pages.",
                None,
                reason="pagination_limit",
            )

        return all_bookings

    def list_event_types(self) -> list[EventType]:
        params: dict = {"username": self.username, "sortCreatedAt": "desc"}
        return self._get(
            "/event-types",
            params=params,
            version=_VERSION_EVENT_TYPES,
            parse=lambda data: [EventType(**e) for e in data.get("data", [])],
        )

    def find_slots(
        self,
        start: datetime,
        end: datetime,
        duration_minutes: Optional[int] = None,
        timezone: Optional[str] = None,
        booking_uid_to_reschedule: Optional[str] = None,
        event_type_id: Optional[int] = None,
    ) -> list[Slot]:
        tz = timezone or self.timezone
        selected_event_type_id = event_type_id or self.event_type_id
        if selected_event_type_id is None:
            raise CalClientError(
                "No Cal.com event type selected for slot lookup.",
                None,
                reason="missing_event_type",
            )
        params: dict = {
            "eventTypeId": selected_event_type_id,
            "start": _to_utc_iso(start),
            "end": _to_utc_iso(end),
            "timeZone": tz,
            "format": "range",
        }
        if duration_minutes is not None:
            params["duration"] = duration_minutes
        if booking_uid_to_reschedule is not None:
            params["bookingUidToReschedule"] = booking_uid_to_reschedule

        def _parse(data: dict) -> list[Slot]:
            slots: list[Slot] = []
            data_val = data.get("data", {})
            if not isinstance(data_val, dict):
                raise CalClientError(
                    "Invalid slots response: 'data' is not a dict", None, reason="malformed"
                )
            for day_slots in data_val.values():
                for entry in day_slots:
                    try:
                        slots.append(
                            Slot(
                                start=datetime.fromisoformat(entry["start"]),
                                end=datetime.fromisoformat(entry["end"]),
                            )
                        )
                    except (KeyError, ValueError) as exc:
                        raise CalClientError(
                            f"Malformed slot entry: {exc}", None, reason="malformed"
                        ) from exc
            return slots

        return self._get("/slots", params=params, version=_VERSION_SLOTS, parse=_parse)

    def create_booking(self, request: BookingRequest) -> Booking:
        body = {
            "eventTypeId": request.event_type_id,
            "start": _to_utc_iso(request.start_time),
            "attendee": {
                "name": request.attendee_name,
                "email": request.attendee_email,
                "timeZone": request.timezone,
            },
            "metadata": {"externalRef": request.idempotency_key},
        }
        if getattr(request, "include_length_in_minutes", False):
            body["lengthInMinutes"] = request.duration_minutes
        return self._post(
            "/bookings",
            body=body,
            version=_VERSION_BOOKINGS_WRITE,
            parse=_parse_booking_data,
        )

    def cancel_booking(
        self,
        uid: str,
        cancellation_reason: str = "User requested cancellation",
    ) -> None:
        self._post(
            f"/bookings/{uid}/cancel",
            body={"cancellationReason": cancellation_reason},
            version=_VERSION_BOOKINGS_WRITE,
            parse=lambda _: None,
        )

    def reschedule_booking(self, uid: str, new_start: datetime) -> Booking:
        body = {"start": _to_utc_iso(new_start)}
        return self._post(
            f"/bookings/{uid}/reschedule",
            body=body,
            version=_VERSION_BOOKINGS_WRITE,
            parse=_parse_booking_data,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_data_status(self, data: dict) -> None:
        status = data.get("status")
        if status is not None and status != "success":
            msg = (
                data.get("error", {}).get("message")
                if isinstance(data.get("error"), dict)
                else data.get("message")
            ) or "Cal.com returned an error."
            reason = (
                "slot_unavailable"
                if any(h in msg.lower() for h in _SLOT_UNAVAILABLE_HINTS)
                else ""
            )
            raise CalClientError(msg, None, reason=reason)

    def _get_raw(self, path: str, params: dict, version: str) -> dict:
        try:
            response = self._client.get(
                path,
                params=params,
                headers={"cal-api-version": version},
            )
            self._check_response(response)
            data = response.json()
            self._check_data_status(data)
            return data
        except httpx.TimeoutException as exc:
            raise CalClientError("Request timed out. Please try again.", None, reason="timeout") from exc
        except httpx.RequestError as exc:
            raise CalClientError(str(exc), None, reason="network") from exc
        except CalClientError:
            raise
        except Exception as exc:
            raise CalClientError(str(exc), None, reason="malformed") from exc

    def _get(self, path: str, params: dict, version: str, parse):
        data = self._get_raw(path, params, version)
        try:
            return parse(data)
        except CalClientError:
            raise
        except Exception as exc:
            raise CalClientError(str(exc), None, reason="malformed") from exc

    def _post(self, path: str, body: dict, version: str, parse):
        try:
            response = self._client.post(
                path,
                json=body,
                headers={"cal-api-version": version},
            )
            self._check_response(response)
            data = response.json()
            self._check_data_status(data)
            return parse(data)
        except httpx.TimeoutException as exc:
            raise CalClientError("Request timed out. Please try again.", None, reason="timeout") from exc
        except httpx.RequestError as exc:
            raise CalClientError(str(exc), None, reason="network") from exc
        except CalClientError:
            raise
        except Exception as exc:
            raise CalClientError(str(exc), None, reason="malformed") from exc

    def _check_response(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise CalClientError(
                "API authentication failed. Check your CAL_API_KEY.", 401
            )
        if response.status_code == 429:
            raise CalClientError(
                "Cal.com rate limit reached. Please try again shortly.", 429
            )
        if not response.is_success:
            message = _response_error_message(response)
            reason = (
                "slot_unavailable"
                if any(h in message.lower() for h in _SLOT_UNAVAILABLE_HINTS)
                else ""
            )
            raise CalClientError(
                message,
                response.status_code,
                reason=reason,
            )


def build_from_env() -> CalClient:
    event_type_id_raw = os.environ.get("CAL_EVENT_TYPE_ID", "").strip()
    event_type_id = int(event_type_id_raw) if event_type_id_raw.isdigit() else None
    return CalClient(
        api_key=os.environ["CAL_API_KEY"],
        base_url=os.environ.get("CAL_API_BASE_URL", "https://api.cal.com/v2"),
        event_type_id=event_type_id,
        username=os.environ["CAL_USERNAME"],
        timezone=os.environ.get("CAL_TIMEZONE", "America/New_York"),
    )
