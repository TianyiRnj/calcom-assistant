from __future__ import annotations

import os
from datetime import datetime, timezone as _utc_zone
from typing import Optional

import httpx

from schemas import Booking, BookingRequest, CalClientError, Slot

_VERSION_BOOKINGS_READ = "2026-05-01"
_VERSION_BOOKINGS_WRITE = "2026-02-25"
_VERSION_SLOTS = "2024-09-04"

_PAGE_LIMIT = 20

_SLOT_UNAVAILABLE_HINTS = ("slot", "unavailable", "no longer", "already booked")


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
        event_type_id: int,
        username: str,
        timezone: str,
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
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_bookings(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        status: str = "upcoming",
    ) -> list[Booking]:
        params: dict = {"status": status}
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

    def find_slots(
        self,
        start: datetime,
        end: datetime,
        duration_minutes: Optional[int] = None,
        timezone: Optional[str] = None,
        booking_uid_to_reschedule: Optional[str] = None,
    ) -> list[Slot]:
        tz = timezone or self.timezone
        params: dict = {
            "eventTypeId": self.event_type_id,
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
            for day_slots in data.get("data", {}).values():
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
            "lengthInMinutes": request.duration_minutes,
            "attendee": {
                "name": request.attendee_name,
                "email": request.attendee_email,
                "timeZone": request.timezone,
            },
        }
        return self._post(
            "/bookings",
            body=body,
            version=_VERSION_BOOKINGS_WRITE,
            parse=_parse_booking_data,
        )

    def cancel_booking(self, uid: str) -> None:
        self._post(
            f"/bookings/{uid}/cancel",
            body={},
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
        return parse(self._get_raw(path, params, version))

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
            raise CalClientError(
                f"Cal.com API error: {response.status_code}", response.status_code
            )


def build_from_env() -> CalClient:
    return CalClient(
        api_key=os.environ["CAL_API_KEY"],
        base_url=os.environ.get("CAL_API_BASE_URL", "https://api.cal.com/v2"),
        event_type_id=int(os.environ["CAL_EVENT_TYPE_ID"]),
        username=os.environ["CAL_USERNAME"],
        timezone=os.environ.get("CAL_TIMEZONE", "America/New_York"),
    )
