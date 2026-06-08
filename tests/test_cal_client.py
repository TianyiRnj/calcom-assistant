"""Tests for CalClient — v2 API request construction and error handling."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from cal_client import CalClient, build_from_env
from schemas import AssistantError, Booking, BookingRequest, CalClientError, EventType, Slot

from tests.conftest import booking_dict, make_response

_T0 = datetime(2050, 9, 5, 14, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2050, 9, 5, 14, 30, 0, tzinfo=timezone.utc)


# ===========================================================================
# TestListBookings
# ===========================================================================


class TestListBookings:
    def test_list_bookings_bearer_token(self, cal_client: CalClient) -> None:
        """CalClient passes Authorization: Bearer <key> when constructing httpx.Client."""
        # The bearer token is baked into the httpx.Client headers at init time.
        # We verify it by patching httpx.Client and inspecting constructor args.
        with patch("cal_client.httpx.Client") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            mock_instance.get.return_value = make_response(
                200, {"status": "success", "data": []}
            )
            client = CalClient(
                api_key="secret-key",
                base_url="https://api.cal.com/v2",
                event_type_id=42,
                username="u",
                timezone="UTC",
            )
            client.list_bookings()
        # Check via the actual call args
        all_args = mock_cls.call_args
        # headers is a keyword arg
        kw_headers = all_args.kwargs.get("headers", {})
        assert kw_headers.get("Authorization") == "Bearer secret-key"

    def test_list_bookings_api_version(self, cal_client: CalClient, mock_http: MagicMock) -> None:
        """GET /bookings includes cal-api-version: 2026-05-01."""
        mock_http.get.return_value = make_response(200, {"status": "success", "data": []})
        cal_client.list_bookings()
        _, kwargs = mock_http.get.call_args
        assert kwargs["headers"]["cal-api-version"] == "2026-05-01"

    def test_list_bookings_status_upcoming_param(self, cal_client: CalClient, mock_http: MagicMock) -> None:
        """Default call sends status=upcoming."""
        mock_http.get.return_value = make_response(200, {"status": "success", "data": []})
        cal_client.list_bookings()
        _, kwargs = mock_http.get.call_args
        assert kwargs["params"]["status"] == "upcoming"

    def test_list_bookings_parses_data_array(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        """Response data array is parsed into list[Booking]."""
        mock_http.get.return_value = make_response(
            200, {"status": "success", "data": [booking_dict()]}
        )
        result = cal_client.list_bookings()
        assert len(result) == 1
        assert isinstance(result[0], Booking)
        assert result[0].uid == "uid-123"
        assert result[0].title == "Intro Call"


# ===========================================================================
# TestListEventTypes
# ===========================================================================


class TestListEventTypes:
    def test_list_event_types_uses_username_and_api_version(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(
            200,
            {
                "status": "success",
                "data": [
                    {
                        "id": 42,
                        "title": "30 min meeting",
                        "slug": "30min",
                        "lengthInMinutes": 30,
                    }
                ],
            },
        )

        event_types = cal_client.list_event_types()

        assert isinstance(event_types[0], EventType)
        assert event_types[0].id == 42
        _, kwargs = mock_http.get.call_args
        assert kwargs["headers"]["cal-api-version"] == "2024-06-14"
        assert kwargs["params"]["username"] == "testuser"


# ===========================================================================
# TestBuildFromEnv
# ===========================================================================


class TestBuildFromEnv:
    def test_invalid_legacy_event_type_id_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CAL_EVENT_TYPE_ID", "https://cal.com/user/30min")

        client = build_from_env()

        assert client.event_type_id is None


# ===========================================================================
# TestFindSlots
# ===========================================================================


class TestFindSlots:
    def _slot_response(self) -> dict:
        return {
            "status": "success",
            "data": {
                "2050-09-05": [
                    {"start": _T0.isoformat(), "end": _T1.isoformat()}
                ]
            },
        }

    def test_find_slots_api_version_2024_09_04(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1)
        _, kwargs = mock_http.get.call_args
        assert kwargs["headers"]["cal-api-version"] == "2024-09-04"

    def test_find_slots_sends_start_and_end_not_starttime_endtime(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1)
        _, kwargs = mock_http.get.call_args
        params = kwargs["params"]
        assert "start" in params
        assert "end" in params
        # Ensure v1-style camelCase param names are absent
        v1_keys = {"start" + "Time", "end" + "Time"}
        assert not v1_keys.intersection(params.keys())

    def test_find_slots_sends_timezone_and_format_range(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1)
        _, kwargs = mock_http.get.call_args
        params = kwargs["params"]
        assert params["timeZone"] == "America/New_York"
        assert params["format"] == "range"

    def test_find_slots_sends_duration_when_provided(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1, duration_minutes=30)
        _, kwargs = mock_http.get.call_args
        assert kwargs["params"]["duration"] == 30

    def test_find_slots_can_override_default_event_type_id(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1, event_type_id=99)
        _, kwargs = mock_http.get.call_args
        assert kwargs["params"]["eventTypeId"] == 99

    def test_find_slots_sends_booking_uid_to_reschedule_when_provided(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1, booking_uid_to_reschedule="uid-123")
        _, kwargs = mock_http.get.call_args
        assert kwargs["params"]["bookingUidToReschedule"] == "uid-123"

    def test_find_slots_omits_booking_uid_to_reschedule_for_normal_booking(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, self._slot_response())
        cal_client.find_slots(_T0, _T1)
        _, kwargs = mock_http.get.call_args
        assert "bookingUidToReschedule" not in kwargs["params"]


# ===========================================================================
# TestFindSlotsResponseParsing
# ===========================================================================


class TestFindSlotsResponseParsing:
    def test_find_slots_parses_date_keyed_response(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(
            200,
            {
                "status": "success",
                "data": {
                    "2050-09-05": [
                        {"start": _T0.isoformat(), "end": _T1.isoformat()}
                    ]
                },
            },
        )
        slots = cal_client.find_slots(_T0, _T1)
        assert len(slots) == 1
        assert isinstance(slots[0], Slot)

    def test_find_slots_empty_data_returns_empty_list(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(
            200, {"status": "success", "data": {}}
        )
        slots = cal_client.find_slots(_T0, _T1)
        assert slots == []

    def test_find_slots_malformed_entry_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(
            200,
            {
                "status": "success",
                "data": {"2050-09-05": [{"end": _T1.isoformat()}]},  # missing "start"
            },
        )
        with pytest.raises(CalClientError):
            cal_client.find_slots(_T0, _T1)


# ===========================================================================
# TestCreateBooking
# ===========================================================================


class TestCreateBooking:
    def _request(self) -> BookingRequest:
        return BookingRequest(
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=_T0,
            duration_minutes=30,
            timezone="America/New_York",
            event_type_id=42,
        )

    def test_create_booking_api_version_2026_02_25(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.create_booking(self._request())
        _, kwargs = mock_http.post.call_args
        assert kwargs["headers"]["cal-api-version"] == "2026-02-25"

    def test_create_booking_attendee_object_not_array(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        """v2 uses singular attendee object, not attendees array."""
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.create_booking(self._request())
        _, kwargs = mock_http.post.call_args
        body = kwargs["json"]
        assert "attendee" in body
        assert isinstance(body["attendee"], dict)
        assert "attendees" not in body
        assert body["attendee"]["name"] == "Jane"
        assert body["attendee"]["email"] == "jane@example.com"

    def test_create_booking_omits_length_in_minutes_by_default(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.create_booking(self._request())
        _, kwargs = mock_http.post.call_args
        assert "lengthInMinutes" not in kwargs["json"]

    def test_create_booking_sends_length_in_minutes_when_requested(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        request = self._request()
        request.include_length_in_minutes = True
        cal_client.create_booking(request)
        _, kwargs = mock_http.post.call_args
        assert kwargs["json"]["lengthInMinutes"] == 30

    def test_create_booking_sends_event_type_and_start(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.create_booking(self._request())
        _, kwargs = mock_http.post.call_args
        body = kwargs["json"]
        assert body["eventTypeId"] == 42
        assert "start" in body

    def test_create_booking_returns_booking(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        result = cal_client.create_booking(self._request())
        assert isinstance(result, Booking)
        assert result.uid == "uid-123"


# ===========================================================================
# TestCancelBooking
# ===========================================================================


class TestCancelBooking:
    def test_cancel_uses_post_not_delete(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {})
        cal_client.cancel_booking("uid-123")
        assert mock_http.post.called
        assert not mock_http.delete.called

    def test_cancel_endpoint_is_bookings_uid_cancel(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {})
        cal_client.cancel_booking("uid-123")
        args, _ = mock_http.post.call_args
        assert args[0] == "/bookings/uid-123/cancel"

    def test_cancel_api_version_2026_02_25(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {})
        cal_client.cancel_booking("uid-123")
        _, kwargs = mock_http.post.call_args
        assert kwargs["headers"]["cal-api-version"] == "2026-02-25"

    def test_cancel_sends_required_cancellation_reason(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {})
        cal_client.cancel_booking("uid-123")
        _, kwargs = mock_http.post.call_args
        assert kwargs["json"]["cancellationReason"] == "User requested cancellation"

    def test_cancel_can_override_cancellation_reason(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {})
        cal_client.cancel_booking("uid-123", cancellation_reason="Need to move it")
        _, kwargs = mock_http.post.call_args
        assert kwargs["json"]["cancellationReason"] == "Need to move it"


# ===========================================================================
# TestRescheduleBooking
# ===========================================================================


class TestRescheduleBooking:
    def test_reschedule_uses_post_not_patch(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.reschedule_booking("uid-123", _T1)
        assert mock_http.post.called
        assert not mock_http.patch.called

    def test_reschedule_endpoint_is_bookings_uid_reschedule(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.reschedule_booking("uid-123", _T1)
        args, _ = mock_http.post.call_args
        assert args[0] == "/bookings/uid-123/reschedule"

    def test_reschedule_sends_new_start_in_body(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.reschedule_booking("uid-123", _T1)
        _, kwargs = mock_http.post.call_args
        assert kwargs["json"]["start"] == _T1.isoformat()


# ===========================================================================
# TestErrorHandling
# ===========================================================================


class TestErrorHandling:
    def test_401_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(401)
        with pytest.raises(CalClientError) as exc_info:
            cal_client.list_bookings()
        assert exc_info.value.status_code == 401

    def test_429_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(429)
        with pytest.raises(CalClientError) as exc_info:
            cal_client.list_bookings()
        assert exc_info.value.status_code == 429

    def test_500_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(500)
        with pytest.raises(CalClientError) as exc_info:
            cal_client.list_bookings()
        assert exc_info.value.status_code == 500

    def test_timeout_raises_cal_client_error_not_httpx_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(CalClientError):
            cal_client.list_bookings()
        # Must NOT propagate raw httpx.TimeoutException
        try:
            cal_client.list_bookings()
        except CalClientError:
            pass
        except httpx.TimeoutException:
            pytest.fail("Raw httpx.TimeoutException leaked out of CalClient")

    def test_malformed_response_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        """Booking missing required uid field → CalClientError, not raw ValidationError."""
        mock_http.get.return_value = make_response(
            200,
            {"status": "success", "data": [{"title": "No UID here"}]},
        )
        with pytest.raises(CalClientError):
            cal_client.list_bookings()

    def test_cal_client_never_raises_assistant_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        """CalClient must not raise AssistantError — that's a separate error class."""
        mock_http.get.return_value = make_response(500)
        with pytest.raises(Exception) as exc_info:
            cal_client.list_bookings()
        assert not isinstance(exc_info.value, AssistantError)


# ===========================================================================
# TestStatusErrorHandling
# ===========================================================================


class TestStatusErrorHandling:
    def test_list_bookings_status_error_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(
            200, {"status": "error", "error": {"message": "Not found"}}
        )
        with pytest.raises(CalClientError):
            cal_client.list_bookings()

    def test_create_booking_status_failed_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(
            200, {"status": "failed", "message": "Bad request"}
        )
        request = BookingRequest(
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=_T0,
            duration_minutes=30,
            timezone="America/New_York",
            event_type_id=42,
        )
        with pytest.raises(CalClientError):
            cal_client.create_booking(request)

    def test_create_booking_http_400_preserves_response_message(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(
            400, {"error": {"message": "Email verification code is required"}}
        )
        request = BookingRequest(
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=_T0,
            duration_minutes=30,
            timezone="America/New_York",
            event_type_id=42,
        )

        with pytest.raises(CalClientError) as exc_info:
            cal_client.create_booking(request)

        assert exc_info.value.status_code == 400
        assert "Email verification code is required" in exc_info.value.message

    def test_find_slots_status_not_found_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, {"status": "not_found"})
        with pytest.raises(CalClientError):
            cal_client.find_slots(_T0, _T1)


# ===========================================================================
# TestUtcSerialization
# ===========================================================================


class TestUtcSerialization:
    from zoneinfo import ZoneInfo as _ZI

    _NY = _ZI("America/New_York")
    _T0_NY = datetime(2050, 9, 5, 10, 0, 0, tzinfo=_ZI("America/New_York"))

    def test_list_bookings_serializes_dates_to_utc(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(200, {"status": "success", "data": []})
        cal_client.list_bookings(start=self._T0_NY, end=self._T0_NY)
        _, kwargs = mock_http.get.call_args
        params = kwargs["params"]
        assert "+00:00" in params["afterStart"]
        assert "+00:00" in params["beforeEnd"]

    def test_find_slots_serializes_start_end_to_utc(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.get.return_value = make_response(
            200,
            {"status": "success", "data": {"2050-09-05": [{"start": _T0.isoformat(), "end": _T1.isoformat()}]}},
        )
        cal_client.find_slots(self._T0_NY, self._T0_NY)
        _, kwargs = mock_http.get.call_args
        params = kwargs["params"]
        assert "+00:00" in params["start"]
        assert "+00:00" in params["end"]

    def test_create_booking_serializes_start_to_utc(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        request = BookingRequest(
            attendee_name="Jane",
            attendee_email="jane@example.com",
            start_time=self._T0_NY,
            duration_minutes=30,
            timezone="America/New_York",
            event_type_id=42,
        )
        cal_client.create_booking(request)
        _, kwargs = mock_http.post.call_args
        assert "+00:00" in kwargs["json"]["start"]

    def test_reschedule_booking_serializes_new_start_to_utc(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        mock_http.post.return_value = make_response(200, {"status": "success", "data": booking_dict()})
        cal_client.reschedule_booking("uid-123", self._T0_NY)
        _, kwargs = mock_http.post.call_args
        assert "+00:00" in kwargs["json"]["start"]


# ===========================================================================
# TestNaiveDatetimeRejection
# ===========================================================================


class TestNaiveDatetimeRejection:
    def test_list_bookings_naive_start_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        naive_dt = datetime(2050, 9, 5, 14, 0, 0)
        with pytest.raises(CalClientError):
            cal_client.list_bookings(start=naive_dt)

    def test_find_slots_naive_start_raises_cal_client_error(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        naive_dt = datetime(2050, 9, 5, 14, 0, 0)
        with pytest.raises(CalClientError):
            cal_client.find_slots(naive_dt, _T1)


# ===========================================================================
# TestPagination
# ===========================================================================


class TestPagination:
    def _page(self, bookings: list, has_more: bool, cursor: str | None = None) -> dict:
        pagination: dict = {"hasMore": has_more}
        if cursor:
            pagination["nextCursor"] = cursor
        return {"status": "success", "data": bookings, "pagination": pagination}

    def test_list_bookings_follows_pagination_cursor(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        page1 = self._page([booking_dict(uid="uid-1")], has_more=True, cursor="c1")
        page2 = self._page([booking_dict(uid="uid-2")], has_more=False)
        mock_http.get.side_effect = [make_response(200, page1), make_response(200, page2)]
        cal_client.list_bookings()
        assert mock_http.get.call_count == 2
        # Second call must include the cursor
        _, kwargs2 = mock_http.get.call_args_list[1]
        assert kwargs2["params"]["cursor"] == "c1"

    def test_list_bookings_combines_results_across_pages(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        page1 = self._page([booking_dict(uid="uid-1")], has_more=True, cursor="c1")
        page2 = self._page([booking_dict(uid="uid-2", title="Second Call")], has_more=False)
        mock_http.get.side_effect = [make_response(200, page1), make_response(200, page2)]
        result = cal_client.list_bookings()
        assert len(result) == 2
        uids = {b.uid for b in result}
        assert uids == {"uid-1", "uid-2"}

    def test_list_bookings_preserves_status_param_across_pages(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        page1 = self._page([booking_dict(uid="uid-1")], has_more=True, cursor="c1")
        page2 = self._page([], has_more=False)
        mock_http.get.side_effect = [make_response(200, page1), make_response(200, page2)]
        cal_client.list_bookings(status="cancelled")
        for call_args in mock_http.get.call_args_list:
            _, kwargs = call_args
            assert kwargs["params"]["status"] == "cancelled"

    def test_list_bookings_raises_on_safety_limit(
        self, cal_client: CalClient, mock_http: MagicMock
    ) -> None:
        always_more = self._page([booking_dict()], has_more=True, cursor="cx")
        mock_http.get.side_effect = [make_response(200, always_more)] * 25
        with pytest.raises(CalClientError) as exc_info:
            cal_client.list_bookings()
        assert exc_info.value.reason == "pagination_limit"
