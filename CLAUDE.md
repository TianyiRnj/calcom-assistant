# CLAUDE.md

## Project Overview

This project is a Python coding challenge for a conversational Cal.com scheduling assistant. The application should let a user manage their Cal.com calendar through natural language in an interactive web chat UI.

The minimum working prototype supports:

- Listing scheduled events.
- Booking a new event.
- Canceling an existing event.
- Rescheduling an existing event.
- Asking follow-up questions when required details are missing.
- Confirming before any action that changes the calendar.

## Tech Stack

Use Python as the primary language.

Recommended runtime and libraries:

- Python 3.11 or newer.
- `streamlit` for the interactive browser chat UI.
- `httpx` for HTTP requests to Cal.com.
- `pydantic` for request, response, intent, and action schemas.
- `python-dotenv` for local environment variable loading.
- `pytest` for tests.
- One LLM provider for natural-language intent extraction and assistant replies.

Do not introduce a database for the MVP. Use Streamlit session state for temporary chat state and pending actions.

## Expected Directory Structure

Keep the project small and easy to inspect.

```text
.
├── Calcom-README.md
├── CLAUDE.md
├── README.md
├── .env.example
├── app.py
├── assistant.py
├── cal_client.py
├── schemas.py
└── tests/
    ├── test_assistant.py
    ├── test_cal_client.py
    └── test_conversation_flows.py
```

If the implementation grows beyond the MVP, create a `src/` package only when it clearly improves organization.

## Module Responsibilities

### `app.py`

- Owns the Streamlit UI.
- Renders the chat interface.
- Stores chat messages and pending actions in `st.session_state`.
- Calls assistant orchestration code.
- Displays slot options and confirmation prompts.
- Does not contain direct Cal.com API request code.

### `assistant.py`

- Owns conversation orchestration.
- Converts user text into structured intents.
- Tracks missing fields.
- Decides whether to ask a follow-up question, fetch data, present options, or request confirmation.
- Produces concise assistant-facing messages.
- Does not directly render Streamlit components.

### `cal_client.py`

- Owns all Cal.com REST API calls.
- Uses bearer-token authentication.
- Exposes clear methods such as:
  - `list_bookings`
  - `find_slots`
  - `create_booking`
  - `cancel_booking`
  - `reschedule_booking`
- Converts API errors into project-specific exceptions or safe result objects.
- Does not contain conversational logic.

### `schemas.py`

- Defines Pydantic models and enums.
- Keeps all structured intent, action, booking, slot, and confirmation shapes in one place.
- Avoids untyped dictionaries where a stable schema is known.

### `tests/`

- Contains unit and flow tests.
- Mocks Cal.com API calls.
- Avoids requiring live API keys for normal test runs.

## Environment Variables

Use environment variables for credentials and local configuration.

```text
CAL_API_KEY=
CAL_API_BASE_URL=https://api.cal.com/v1
CAL_EVENT_TYPE_ID=
CAL_USERNAME=
CAL_TIMEZONE=America/New_York
LLM_API_KEY=
```

Rules:

- Never hard-code API keys.
- Never commit `.env`.
- Commit `.env.example` with empty placeholder values.
- Treat `CAL_TIMEZONE` as the default timezone unless the user explicitly asks for another timezone.

## Code Style

Follow these conventions:

- Use type hints for public functions.
- Keep functions small and purpose-specific.
- Prefer plain functions and small classes over heavy abstractions.
- Use Pydantic models for structured data crossing module boundaries.
- Use standard library `datetime`, `zoneinfo`, and ISO 8601 strings for time handling.
- Use timezone-aware datetimes only.
- Raise or return clear errors instead of letting raw API exceptions leak into the UI.
- Keep assistant messages concise and action-oriented.

Avoid:

- Global mutable state outside Streamlit session state.
- Direct API calls from UI code.
- Large prompt strings scattered across files.
- Unstructured dictionaries for core scheduling actions.
- Silent failure on calendar-changing actions.

## Naming Conventions

Use Python naming conventions consistently.

- Files and modules: `snake_case.py`
- Functions: `snake_case`
- Variables: `snake_case`
- Classes and Pydantic models: `PascalCase`
- Enum classes: `PascalCase`
- Enum values: lowercase strings
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_<module_name>.py`
- Test functions: `test_<behavior_being_verified>`

Suggested names:

- Intent enum: `IntentType`
- Intent model: `UserIntent`
- Booking request model: `BookingRequest`
- Cancel request model: `CancelRequest`
- Reschedule request model: `RescheduleRequest`
- Pending action model: `PendingAction`
- Cal.com wrapper class: `CalClient`
- Cal.com error class: `CalClientError`

## Conversation Rules

The assistant should support only scheduling-related actions in the MVP:

- `list`
- `book`
- `cancel`
- `reschedule`

For calendar-changing actions:

- Always confirm before calling Cal.com.
- If multiple bookings match, ask the user to choose one.
- If required fields are missing, ask a specific follow-up question.
- If a requested slot is unavailable, offer nearby alternatives.
- If the user says "cancel" during an in-progress booking or reschedule flow, cancel the pending chat action unless the user clearly means a calendar event.

Do not claim that a booking, cancellation, or reschedule succeeded until Cal.com returns success.

## Error Handling

Handle these cases explicitly:

- Invalid dates.
- Dates or times in the past.
- Missing attendee name.
- Missing attendee email.
- Invalid attendee email.
- No matching booking found.
- Multiple matching bookings found.
- Already canceled booking.
- Unsupported user action.
- No available slots.
- Selected slot becomes unavailable.
- Cal.com authentication error.
- Cal.com rate limit.
- Network timeout.
- Malformed or incomplete API response.

User-facing errors should be short, clear, and helpful.

## Testing Guidelines

Use `pytest`.

Required test areas:

- Assistant intent extraction.
- Missing-field follow-up behavior.
- Cal.com client request construction.
- API error handling.
- Booking confirmation flow.
- Cancellation confirmation flow.
- Reschedule confirmation flow.
- Invalid action handling.
- Corner cases such as ambiguous booking matches and stale slot selection.

Detailed assistant intent tests:

- User asks "What's on my calendar tomorrow?" and the assistant returns a `list` intent with a concrete date range.
- User asks "Book a 30-minute intro with Jane Thursday afternoon" and the assistant returns a `book` intent with duration, attendee name, and time preference.
- User asks "Cancel my call with Jane" and the assistant returns a `cancel` intent with attendee or title search text.
- User asks "Move my 3pm to later today" and the assistant returns a `reschedule` intent with original booking search details and new time preference.
- User gives an incomplete booking request without a name and the assistant asks for the missing attendee name.
- User gives an incomplete booking request without an email and the assistant asks for the missing attendee email.
- User gives an ambiguous cancellation request with multiple possible matches and the assistant asks the user to choose one.

Detailed Cal.com client tests:

- `list_bookings` sends the correct bearer-token header and query parameters.
- `find_slots` sends event type, start, end, timezone, duration, and format parameters.
- `create_booking` sends attendee details and selected start time in the request body.
- `cancel_booking` calls the cancellation endpoint with the selected booking UID.
- `reschedule_booking` calls the reschedule endpoint with the selected booking UID and replacement start time.
- API errors are converted into clear assistant-facing error messages instead of crashing the app.

Detailed conversation flow tests:

- Listing flow returns a readable agenda when upcoming bookings exist.
- Listing flow returns a friendly empty-state message when no bookings exist.
- Booking flow collects missing fields, shows available slots, waits for confirmation, then creates the booking.
- Booking flow does not create a booking when the user declines confirmation.
- Cancel flow finds a matching booking, asks for confirmation, then cancels it.
- Cancel flow does not cancel anything when the user declines confirmation.
- Reschedule flow finds the original booking, shows replacement slots, waits for confirmation, then reschedules it.
- Reschedule flow handles "no available slots" by asking for a different time window.

Invalid action tests:

- User asks to book on an invalid date, such as "February 30", and the assistant asks for a valid date.
- User asks to book with an unclear relative date, such as "next Friday", and the assistant resolves or confirms the exact date before checking slots.
- User asks to book in the past and the assistant explains that it can only book future times.
- User asks to book outside available hours and the assistant offers the nearest available slots.
- User asks to cancel an event that does not exist and the assistant says no matching booking was found.
- User asks to reschedule an event that does not exist and the assistant asks for more identifying details.
- User asks to cancel or reschedule a booking that is already canceled and the assistant does not call the API again.
- User confirms a booking after the selected slot has become unavailable and the assistant asks the user to choose a new slot.
- User provides an invalid attendee email and the assistant asks for a corrected email address.
- User asks for an unsupported action, such as "send Jane the notes", and the assistant explains that it can only manage scheduling actions.

Corner case tests:

- Multiple bookings match a cancellation request and the assistant asks the user to pick the exact booking.
- Multiple bookings match a reschedule request and the assistant asks the user to pick the exact booking.
- Two attendees have the same first name and the assistant uses email, title, or time to disambiguate.
- User changes their mind mid-flow, such as starting a booking and then asking to list tomorrow's events.
- User says "yes" or "confirm" when there is no pending action and the assistant asks what they want to do.
- User says "cancel" during a booking flow and the assistant cancels the pending chat action, not a calendar event.
- User requests a duration that is not supported by the event type and the assistant falls back to supported durations or asks the user to choose.
- User requests a timezone different from the default timezone and the assistant displays the final selected time clearly.
- Cal.com returns a rate-limit response and the assistant shows a retry-friendly error.
- Cal.com returns an authentication error and the assistant reports that the API key or configuration needs attention.
- Cal.com returns malformed or incomplete data and the assistant shows a safe error instead of crashing.
- Network timeout occurs during a booking, cancellation, or reschedule request and the assistant avoids claiming success.
- Booking creation succeeds but the response is missing optional fields and the assistant still summarizes the confirmed start time.

Mock external services:

- Mock Cal.com API responses.
- Mock LLM responses for deterministic tests.
- Do not require real credentials for automated tests.

## UI Guidelines

The UI should be simple and usable.

- Use a chat-style Streamlit interface.
- Keep chat history visible.
- Show available slots as clear selectable options.
- Make confirmation prompts obvious.
- Keep success and failure messages brief.
- Do not build a marketing landing page for the MVP.
- Match the persona: a busy founder who wants to type naturally, not fill forms. Replies should be short and direct. Ask for one missing piece of information at a time. Prefer "What's their email?" over "Please provide the attendee email address."

## Implementation Priorities

Build in this order:

1. Define schemas.
2. Implement Cal.com client wrapper.
3. Implement assistant intent and conversation orchestration.
4. Implement Streamlit chat UI.
5. Add mocked unit tests.
6. Add manual demo instructions in `README.md`.

## Out of Scope

Do not spend MVP time on:

- OAuth.
- Multi-user accounts.
- Persistent database storage.
- Webhooks.
- Recurring event management.
- Team scheduling rules.
- Production deployment.
- Complex analytics.
- Advanced UI polish.

## Quality Bar

The MVP is acceptable when a reviewer can use the web chat to:

- Ask what is scheduled.
- Book an event.
- Reschedule an event.
- Cancel an event.
- See useful follow-up questions when details are missing.
- See safe behavior for invalid or ambiguous requests.
