# Chat Flow Reference

This file is a manual browser-test script and response reference for the
Cal.com Scheduling Assistant chat input.

Response examples are representative. Exact event titles, dates, timezones,
available slots, and Cal.com error details can vary by account data and current
date.

## Setup

Run the app:

```bash
python -m streamlit run app.py
```

Open the browser at:

```text
http://localhost:8501
```

Use the chat input with future dates for booking and rescheduling. For safe
manual tests, use a throwaway attendee such as:

```text
Taylor <taylor@livex.ai>
```

## Global Cases

| Case                               | User input                                                              | Expected response example                                                         |
| ---------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Welcome state                      | Load the app                                                            | `Hi! I can help you manage your Cal.com calendar. Try: ...`                       |
| Unsupported request                | `Send Taylor the notes`                                                 | `I can only help with scheduling — listing, booking, canceling, or rescheduling events.` |
| Yes with no pending action         | `yes`                                                                   | `What would you like to do? I can help with booking, canceling, or rescheduling.` |
| Ambiguous/invalid natural language | `asdfasdf`                                                              | `Sorry, I didn't catch that, could you rephrase?`                                 |
| LLM/API extraction failure         | Any scheduling request while LLM is unavailable                         | `I'm having trouble right now. Please try again.`                                 |
| Invalid date parse                 | `Book a call on February 30`                                            | `That date doesn't look right. What date did you mean?`                           |
| Past date                          | `Book a 30 minute call with Taylor at taylor@livex.ai yesterday at 2pm` | `I can only book future times.`                                                   |
| Invalid email                      | `Book a 30 minute call with Taylor at not-an-email tomorrow at 2pm`     | `That email doesn't look right. What's their email?`                              |
| Cal.com auth error                 | Any Cal.com-backed request with a bad API key                           | `There's an issue with the Cal.com API key. Please check your configuration.`     |
| Cal.com rate limit                 | Any Cal.com-backed request while rate limited                           | `Cal.com is busy right now. Please try again in a moment.`                        |
| Cal.com timeout/network            | Any Cal.com-backed request during timeout/network failure               | During confirmation: `Cal.com timed out. The action may or may not have gone through — please check your calendar before trying again.` For other operations: `Something went wrong with Cal.com. Please try again.` |
| Generic Cal.com failure            | Any Cal.com-backed request with an unexpected failure                   | `Something went wrong with Cal.com. Please try again.`                            |

## Seeing Scheduled Events

Use these to test listing calendar events.

| Case                           | Intended action                               | User input                                                      | Expected response example                                                                  |
| ------------------------------ | --------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| List tomorrow                  | Show only events scheduled for tomorrow.      | `What's on my calendar tomorrow?`                               | Tomorrow's matching events. Example: `Here's what's coming up:` followed by tomorrow events. |
| List next week                 | Show events scheduled during the next week.   | `What's on my calendar next week?`                              | Next week's matching events. Example: `Here's what's coming up:` followed by weekly events. |
| List specific date             | Show events scheduled on one requested date.  | `What's on my calendar on June 12?`                             | Events for June 12, if any.                                                                 |
| Empty calendar window          | Confirm no events exist in the requested window. | `What's on my calendar on January 1, 2050?`                     | `You have nothing scheduled for that period.`                                                |
| Listing during a booking draft | Switch intent from booking to calendar listing. | Start a booking, then type `What's on tomorrow?`                | Pending booking is cleared and tomorrow's events are listed.                                |
| Listing with auth issue        | Report that calendar lookup cannot authenticate. | `What's on my calendar tomorrow?` with invalid Cal.com API key  | `There's an issue with the Cal.com API key. Please check your configuration.`                |
| Listing rate limited           | Report that calendar lookup should be retried later. | `What's on my calendar tomorrow?` while Cal.com is rate limited | `Cal.com is busy right now. Please try again in a moment.`                                   |

## Booking Flow

### Happy Path

| Step | User input/action                                                      | Expected response example                                                |
| ---- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| 1    | `Book a 30 minute call with Taylor at taylor@livex.ai tomorrow at 2pm` | `Here are some available slots - pick one above or reply with a number.` |
| 2    | Click the first slot button, or type `1`                               | `You selected: 1. Jun 12 at 2:00 PM UTC` followed by `Book a 30-minute call with Taylor at this time?` |
| 3    | `yes`                                                                  | `Done! '30 min meeting between ...' is booked for Jun 12, 2:00 PM UTC.`  |

### Missing Field Cases

| Case                   | User input                                                                | Expected response example                                                                  |
| ---------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Missing name           | `Book a 30 minute call with taylor@livex.ai tomorrow at 2pm`              | `What's their name?`                                                                       |
| Missing email          | `Book a 30 minute call with Taylor tomorrow at 2pm`                       | `What's their email?`                                                                      |
| Missing name and email | `Book a 30 minute call tomorrow at 2pm`                                   | Asks for exactly one missing field first. Example: `What's their name?`                    |
| Email follow-up        | After missing email prompt, type `taylor@livex.ai`                        | Continues the draft. If time is concrete, shows slots.                                     |
| Name follow-up         | After missing name prompt, type `Taylor`                                  | Continues the draft. If enough fields are present, shows slots or asks next missing field. |
| Missing duration       | `Book a call with Taylor at taylor@livex.ai tomorrow at 2pm`              | `How long should it be - 15 or 30 minutes?`                                                |
| Duration follow-up     | After missing duration prompt, type `30`                                  | Continues booking and finds slots.                                                         |
| Vague time             | `Book a 30 minute call with Taylor at taylor@livex.ai sometime next week` | `What specific date and time works? For example, 'Thursday at 2pm'.`                       |
| Vague follow-up        | After vague time prompt, type `Thursday at 2pm`                           | Continues booking and finds slots.                                                         |

### Slot Selection Cases

| Case                           | User input/action              | Expected response example                                      |
| ------------------------------ | ------------------------------ | -------------------------------------------------------------- |
| Slot button                    | Click an available slot button | `You selected: <slot line>` followed by `Book a <N>-minute call with <name> at this time?` |
| Slot number                    | Type `1`                       | `You selected: <slot line>` followed by `Book a <N>-minute call with <name> at this time?` |
| Ordinal slot                   | Type `first`                   | Confirmation prompt for first slot.                            |
| Invalid slot input             | Type `banana`                  | `Please reply with a number between 1 and 5 to select a slot.` |
| Cancel while slots are showing | Type `cancel`                  | `Request cancelled. What would you like to do?`                |

### Confirmation Cases

| Case                        | User input | Expected response example             |
| --------------------------- | ---------- | ------------------------------------- |
| Confirm booking             | `yes`      | `Done! '...' is booked for ...`       |
| Decline booking             | `no`       | `Got it, no changes made.`            |
| Cancel word at confirmation | `cancel`   | `Got it, no changes made.`            |
| Invalid confirmation reply  | `maybe`    | `Confirm or decline above.`           |

### Booking Error Cases

| Case                                        | User input/state                                                       | Expected response example                                                      |
| ------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| No available slots                          | Request a time outside availability                                    | If nearby fallback slots exist: `The exact time wasn't available, but here are nearby options <label> — pick one above or reply with a number.` If no nearby slots at all: `Cal.com did not return bookable slots for that time. Availability rules, buffers, minimum notice, booking limits, or connected-calendar conflicts may be blocking it. Try a different time.` |
| Requested duration unavailable              | `Book a 45 minute call with Taylor at taylor@livex.ai tomorrow at 2pm` | `I don't see a 45-minute event type. Available options are ... minutes.`       |
| No Cal.com event types                      | Book when account has no event types                                   | `I couldn't find any Cal.com event types. Please create one in Cal.com first.` |
| Slot becomes unavailable after confirmation | Choose slot, then `yes`, while Cal.com rejects it                      | If nearby slots: `That slot is no longer available. Here are nearby options <label> — pick one above or reply with a number.` If none: `Cal.com did not return bookable slots for that time. Availability rules, buffers, minimum notice, booking limits, or connected-calendar conflicts may be blocking it. Try a different time.` |
| Cal.com rejects booking request             | Confirm booking with invalid Cal.com payload/server response           | `Cal.com rejected the booking request: ...`                                    |
| Timeout while confirming                    | Confirm booking during timeout                                         | `Cal.com timed out. The action may or may not have gone through — please check your calendar before trying again.` |
| Unexpected response while confirming        | Confirm booking when Cal.com returns malformed data                    | `Cal.com returned an unexpected response. Please check your calendar.`         |

## Cancel Flow

### Happy Path

| Step | User input/action            | Expected response example                                                      |
| ---- | ---------------------------- | ------------------------------------------------------------------------------ |
| 1    | `Cancel my call with Taylor` | `Cancel '30 min meeting between ...' on Jun 12 at 2:00 PM UTC?` |
| 2    | `yes`                        | `Booking cancelled.`                                                           |

### Cancel Cases

| Case                             | User input                                   | Expected response example                                                               |
| -------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| Decline cancel                   | After cancel confirmation, type `no`         | `Got it, no changes made.`                                                              |
| Cancel word at confirmation      | After cancel confirmation, type `cancel`     | `Got it, no changes made.`                                                              |
| Invalid confirmation reply       | After cancel confirmation, type `maybe`      | `Confirm to cancel this booking, or decline to keep it.`                                |
| No matching booking              | `Cancel my meeting with Unknown LiveX Guest` | `I couldn't find a matching booking.`                                                   |
| Multiple matches                 | `Cancel my call` when several calls match    | `I found 2 matching bookings. Which one do you mean?` followed by numbered bookings.    |
| Select multiple-match by number  | After multiple matches, type `1`             | Confirmation prompt for the selected booking.                                           |
| Select multiple-match by ordinal | After multiple matches, type `first`         | Confirmation prompt for the selected booking.                                           |
| Invalid multiple-match selection | After multiple matches, type `banana`        | Repeats the matches and adds: `Please reply with a number.`                             |
| Cancel during booking draft      | Start booking, then type `cancel`            | `Booking request cancelled. What would you like to do?` No calendar event is cancelled. |

## Reschedule Flow

### Happy Path

| Step | User input/action                        | Expected response example                                                  |
| ---- | ---------------------------------------- | -------------------------------------------------------------------------- |
| 1    | `Move my Taylor call to tomorrow at 3pm` | `Here are some available slots - pick one above or reply with a number.`   |
| 2    | Click a slot button, or type `1`         | `You selected: 1. Jun 12 at 3:00 PM UTC` followed by `Reschedule to this time?` |
| 3    | `yes`                                    | `Rescheduled! '30 min meeting between ...' is now at Jun 12, 3:00 PM UTC.` |

### Reschedule Cases

| Case                             | User input                                                       | Expected response example                                                                                                             |
| -------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Vague target time                | `Move my Taylor call to later today`                             | `What specific day and time would you like to reschedule to?` or `What specific date and time works? For example, 'Thursday at 2pm'.` |
| Vague target follow-up           | After vague prompt, type `tomorrow at 3pm`                       | Shows available slots or no-slots response.                                                                                           |
| No matching booking              | `Move my meeting with Unknown LiveX Guest to tomorrow at 3pm`    | `I couldn't find a matching booking. Could you provide more details?`                                                                 |
| Multiple matches                 | `Reschedule my call to tomorrow at 3pm` when several calls match | `I found 2 matching bookings. Which one do you mean?`                                                                                 |
| Select multiple-match by number  | After multiple matches, type `1`                                 | Slot search for the selected booking.                                                                                                 |
| Invalid multiple-match selection | After multiple matches, type `banana`                            | Repeats the matches and adds: `Please reply with a number.`                                                                           |
| No available slots               | `Move my Taylor call to tomorrow at 3am`                         | If nearby fallback slots exist: `The exact time wasn't available, but here are nearby options <label> — pick one above or reply with a number.` If no nearby slots at all: the no-availability explanation. |
| Retry after no slots             | After no-slots response, type `tomorrow at 2pm`                  | Reuses the same booking and searches slots again.                                                                                     |
| Cannot identify event type       | Reschedule a booking whose event type cannot be determined       | `I couldn't identify the event type for that booking.`                                                                                |

### Reschedule Slot And Confirmation Cases

| Case                        | User input/action              | Expected response example                                      |
| --------------------------- | ------------------------------ | -------------------------------------------------------------- |
| Slot button                 | Click an available slot button | `You selected: <slot line>` followed by `Reschedule to this time?` |
| Slot number                 | Type `1`                       | `You selected: <slot line>` followed by `Reschedule to this time?` |
| Invalid slot input          | Type `banana`                  | `Please reply with a number between 1 and 5 to select a slot.` |
| Confirm reschedule          | `yes`                          | `Rescheduled! '...' is now at ...`                             |
| Decline reschedule          | `no`                           | `Got it, no changes made.`                                     |
| Cancel word at confirmation | `cancel`                       | `Got it, no changes made.`                                     |
| Invalid confirmation reply  | `maybe`                        | `Confirm to reschedule, or decline to keep the original time.` |

## Browser Interaction Checklist

Use this checklist after running the chat flows:

| Check                                      | Expected result                                                    |
| ------------------------------------------ | ------------------------------------------------------------------ |
| Chat input accepts text                    | Typing in `What can I help you schedule?` enables the send button. |
| Enter submits                              | Pressing Enter submits the message.                                |
| Slot buttons render                        | Available slots appear as clickable buttons above the chat input.  |
| Slot number fallback works                 | Typing `1` selects the first slot when buttons are shown.          |
| Confirm and Decline buttons render         | During confirmation, both buttons appear and work.                 |
| No persistent sidebar controls             | The app should not show a left-side `Start over` button.           |
| Standard copy shortcut works               | Selecting text and pressing `Ctrl+C` or `Cmd+C` copies text.       |
| Persistent top-right controls are trimmed  | The app should hide `Deploy` and keep the three-dot menu visible.   |
| File-change controls still appear if needed | After a real file change, Streamlit can still show rerun controls. |

## Minimal Smoke Script

Run this short set when you only need a quick sanity check:

1. `yes`
   - Expect: `What would you like to do? ...`
2. `What's on my calendar tomorrow?`
   - Expect: list response or empty scheduled response.
3. `Book a 30 minute call with Taylor at taylor@livex.ai tomorrow at 2pm`
   - Expect: available slots or no-slots response.
4. If slots appear, type `banana`
   - Expect: `Please reply with a number between 1 and 5 to select a slot.`
5. Type `cancel`
   - Expect: pending flow clears.
6. `Cancel my meeting with Unknown LiveX Guest`
   - Expect: no matching booking response.
7. `Move my meeting with Unknown LiveX Guest to tomorrow at 3pm`
   - Expect: no matching booking response.
