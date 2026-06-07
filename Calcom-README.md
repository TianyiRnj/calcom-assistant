# Conversational scheduling assistant coding challenge

## Overview

Meet your user: a busy founder who lives in their inbox and runs their day out of cal.com. They don't want forms or
menus — they just want to type things like "book a 30-min intro with a candidate Thursday afternoon", "what's on my
calendar tomorrow?", or "move my 3pm to later today" and have it handled.

Your task is to build an interactive chatbot that lets this user manage their cal.com account through plain conversation.

How you build the conversational layer is up to you. You are free to use any LLM provider.

## Requirements

Build a simple chatbot that can interact with the cal.com REST API. Through the chat interface, the user should be able to:

 - Book a new event (the assistant gathers whatever details it needs, then creates the event).
 - See their scheduled events.
 - Cancel an event.
 - Reschedule an event they've booked.

It's a plus if the chatbot is usable through an interactive web UI.

Build the experience you'd want this user to have.

### Language

Please use Python for this code challenge.

## Cal.com API Reference

First, you'll need to create a cal.com account and obtain an API key. Follow the instructions in the
[authentication document](https://cal.com/docs/enterprise-features/api/authentication) to get started.

Second, here is the documentation for the cal.com [booking API](https://cal.com/docs/api-reference/v2/bookings/get-all-bookings) and [slot api](https://cal.com/docs/api-reference/v2/slots/find-out-when-is-an-event-type-ready-to-be-booked).
