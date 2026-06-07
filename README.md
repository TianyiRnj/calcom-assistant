# Cal.com Scheduling Assistant

A conversational web chat UI for managing your Cal.com calendar through natural language.

## Setup

### 1. Clone and enter the directory

```bash
cd "LiveX AI"
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
CAL_API_KEY=your_cal_api_key_here
CAL_API_BASE_URL=https://api.cal.com/v2
CAL_EVENT_TYPE_ID=your_event_type_id
CAL_USERNAME=your_cal_username
CAL_TIMEZONE=America/New_York
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-5.4-nano
```

| Variable | Required | Description |
|---|---|---|
| `CAL_API_KEY` | Yes | Cal.com API key (v2) |
| `CAL_EVENT_TYPE_ID` | Yes | ID of the event type used for booking |
| `CAL_USERNAME` | Yes | Your Cal.com username |
| `CAL_API_BASE_URL` | No | Defaults to `https://api.cal.com/v2` |
| `CAL_TIMEZONE` | No | Defaults to `America/New_York` |
| `OPENAI_API_KEY` | Yes | OpenAI API key for intent extraction |
| `LLM_MODEL` | No | Defaults to `gpt-5.4-nano` |

### 5. Run the app

```bash
python -m streamlit run app.py
```

The chat interface opens at `http://localhost:8501`.

## Demo prompts

Once the app is running, try these in the chat:

- `What's on my calendar tomorrow?`
- `Book a 30-minute intro with Jane Thursday afternoon`
- `Cancel my call with Jane`
- `Move my 3pm to later today`

The assistant will ask follow-up questions for any missing details, show available slots as clickable buttons, and request confirmation before making any changes.

## Run tests

```bash
python -m pytest -q
```

Tests are fully mocked — no live API credentials required.
