# Cal.com Scheduling Assistant

A conversational web chat UI for managing your Cal.com calendar through natural language.

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
CAL_API_KEY=your_cal_api_key_here
CAL_API_BASE_URL=https://api.cal.com/v2
CAL_USERNAME=your_cal_username
CAL_TIMEZONE=America/New_York
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=your_openai_model_here
```

| Variable | Required | Description |
|---|---|---|
| `CAL_API_KEY` | Yes | Cal.com API key (v2) |
| `CAL_USERNAME` | Yes | Your Cal.com username |
| `CAL_API_BASE_URL` | No | Defaults to `https://api.cal.com/v2` |
| `CAL_TIMEZONE` | No | Defaults to `America/New_York` |
| `OPENAI_API_KEY` | Yes | OpenAI API key for intent extraction |
| `LLM_MODEL` | Yes | OpenAI model to use for intent extraction |

The app reads your Cal.com event types automatically. For booking, it picks the
matching type by duration, such as `15 min meeting` or `30 min meeting`. If the
duration is unclear, it asks which duration to use.

### 4. Run the app

```bash
python -m streamlit run app.py
```

The chat interface opens at `http://localhost:8501`.

## Cache and shortcuts

The app caches the Cal.com client while Streamlit is running. Restart Streamlit
after changing credentials or calendar configuration.

Keep copy on the standard `Ctrl+C` / `Cmd+C` shortcut. Do not bind clear cache
to that shortcut. If a keyboard shortcut is added for clear cache later, use
`Ctrl+Shift+K` on Windows/Linux and `Cmd+Shift+K` on macOS, and ignore the
shortcut while the user is typing in an input or editable field.

Streamlit's file-change and rerun controls remain available when needed. The
persistent top-right `Deploy` button is hidden, while the three-dot menu remains
available. The app labels rerun as `Ctrl+R` and clear cache away from bare `C`.

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
