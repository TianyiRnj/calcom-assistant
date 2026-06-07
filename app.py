from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from assistant import handle_message
from cal_client import CalClient, build_from_env
from schemas import PendingAction, Slot

_REQUIRED_ENV_VARS = (
    "CAL_API_KEY",
    "CAL_EVENT_TYPE_ID",
    "CAL_USERNAME",
    "OPENAI_API_KEY",
)

_WELCOME = (
    "Hi! I can help you manage your Cal.com calendar. Try:\n"
    "- What's on my calendar tomorrow?\n"
    "- Book a 30-minute intro with Jane Thursday afternoon\n"
    "- Cancel my call with Jane\n"
    "- Move my 3pm to later today"
)


@st.cache_resource
def _get_cal_client() -> CalClient:
    return build_from_env()


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": _WELCOME}]
    if "pending_action" not in st.session_state:
        st.session_state["pending_action"] = None
    if "available_slots" not in st.session_state:
        st.session_state["available_slots"] = []


def _in_confirmation_phase(pending: PendingAction | None) -> bool:
    if pending is None:
        return False
    return (
        pending.booking_request is not None
        or pending.cancel_request is not None
        or pending.reschedule_request is not None
    )


def _missing_required_env() -> list[str]:
    return [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name, "").strip()]


def _dispatch(user_text: str, cal_client: CalClient) -> None:
    reply = handle_message(user_text, st.session_state, cal_client)
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.session_state["messages"].append({"role": "assistant", "content": reply})


def main() -> None:
    st.set_page_config(page_title="Cal.com Assistant", page_icon="📅")
    st.title("Cal.com Scheduling Assistant")

    missing_env = _missing_required_env()
    if missing_env:
        st.error(f"Missing required environment variable: {', '.join(missing_env)}")
        st.info("Copy .env.example to .env, fill in your credentials, and restart the app.")
        return

    try:
        cal_client = _get_cal_client()
    except KeyError as exc:
        st.error(f"Missing required environment variable: {exc.args[0]}")
        st.info("Copy .env.example to .env, fill in your credentials, and restart the app.")
        return
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.info("Check your .env file and restart the app.")
        return

    _init_session()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    slots: list[Slot] = st.session_state.get("available_slots", [])
    pending: PendingAction | None = st.session_state.get("pending_action")

    if slots:
        st.markdown("**Available slots — click to select:**")
        for i, slot in enumerate(slots[:5]):
            tz = slot.start.strftime("%Z") or ""
            label = f"{i + 1}. {slot.start.strftime('%b %-d, %I:%M %p')} {tz}".strip()
            if st.button(label, key=f"slot_{i}"):
                _dispatch(str(i + 1), cal_client)
                st.rerun()

    elif _in_confirmation_phase(pending):
        col1, col2 = st.columns(2)
        if col1.button("Confirm", type="primary", key="btn_confirm"):
            _dispatch("yes", cal_client)
            st.rerun()
        if col2.button("Decline", type="secondary", key="btn_decline"):
            _dispatch("no", cal_client)
            st.rerun()

    if prompt := st.chat_input("What can I help you schedule?"):
        _dispatch(prompt, cal_client)
        st.rerun()


if __name__ == "__main__":
    main()
