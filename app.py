from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from assistant import handle_message, _fmt_dt, _in_confirmation_phase
from cal_client import CalClient, build_from_env
from schemas import CalClientError, PendingAction, Slot

_REQUIRED_ENV_VARS = (
    "CAL_API_KEY",
    "CAL_USERNAME",
    "OPENAI_API_KEY",
    "LLM_MODEL",
)

_WELCOME = (
    "Hi! I can help you manage your Cal.com calendar. Try:\n"
    "- What's on my calendar tomorrow?\n"
    "- Book a 30-minute intro with Jane Thursday afternoon\n"
    "- Cancel my call with Jane\n"
    "- Move my 3pm to later today"
)

_STREAMLIT_CHROME_PATCH_HTML = """
<script>
(() => {
  const parentDoc = window.parent && window.parent.document;
  if (!parentDoc) return;

  const HELP_TEXT = [
    "The app file changed while Streamlit is running.",
    "Click Rerun to load the latest code, or Always rerun to apply future edits automatically.",
    "Restart Streamlit after changing .env values or Streamlit config."
  ].join(" ");
  const STYLE_ID = "cal-streamlit-chrome-patch-style";
  const TIP_ID = "cal-file-change-help";
  const CACHE_SHORTCUT_TEXT = /Mac|iPhone|iPad/.test(navigator.platform)
    ? "Cmd+Shift+K"
    : "Ctrl+Shift+K";
  const RERUN_SHORTCUT_TEXT = "Ctrl+R";

  function ensureStyle() {
    if (parentDoc.getElementById(STYLE_ID)) return;
    const style = parentDoc.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      header .cal-file-change-row {
        align-items: center !important;
        background: #fff4f3 !important;
        border: 1px solid #fecdca !important;
        border-radius: 999px !important;
        color: #b42318 !important;
        gap: 0.35rem !important;
        padding: 0.25rem 0.55rem !important;
      }

      header .cal-file-change-label {
        color: #b42318 !important;
        font-weight: 700 !important;
      }

      header .cal-file-change-icon {
        border-radius: 999px !important;
        color: #b42318 !important;
        cursor: help !important;
        outline: none !important;
        pointer-events: auto !important;
      }

      header .cal-file-change-icon:hover,
      header .cal-file-change-icon:focus {
        background: #fee4e2 !important;
        box-shadow: 0 0 0 3px rgba(217, 45, 32, 0.16) !important;
      }

      .cal-file-change-tip {
        background: #1f2937 !important;
        border-radius: 8px !important;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.25) !important;
        color: #ffffff !important;
        font: 500 13px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        max-width: 340px !important;
        padding: 0.7rem 0.8rem !important;
        position: fixed !important;
        z-index: 999999 !important;
      }

      .cal-cache-menu-item {
        border-radius: 6px !important;
      }

      .cal-menu-shortcut {
        background: #f2f4f7 !important;
        border: 1px solid #d0d5dd !important;
        border-radius: 4px !important;
        color: #344054 !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
        font-size: 0.72rem !important;
        padding: 0.05rem 0.35rem !important;
      }

      header .cal-hidden-toolbar-action {
        display: none !important;
      }
    `;
    parentDoc.head.appendChild(style);
  }

  function isEditable(target) {
    if (!target) return false;
    const element = target.nodeType === Node.ELEMENT_NODE ? target : target.parentElement;
    if (!element) return false;
    return Boolean(
      element.closest("input, textarea, select, [contenteditable='true'], [role='textbox']")
    );
  }

  function getTip() {
    let tip = parentDoc.getElementById(TIP_ID);
    if (!tip) {
      tip = parentDoc.createElement("div");
      tip.id = TIP_ID;
      tip.className = "cal-file-change-tip";
      tip.textContent = HELP_TEXT;
      tip.hidden = true;
      parentDoc.body.appendChild(tip);
    }
    return tip;
  }

  function showTip(target) {
    const tip = getTip();
    const rect = target.getBoundingClientRect();
    tip.hidden = false;
    const top = Math.min(rect.bottom + 10, window.innerHeight - tip.offsetHeight - 12);
    const left = Math.min(Math.max(12, rect.left - 10), window.innerWidth - 360);
    tip.style.top = `${Math.max(12, top)}px`;
    tip.style.left = `${left}px`;
  }

  function hideTip() {
    const tip = parentDoc.getElementById(TIP_ID);
    if (tip) tip.hidden = true;
  }

  function patchFileChangeNotice() {
    const labels = Array.from(parentDoc.querySelectorAll("header label"))
      .filter((label) => label.textContent.trim() === "File change.");

    labels.forEach((label) => {
      const row = label.closest("div");
      const icon = row && row.querySelector("span");

      label.classList.add("cal-file-change-label");
      if (row) row.classList.add("cal-file-change-row");
      if (!icon) return;

      icon.classList.add("cal-file-change-icon");
      icon.setAttribute("role", "button");
      icon.setAttribute("tabindex", "0");
      icon.setAttribute("aria-label", HELP_TEXT);
      icon.setAttribute("title", HELP_TEXT);

      if (icon.dataset.calFileChangePatched === "true") return;
      icon.dataset.calFileChangePatched = "true";
      icon.addEventListener("mouseenter", () => showTip(icon));
      icon.addEventListener("focus", () => showTip(icon));
      icon.addEventListener("mouseleave", hideTip);
      icon.addEventListener("blur", hideTip);
      icon.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        showTip(icon);
      });
      icon.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        showTip(icon);
      });
    });
  }

  function visibleMenuItemsStartingWith(prefix) {
    return Array.from(parentDoc.querySelectorAll("[role='menuitem'], div"))
      .filter((item) => {
        const text = item.textContent.replace(/\\s+/g, " ").trim();
        const rect = item.getBoundingClientRect();
        return text.startsWith(prefix) && rect.width > 0 && rect.height > 0;
      });
  }

  function findShortcutNode(item, text) {
    return Array.from(item.querySelectorAll("span, div, kbd"))
      .find((child) => child.textContent.trim() === text);
  }

  function patchRerunShortcutText() {
    visibleMenuItemsStartingWith("Rerun").forEach((item) => {
      item.setAttribute("title", `Rerun the app (${RERUN_SHORTCUT_TEXT}).`);
      const shortcut = findShortcutNode(item, "R");
      if (shortcut) {
        shortcut.textContent = RERUN_SHORTCUT_TEXT;
        shortcut.classList.add("cal-menu-shortcut");
      }
    });

    Array.from(parentDoc.querySelectorAll("header button"))
      .filter((button) => button.textContent.trim() === "Rerun")
      .forEach((button) => {
        button.setAttribute("title", `Rerun the app (${RERUN_SHORTCUT_TEXT}).`);
        button.setAttribute("aria-label", `Rerun the app (${RERUN_SHORTCUT_TEXT})`);
      });
  }

  function patchCacheShortcutText() {
    const items = Array.from(parentDoc.querySelectorAll("[role='menuitem'], div"))
      .filter((item) => {
        const text = item.textContent.replace(/\\s+/g, " ").trim();
        const rect = item.getBoundingClientRect();
        return text.startsWith("Clear cache") && rect.width > 0 && rect.height > 0;
      });

    items.forEach((item) => {
      item.classList.add("cal-cache-menu-item");
      item.setAttribute(
        "title",
        `Clear cache no longer uses C. Restart Streamlit after config changes; copy remains Ctrl/Cmd+C.`
      );
      const shortcut = findShortcutNode(item, "C");
      if (shortcut) {
        shortcut.textContent = CACHE_SHORTCUT_TEXT;
        shortcut.classList.add("cal-menu-shortcut");
      }
    });
  }

  function hidePersistentToolbarActions() {
    Array.from(parentDoc.querySelectorAll("header button")).forEach((button) => {
      const label = [
        button.getAttribute("aria-label"),
        button.getAttribute("title"),
        button.textContent
      ].filter(Boolean).join(" ").replace(/\\s+/g, " ").trim();

      if (label === "Deploy") {
        button.classList.add("cal-hidden-toolbar-action");
        button.setAttribute("aria-hidden", "true");
        button.setAttribute("tabindex", "-1");
      }
    });
  }

  function installShortcutGuard() {
    const parentWindow = parentDoc.defaultView;
    if (parentWindow.__calShortcutGuardInstalled) return;
    parentWindow.__calShortcutGuardInstalled = true;
    parentDoc.addEventListener(
      "keydown",
      (event) => {
        const key = event.key.toLowerCase();
        if ((event.ctrlKey || event.metaKey) && key === "c") return;
        if (event.ctrlKey && !event.metaKey && key === "r" && !isEditable(event.target)) {
          const rerunButton = Array.from(parentDoc.querySelectorAll("header button"))
            .find((button) => button.textContent.trim() === "Rerun");
          if (rerunButton) {
            event.preventDefault();
            event.stopImmediatePropagation();
            rerunButton.click();
          }
        }
        if (!event.ctrlKey && !event.metaKey && !event.altKey && key === "c" && !isEditable(event.target)) {
          event.stopImmediatePropagation();
        }
      },
      true
    );
  }

  function patch() {
    ensureStyle();
    patchFileChangeNotice();
    patchRerunShortcutText();
    patchCacheShortcutText();
    hidePersistentToolbarActions();
    installShortcutGuard();
  }

  patch();
  const observer = new MutationObserver(patch);
  observer.observe(parentDoc.body, { childList: true, subtree: true });
  setTimeout(patch, 250);
  setTimeout(patch, 1000);
})();
</script>
"""


@st.cache_resource
def _get_cal_client() -> CalClient:
    return build_from_env()


def _patch_streamlit_chrome() -> None:
    st.html(_STREAMLIT_CHROME_PATCH_HTML, unsafe_allow_javascript=True, width="content")


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": _WELCOME}]
    if "pending_action" not in st.session_state:
        st.session_state["pending_action"] = None
    if "available_slots" not in st.session_state:
        st.session_state["available_slots"] = []


def _missing_required_env() -> list[str]:
    return [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name, "").strip()]


def _dispatch_error_reply(exc: Exception) -> str:
    if isinstance(exc, CalClientError):
        if exc.status_code == 400:
            return f"Cal.com rejected the booking request: {exc.message}"
        if exc.status_code == 401:
            return "There's an issue with the Cal.com API key. Please check your configuration."
        if exc.status_code == 429:
            return "Cal.com is busy right now. Please try again in a moment."
        if exc.reason in ("timeout", "network"):
            return "Cal.com timed out or could not be reached. Please try again."
        return "Something went wrong with Cal.com. Please try again."
    return "Something went wrong while handling that request. Please try again."


def _dispatch(user_text: str, cal_client: CalClient) -> None:
    _pending_snap = st.session_state.get("pending_action")
    _slots_snap = st.session_state.get("available_slots", [])
    try:
        reply = handle_message(user_text, st.session_state, cal_client)
    except Exception as exc:
        st.session_state["pending_action"] = _pending_snap
        st.session_state["available_slots"] = _slots_snap
        reply = _dispatch_error_reply(exc)
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.session_state["messages"].append({"role": "assistant", "content": reply})


def main() -> None:
    st.set_page_config(page_title="Cal.com Assistant", page_icon="📅")
    _patch_streamlit_chrome()
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
            label = f"{i + 1}. {_fmt_dt(slot.start)} {tz}".strip()
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
