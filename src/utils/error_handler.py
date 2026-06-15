"""
NexMatch AI — Centralized Error Handler (Phase 8.10)
Provides safe_run() wrapper and handle_exception() for friendly UI error display.
Logs all exceptions to audit_log.csv via audit_logger.
"""

import traceback
import streamlit as st
from src.utils.audit_logger import log_event


def handle_exception(
    exc: Exception,
    user_id: str = "SYSTEM",
    context: str = "",
    show_ui_error: bool = True,
) -> None:
    """
    Log exception to audit log and optionally display a friendly Streamlit error.

    Args:
        exc:           The caught exception.
        user_id:       Who was performing the action.
        context:       Brief description of what was being attempted.
        show_ui_error: If True, renders st.error() with friendly message.
    """
    tb = traceback.format_exc()
    detail = f"{context} | {type(exc).__name__}: {str(exc)}"
    log_event(user_id, "ERROR", detail[:500])

    if show_ui_error:
        st.error(
            f"Something went wrong{' while ' + context if context else ''}. "
            "Please try again or contact support."
        )


def safe_run(func, *args, user_id: str = "SYSTEM", context: str = "", **kwargs):
    """
    Execute func(*args, **kwargs) inside a try/except.
    On failure: logs to audit_log, shows friendly st.error(), returns None.

    Usage:
        result = safe_run(my_function, arg1, arg2, user_id="U001", context="loading recs")
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        handle_exception(exc, user_id=user_id, context=context, show_ui_error=True)
        return None
