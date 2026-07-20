"""
nextmatchAi — Audit Logger (Phase 8.7)
Appends structured event records to data/audit_log.csv.

Supported event types:
  LOGIN, LOGOUT, REGISTER, PROFILE_UPDATE,
  REC_GENERATED, FEEDBACK_SUBMITTED,
  PASSWORD_RESET, ADMIN_RETRAIN,
  ACCOUNT_LOCKED, ACCOUNT_ENABLED, ACCOUNT_DISABLED,
  PASSWORD_RESET_REQUESTED, SESSION_EXPIRED, ERROR
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
AUDIT_LOG_PATH = DATA_DIR / "audit_log.csv"

AUDIT_COLUMNS = ["timestamp", "user_id", "event_type", "details"]


def _load_audit_log() -> pd.DataFrame:
    if AUDIT_LOG_PATH.exists() and AUDIT_LOG_PATH.stat().st_size > 0:
        try:
            return pd.read_csv(AUDIT_LOG_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=AUDIT_COLUMNS)


def log_event(user_id: str, event_type: str, details: str = "") -> None:
    """
    Append one audit record to data/audit_log.csv.

    Args:
        user_id:    The user identifier (or "SYSTEM" for system-level events).
        event_type: One of the defined event type constants above.
        details:    Free-form context string (truncated to 500 chars).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_id": str(user_id),
            "event_type": str(event_type).upper(),
            "details": str(details)[:500],
        }
        df = _load_audit_log()
        df = pd.concat([df, pd.DataFrame([row], columns=AUDIT_COLUMNS)], ignore_index=True)
        df.to_csv(AUDIT_LOG_PATH, index=False)
    except Exception as exc:
        # Audit logging must never crash the app
        print(f"[AUDIT_LOG_ERROR] {exc}")


def load_audit_log(
    event_type_filter: str = "All",
    user_id_filter: str = "",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Load audit log with optional filters for the admin viewer.

    Args:
        event_type_filter: "All" or a specific event_type string.
        user_id_filter:    Partial or full user_id to match.
        limit:             Max rows to return (most recent first).
    """
    df = _load_audit_log()
    if df.empty:
        return df

    # Sort newest first
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False)

    if event_type_filter != "All":
        df = df[df["event_type"] == event_type_filter.upper()]

    if user_id_filter.strip():
        df = df[df["user_id"].str.contains(user_id_filter.strip(), case=False, na=False)]

    return df.head(limit).reset_index(drop=True)


def get_audit_event_types() -> list:
    """Returns sorted list of distinct event_types in the log."""
    df = _load_audit_log()
    if df.empty:
        return []
    return sorted(df["event_type"].dropna().unique().tolist())
