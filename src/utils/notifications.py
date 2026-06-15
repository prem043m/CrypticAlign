import os
from datetime import datetime
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
NOTIFICATIONS_PATH = DATA_DIR / "pending_notifications.csv"

def _load_notifications_raw() -> pd.DataFrame:
    if NOTIFICATIONS_PATH.exists() and NOTIFICATIONS_PATH.stat().st_size > 0:
        return pd.read_csv(NOTIFICATIONS_PATH)
    columns = ["notification_id", "user_id", "matched_user_id", "type", "status", "created_at"]
    return pd.DataFrame(columns=columns)

def _save_notifications(df: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(NOTIFICATIONS_PATH, index=False)

def store_pending_notification(user_id: str, matched_user_id: str, notification_type: str = "mutual_match") -> str:
    """Stores a pending match notification in the CSV queue."""
    df = _load_notifications_raw()
    
    # Generate unique ID
    notification_id = f"N{len(df) + 1:04d}"
    
    new_notif = {
        "notification_id": notification_id,
        "user_id": user_id,
        "matched_user_id": matched_user_id,
        "type": notification_type,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    df = pd.concat([df, pd.DataFrame([new_notif])], ignore_index=True)
    _save_notifications(df)
    return notification_id

def get_pending_notifications() -> pd.DataFrame:
    """Returns all pending (unsent) notifications."""
    df = _load_notifications_raw()
    if df.empty:
        return df
    return df[df["status"] == "pending"].copy()

def mark_notifications_sent(notification_ids: list[str]) -> None:
    """Marks specified notification IDs as sent."""
    if not notification_ids:
        return
    df = _load_notifications_raw()
    if df.empty:
        return
    idx = df[df["notification_id"].isin(notification_ids)].index
    df.loc[idx, "status"] = "sent"
    _save_notifications(df)

def get_pending_notifications_count() -> int:
    """Returns the count of pending notifications."""
    df = _load_notifications_raw()
    if df.empty:
        return 0
    return len(df[df["status"] == "pending"])
