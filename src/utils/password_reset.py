import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from src.utils.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESET_TOKENS_PATH = DATA_DIR / "password_reset_tokens.csv"

def _load_tokens_raw() -> pd.DataFrame:
    if RESET_TOKENS_PATH.exists() and RESET_TOKENS_PATH.stat().st_size > 0:
        return pd.read_csv(RESET_TOKENS_PATH)
    columns = ["token", "email", "created_at", "expiry_time"]
    return pd.DataFrame(columns=columns)

def _save_tokens(df: pd.DataFrame):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESET_TOKENS_PATH, index=False)

def generate_reset_token(email: str) -> str:
    """Generates a secure password reset token and saves it to CSV with expiration."""
    # Generate URL-safe random token
    token = secrets.token_urlsafe(16)
    
    # Calculate timestamps
    now = datetime.now()
    expiry = now + timedelta(minutes=Config.PASSWORD_RESET_EXPIRY_MINUTES)
    
    df = _load_tokens_raw()
    
    # Remove any existing tokens for this email first
    df = df[df["email"].str.lower() != email.lower()]
    
    new_token_row = {
        "token": token,
        "email": email.strip(),
        "created_at": now.isoformat(),
        "expiry_time": expiry.isoformat()
    }
    
    df = pd.concat([df, pd.DataFrame([new_token_row])], ignore_index=True)
    _save_tokens(df)
    
    return token

def validate_reset_token(token: str) -> tuple[bool, str, str]:
    """
    Validates token exists and has not expired.
    Returns (is_valid, email, message)
    """
    token = token.strip()
    if not token:
        return False, "", "Token cannot be empty."
        
    df = _load_tokens_raw()
    match = df[df["token"] == token]
    if match.empty:
        return False, "", "Invalid or expired password reset token."
        
    row = match.iloc[0]
    email = str(row["email"])
    expiry_time_str = str(row["expiry_time"])
    
    try:
        expiry = datetime.fromisoformat(expiry_time_str)
    except Exception:
        return False, "", "Token timestamp parsing failed."
        
    if datetime.now() > expiry:
        # Clean up expired token
        consume_reset_token(token)
        return False, "", "Password reset token has expired."
        
    return True, email, "Token is valid."

def consume_reset_token(token: str) -> None:
    """Deletes token from the storage after validation or expiration."""
    df = _load_tokens_raw()
    df = df[df["token"] != token]
    _save_tokens(df)

def get_pending_resets_count() -> int:
    """Returns the count of active (unexpired) reset tokens."""
    df = _load_tokens_raw()
    if df.empty:
        return 0
    now = datetime.now()
    try:
        df["expiry"] = pd.to_datetime(df["expiry_time"])
        active_df = df[df["expiry"] > now]
        return len(active_df)
    except Exception:
        return len(df)
