import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
USERS_PATH = DATA_DIR / "users.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.csv"
RECOMMENDATION_HISTORY_PATH = DATA_DIR / "recommendation_history.csv"
CREDENTIALS_PATH = DATA_DIR / "credentials.csv"

USER_COLUMNS = [
    "user_id",
    "name",
    "age",
    "location",
    "profession",
    "experience_years",
    "education",
    "skills",
    "mbti",
    "traits",
    "career_goal",
    "networking_intent",
    "interests",
    "professional_summary",
    "about_me",
]

MBTI_OPTIONS = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

NETWORKING_INTENT_OPTIONS = [
    "Find Mentor",
    "Find Mentee",
    "Career Growth",
    "Startup Partner",
    "Professional Networking",
    "Research Collaboration",
    "Team Building",
    "Knowledge Sharing",
]

# --- Password Hashing & Security Helpers ---

def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
    return hashed, salt

def verify_password(password: str, salt: str, hashed: str) -> bool:
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest() == hashed

def load_credentials_raw() -> pd.DataFrame:
    if CREDENTIALS_PATH.exists() and CREDENTIALS_PATH.stat().st_size > 0:
        return pd.read_csv(CREDENTIALS_PATH)
    columns = ["username", "password_hash", "salt", "user_id", "role"]
    return pd.DataFrame(columns=columns)

def init_credentials():
    """Initializes credentials file with a default admin account if not existing."""
    if not CREDENTIALS_PATH.exists():
        columns = ["username", "password_hash", "salt", "user_id", "role"]
        h, salt = hash_password("admin")
        df = pd.DataFrame([{
            "username": "admin",
            "password_hash": h,
            "salt": salt,
            "user_id": "ADMIN",
            "role": "admin"
        }], columns=columns)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CREDENTIALS_PATH, index=False)

def authenticate_user(username: str, password: str) -> Tuple[bool, str, str, str]:
    """
    Returns (is_authenticated, user_id, role, message).
    """
    init_credentials()
    df = load_credentials_raw()
    match = df[df["username"].str.lower() == username.lower()]
    if match.empty:
        return False, "", "", "Username not found."
    
    row = match.iloc[0]
    if verify_password(password, str(row["salt"]), str(row["password_hash"])):
        return True, str(row["user_id"]), str(row["role"]), "Success"
    return False, "", "", "Invalid password."

def register_credentials(username: str, password: str, user_id: str, role: str = "user") -> Tuple[bool, str]:
    init_credentials()
    df = load_credentials_raw()
    if not df.empty and username.lower() in df["username"].str.lower().tolist():
        return False, "Username already exists."
    
    h, salt = hash_password(password)
    new_cred = {
        "username": username,
        "password_hash": h,
        "salt": salt,
        "user_id": user_id,
        "role": role
    }
    df = pd.concat([df, pd.DataFrame([new_cred])], ignore_index=True)
    df.to_csv(CREDENTIALS_PATH, index=False)
    return True, "Registered successfully."

# --- User Data & Profile Loading Helpers ---

def load_users_raw() -> pd.DataFrame:
    return pd.read_csv(USERS_PATH)

def load_feedback_raw() -> pd.DataFrame:
    return pd.read_csv(FEEDBACK_PATH)

def get_feedback_for_user(user_id: str) -> pd.DataFrame:
    feedback_df = load_feedback_raw()
    return feedback_df[feedback_df["user_id"] == user_id].copy()

def generate_next_user_id(users_df: pd.DataFrame) -> str:
    numeric_ids = users_df["user_id"].astype(str).str.replace("U", "", regex=False).astype(int)
    return f"U{numeric_ids.max() + 1:03d}"

def generate_next_user_id_combined() -> str:
    users_df = load_users_raw()
    ids = users_df["user_id"].astype(str).str.replace("U", "", regex=False).astype(int).tolist()
    
    if USER_PROFILES_PATH.exists() and USER_PROFILES_PATH.stat().st_size > 0:
        try:
            profiles_df = pd.read_csv(USER_PROFILES_PATH)
            if not profiles_df.empty:
                profile_ids = profiles_df["user_id"].astype(str).str.replace("U", "", regex=False).astype(int).tolist()
                ids.extend(profile_ids)
        except Exception:
            pass
    return f"U{max(ids) + 1:03d}"

def load_user_profiles_raw() -> pd.DataFrame:
    if USER_PROFILES_PATH.exists() and USER_PROFILES_PATH.stat().st_size > 0:
        return pd.read_csv(USER_PROFILES_PATH)
    columns = [
        "user_id", "name", "profession", "location", "experience_years", "mbti", 
        "career_goal", "skills", "interests", "networking_intent", 
        "professional_summary", "about_me", "created_at"
    ]
    return pd.DataFrame(columns=columns)

def register_user_profile(profile: dict) -> str:
    new_user_id = generate_next_user_id_combined()
    
    new_profile = {
        "user_id": new_user_id,
        "name": profile.get("name", "").strip(),
        "profession": profile.get("profession", "").strip(),
        "location": profile.get("location", "").strip(),
        "experience_years": int(profile.get("experience_years", 0)),
        "mbti": profile.get("mbti", "INTJ"),
        "career_goal": profile.get("career_goal", "").strip(),
        "skills": profile.get("skills", "").strip(),
        "interests": profile.get("interests", "").strip(),
        "networking_intent": profile.get("networking_intent", "Career Growth"),
        "professional_summary": profile.get("professional_summary", "").strip(),
        "about_me": profile.get("about_me", "").strip(),
        "created_at": datetime.now().isoformat()
    }
    
    columns = [
        "user_id", "name", "profession", "location", "experience_years", "mbti", 
        "career_goal", "skills", "interests", "networking_intent", 
        "professional_summary", "about_me", "created_at"
    ]
    
    df = load_user_profiles_raw()
    df = pd.concat([df, pd.DataFrame([new_profile], columns=columns)], ignore_index=True)
    df.to_csv(USER_PROFILES_PATH, index=False)
    return new_user_id

def update_user_profile(user_id: str, profile: dict) -> None:
    if not USER_PROFILES_PATH.exists():
        return
    
    df = pd.read_csv(USER_PROFILES_PATH)
    idx = df[df["user_id"] == user_id].index
    if len(idx) > 0:
        df.loc[idx[0], "name"] = profile.get("name", "").strip()
        df.loc[idx[0], "profession"] = profile.get("profession", "").strip()
        df.loc[idx[0], "location"] = profile.get("location", "").strip()
        df.loc[idx[0], "experience_years"] = int(profile.get("experience_years", 0))
        df.loc[idx[0], "mbti"] = profile.get("mbti", "INTJ")
        df.loc[idx[0], "career_goal"] = profile.get("career_goal", "").strip()
        df.loc[idx[0], "skills"] = profile.get("skills", "").strip()
        df.loc[idx[0], "interests"] = profile.get("interests", "").strip()
        df.loc[idx[0], "networking_intent"] = profile.get("networking_intent", "Career Growth")
        df.loc[idx[0], "professional_summary"] = profile.get("professional_summary", "").strip()
        df.loc[idx[0], "about_me"] = profile.get("about_me", "").strip()
        df.to_csv(USER_PROFILES_PATH, index=False)

# --- Recommendation History Helpers ---

def load_recommendation_history() -> pd.DataFrame:
    if RECOMMENDATION_HISTORY_PATH.exists() and RECOMMENDATION_HISTORY_PATH.stat().st_size > 0:
        return pd.read_csv(RECOMMENDATION_HISTORY_PATH)
    columns = ["user_id", "recommendation_batch_id", "recommended_user_id", "score", "timestamp"]
    return pd.DataFrame(columns=columns)

def append_recommendation_history_batch(user_id: str, batch_id: str, recs: List[Dict], timestamp: str) -> None:
    columns = ["user_id", "recommendation_batch_id", "recommended_user_id", "score", "timestamp"]
    new_records = []
    for rec in recs:
        new_records.append({
            "user_id": user_id,
            "recommendation_batch_id": batch_id,
            "recommended_user_id": rec["user_id"],
            "score": float(rec.get("final_ranking_score", rec.get("ml_score", 0.0))),
            "timestamp": timestamp
        })
    df = load_recommendation_history()
    df = pd.concat([df, pd.DataFrame(new_records, columns=columns)], ignore_index=True)
    df.to_csv(RECOMMENDATION_HISTORY_PATH, index=False)

# --- Legacy Feedback & Base Profile Helpers (Do not modify logic) ---

def append_user_profile(profile: Dict) -> str:
    users_df = load_users_raw()
    new_user_id = generate_next_user_id(users_df)

    new_user = {"user_id": new_user_id}
    for column in USER_COLUMNS[1:]:
        new_user[column] = profile.get(column, "")

    updated_users = pd.concat(
        [users_df, pd.DataFrame([new_user], columns=USER_COLUMNS)],
        ignore_index=True,
    )
    updated_users.to_csv(USERS_PATH, index=False)
    return new_user_id

def append_feedback(user_id: str, matched_user_id: str, action: int) -> None:
    feedback_df = load_feedback_raw()
    feedback_df.loc[len(feedback_df)] = [
        user_id,
        matched_user_id,
        int(action),
        datetime.now().isoformat(timespec="seconds"),
    ]
    feedback_df.to_csv(FEEDBACK_PATH, index=False)

def append_feedback_batch(feedback_rows: List[Dict]) -> None:
    if not feedback_rows:
        return
    feedback_df = load_feedback_raw()
    columns = ["user_id", "matched_user_id", "action", "timestamp"]
    new_df = pd.DataFrame(feedback_rows, columns=columns)
    feedback_df = pd.concat([feedback_df, new_df], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_PATH, index=False)
