"""
NexMatch AI — Input Validation Module
Provides field-level and profile-level validation for registration and profile updates.
"""

import re
from typing import Dict, List, Tuple


def validate_username(username: str) -> Tuple[bool, str]:
    """
    Validate username meets requirements:
    - 4-20 characters
    - Alphanumeric + underscore only
    """
    username = username.strip()
    if len(username) < 4:
        return False, "Username must be at least 4 characters long."
    if len(username) > 20:
        return False, "Username must be at most 20 characters long."
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate email address syntax and length.
    """
    email = email.strip()
    if not email:
        return False, "Email address is required."
    if len(email) > 100:
        return False, "Email address must be at most 100 characters long."
    # Standard email regex pattern
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(pattern, email):
        return False, "Invalid email address format."
    return True, ""



def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validate password strength:
    - >= 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit
    - At least 1 special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit."
    if not re.search(r'[^a-zA-Z0-9]', password):
        return False, "Password must contain at least one special character (!@#$%^&* etc.)."
    return True, ""


def validate_skills(skills_text: str) -> Tuple[bool, str]:
    """
    Validate skills list:
    - Minimum 3, maximum 15 comma-separated entries
    """
    skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    if len(skills) < 3:
        return False, f"Please provide at least 3 skills (found {len(skills)}). Separate with commas."
    if len(skills) > 15:
        return False, f"Maximum 15 skills allowed (found {len(skills)}). Remove some entries."
    return True, ""


def validate_interests(interests_text: str) -> Tuple[bool, str]:
    """
    Validate interests list:
    - Minimum 2, maximum 10 comma-separated entries
    """
    interests = [i.strip() for i in interests_text.split(",") if i.strip()]
    if len(interests) < 2:
        return False, f"Please provide at least 2 interests (found {len(interests)}). Separate with commas."
    if len(interests) > 10:
        return False, f"Maximum 10 interests allowed (found {len(interests)}). Remove some entries."
    return True, ""


def validate_summary(summary: str) -> Tuple[bool, str]:
    """
    Validate professional summary:
    - 50-500 characters
    """
    summary = summary.strip()
    if len(summary) < 50:
        return False, f"Professional Summary must be at least 50 characters ({len(summary)}/50)."
    if len(summary) > 500:
        return False, f"Professional Summary must be at most 500 characters ({len(summary)}/500)."
    return True, ""


def validate_about_me(about_me: str) -> Tuple[bool, str]:
    """
    Validate about me text:
    - 100-1000 characters
    """
    about_me = about_me.strip()
    if len(about_me) < 100:
        return False, f"About Me must be at least 100 characters ({len(about_me)}/100)."
    if len(about_me) > 1000:
        return False, f"About Me must be at most 1000 characters ({len(about_me)}/1000)."
    return True, ""


def validate_name(name: str) -> Tuple[bool, str]:
    """Validate name is non-empty and reasonable length."""
    name = name.strip()
    if len(name) < 2:
        return False, "Name must be at least 2 characters long."
    if len(name) > 50:
        return False, "Name must be at most 50 characters long."
    return True, ""


def validate_location(location: str) -> Tuple[bool, str]:
    """Validate location is non-empty."""
    location = location.strip()
    if len(location) < 2:
        return False, "Location must be at least 2 characters long."
    return True, ""


def validate_profile(profile: dict) -> Tuple[bool, List[str]]:
    """
    Full profile validation. Returns (is_valid, list_of_error_messages).
    """
    errors = []

    ok, msg = validate_name(profile.get("name", ""))
    if not ok:
        errors.append(msg)

    if not profile.get("profession", "").strip():
        errors.append("Profession is required.")

    ok, msg = validate_location(profile.get("location", ""))
    if not ok:
        errors.append(msg)

    if not profile.get("career_goal", "").strip():
        errors.append("Career Goal is required.")

    ok, msg = validate_skills(profile.get("skills", ""))
    if not ok:
        errors.append(msg)

    ok, msg = validate_interests(profile.get("interests", ""))
    if not ok:
        errors.append(msg)

    ok, msg = validate_summary(profile.get("professional_summary", ""))
    if not ok:
        errors.append(msg)

    ok, msg = validate_about_me(profile.get("about_me", ""))
    if not ok:
        errors.append(msg)

    return len(errors) == 0, errors


def calculate_profile_completeness(profile: dict) -> Tuple[float, Dict[str, bool]]:
    """
    Calculate profile completeness as a percentage.
    Returns (percentage, field_status_dict).
    """
    fields = {
        "Name": bool(str(profile.get("name", "")).strip()),
        "Profession": bool(str(profile.get("profession", "")).strip()),
        "Location": bool(str(profile.get("location", "")).strip()),
        "Experience": profile.get("experience_years", 0) is not None,
        "MBTI": bool(str(profile.get("mbti", "")).strip()),
        "Career Goal": bool(str(profile.get("career_goal", "")).strip()),
        "Networking Intent": bool(str(profile.get("networking_intent", "")).strip()),
        "Skills": len([s.strip() for s in str(profile.get("skills", "")).split(",") if s.strip()]) >= 1,
        "Interests": len([i.strip() for i in str(profile.get("interests", "")).split(",") if i.strip()]) >= 1,
        "Professional Summary": len(str(profile.get("professional_summary", "")).strip()) >= 50,
        "About Me": len(str(profile.get("about_me", "")).strip()) >= 100,
    }

    filled = sum(1 for v in fields.values() if v)
    total = len(fields)
    percentage = (filled / total) * 100 if total > 0 else 0.0

    return percentage, fields
