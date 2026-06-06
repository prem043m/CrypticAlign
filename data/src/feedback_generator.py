import random
import pandas as pd
from datetime import datetime, timedelta


# -----------------------------
# MBTI COMPATIBILITY
# -----------------------------

MBTI_COMPATIBILITY = {

    "INTJ": ["ENFP", "ENTP"],
    "INTP": ["ENTJ", "ENFJ"],
    "INFJ": ["ENTP", "ENFP"],
    "ENFP": ["INTJ", "INFJ"]
}


# -----------------------------
# PROFESSION GROUPS
# -----------------------------

TECH_PROFESSIONS = {
    "Data Scientist",
    "ML Engineer",
    "AI Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "DevOps Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst"
}

HEALTHCARE_PROFESSIONS = {
    "Doctor",
    "Nurse",
    "Healthcare Analyst"
}

BUSINESS_PROFESSIONS = {
    "Business Analyst",
    "Product Manager",
    "Project Manager",
    "Consultant"
}

FINANCE_PROFESSIONS = {
    "Financial Analyst",
    "Investment Advisor",
    "Accountant"
}


# -----------------------------
# CAREER GROUPS
# -----------------------------

CAREER_GROUPS = {

    "AI Research": "AI",
    "Data Analytics": "AI",

    "Cloud Computing": "TECH",
    "Cybersecurity": "TECH",

    "Leadership": "BUSINESS",
    "Product Management": "BUSINESS",

    "Financial Growth": "FINANCE",

    "Healthcare Innovation": "HEALTHCARE",

    "Startup Founder": "ENTREPRENEUR"
}


# -----------------------------
# HELPERS
# -----------------------------

def profession_score(
    profession_a,
    profession_b
):

    if profession_a == profession_b:
        return 1.0

    groups = [
        TECH_PROFESSIONS,
        HEALTHCARE_PROFESSIONS,
        BUSINESS_PROFESSIONS,
        FINANCE_PROFESSIONS
    ]

    for group in groups:

        if (
            profession_a in group
            and
            profession_b in group
        ):
            return 0.7

    return 0.0


def career_score(
    goal_a,
    goal_b
):

    if goal_a == goal_b:
        return 1.0

    if (
        CAREER_GROUPS.get(goal_a)
        ==
        CAREER_GROUPS.get(goal_b)
    ):
        return 0.7

    return 0.0


def experience_score(
    exp_a,
    exp_b
):

    diff = abs(exp_a - exp_b)

    if diff <= 2:
        return 1.0

    if diff <= 5:
        return 0.8

    if diff <= 10:
        return 0.6

    return 0.3


def location_score(
    loc_a,
    loc_b
):

    return 1.0 if loc_a == loc_b else 0.0


# Enhanced Acceptance Zones for Stronger Learning Signals
def get_acceptance_probability(score):
    """
    Convert compatibility score to acceptance probability.
    Uses deterministic zones to create strong learning signals.
    Tuned to achieve a target acceptance rate of 25-35% (accept: 1, reject: 0).
    
    Thresholds:
    - score >= 0.75: 95% acceptance (strong match)
    - score >= 0.60: 85% acceptance (good match)
    - score >= 0.45: 70% acceptance (moderate match)
    - score >= 0.30: 45% acceptance (weak match)
    - score < 0.30: 18% acceptance (very weak match)
    """
    if score >= 0.75:
        return 0.95
    elif score >= 0.60:
        return 0.85
    elif score >= 0.45:
        return 0.70
    elif score >= 0.30:
        return 0.45
    else:
        return 0.18


def skills_overlap_score(
    user_skills,
    target_skills
):
    """Calculate skills overlap between two users."""
    user_skill_set = set(
        user_skills.split(",")
    )
    target_skill_set = set(
        target_skills.split(",")
    )
    
    overlap = len(
        user_skill_set & target_skill_set
    )
    
    max_overlap = max(
        len(user_skill_set),
        len(target_skill_set)
    )
    
    return (
        overlap / max_overlap
        if max_overlap > 0
        else 0.0
    )


def networking_intent_compatibility(
    intent_a,
    intent_b
):
    """Determine if two networking intents are compatible."""
    compatible_pairs = {
        ("Find Mentor", "Find Mentee"),
        ("Find Mentee", "Find Mentor"),
        ("Startup Partner", "Startup Partner"),
        ("Research Collaboration", "Research Collaboration"),
        ("Professional Networking", "Professional Networking"),
        ("Team Building", "Team Building"),
        ("Knowledge Sharing", "Knowledge Sharing"),
        ("Career Growth", "Career Growth"),
    }
    
    pair = (intent_a, intent_b)
    reverse_pair = (intent_b, intent_a)
    
    if (
        pair in compatible_pairs
        or reverse_pair in compatible_pairs
    ):
        return 1.0
    
    if (
        intent_a == "Professional Networking"
        or intent_b == "Professional Networking"
    ):
        return 0.6
    
    return 0.3


def mbti_score(
    mbti_a,
    mbti_b
):

    if mbti_a == mbti_b:
        return 0.75

    if (
        mbti_a in MBTI_COMPATIBILITY
        and
        mbti_b in MBTI_COMPATIBILITY[mbti_a]
    ):
        return 1.0

    return 0.5


# -----------------------------
# FEEDBACK SCORE
# -----------------------------

def calculate_score(
    user,
    target
):
    """
    Calculate weighted compatibility score.
    Weights are tuned for strong learning signals:
    - Profession: 0.25 (strong signal)
    - Career Goal: 0.20 (strong signal)
    - Interest Overlap: 0.25 (strong signal)
    - Skills Overlap: 0.10 (additional signal)
    - Networking Intent: 0.05 (compatibility check)
    - MBTI: 0.10 (personality signal)
    - Experience: 0.03 (weak signal)
    - Location: 0.02 (very weak signal)
    """

    score = 0.0

    # Profession Compatibility (25% weight) - STRENGTHENED
    score += (
        profession_score(
            user["profession"],
            target["profession"]
        )
        * 0.25
    )

    # Career Goal Alignment (20% weight) - STRENGTHENED
    score += (
        career_score(
            user["career_goal"],
            target["career_goal"]
        )
        * 0.20
    )

    # Interest Overlap (25% weight) - STRENGTHENED
    user_interests = set(
        user["interests"].split(",")
    )

    target_interests = set(
        target["interests"].split(",")
    )

    interest_overlap = len(
        user_interests &
        target_interests
    )

    score += (
        (interest_overlap / 5.0)
        * 0.25
    )

    # Skills Overlap (10% weight) - NEW
    if "skills" in user and "skills" in target:
        score += (
            skills_overlap_score(
                user["skills"],
                target["skills"]
            )
            * 0.10
        )

    # Networking Intent (5% weight) - NEW
    if (
        "networking_intent" in user
        and "networking_intent" in target
    ):
        score += (
            networking_intent_compatibility(
                user["networking_intent"],
                target["networking_intent"]
            )
            * 0.05
        )

    # MBTI Compatibility (10% weight)
    score += (
        mbti_score(
            user["mbti"],
            target["mbti"]
        )
        * 0.10
    )

    # Experience Compatibility (3% weight) - REDUCED
    score += (
        experience_score(
            user["experience_years"],
            target["experience_years"]
        )
        * 0.03
    )

    # Location (2% weight) - HEAVILY REDUCED
    score += (
        location_score(
            user["location"],
            target["location"]
        )
        * 0.02
    )

    return min(score, 1.0)


# -----------------------------
# GENERATE FEEDBACK
# -----------------------------

def generate_feedback(
    users_df
):
    """
    Generate feedback with stronger learning signals.
    
    - Increased candidate pool from 10 to 20 per user
    - Deterministic acceptance zones based on compatibility score
    - Enhanced feature weighting for meaningful patterns
    
    Expected feedback records: ~6000 (vs 2400 previously)
    """

    feedback = []

    users = users_df.to_dict(
        "records"
    )

    for user in users:

        # Increased from 10 to 20 for better training data
        candidates = random.sample(
            users,
            min(20, len(users))
        )

        for candidate in candidates:

            if (
                user["user_id"]
                ==
                candidate["user_id"]
            ):
                continue

            score = calculate_score(
                user,
                candidate
            )

            # Use deterministic acceptance zones
            # instead of direct probability mapping
            acceptance_probability = (
                get_acceptance_probability(score)
            )

            action = (
                1
                if random.random()
                < acceptance_probability
                else 0
            )

            timestamp = (
                datetime.now()
                - timedelta(
                    days=random.randint(
                        1,
                        365
                    )
                )
            )

            feedback.append(
                {
                    "user_id":
                        user["user_id"],

                    "matched_user_id":
                        candidate["user_id"],

                    "action":
                        action,

                    "timestamp":
                        timestamp
                }
            )

    feedback_df = pd.DataFrame(
        feedback
    )

    feedback_df.to_csv(
        "../feedback.csv",
        index=False
    )

    return feedback_df