import random
import pandas as pd
from datetime import datetime, timedelta


MBTI_COMPATIBILITY = {
    "INTJ": ["ENFP","ENTP"],
    "INTP": ["ENTJ","ENFJ"],
    "INFJ": ["ENTP","ENFP"],
    "ENFP": ["INTJ","INFJ"]
}


def calculate_score(user, target):

    score = 0

    if user["profession"] == target["profession"]:
        score += 0.15

    user_interests = set(user["interests"].split(","))
    target_interests = set(target["interests"].split(","))

    overlap = len(
        user_interests & target_interests
    )

    score += (overlap / 5) * 0.20

    if (
        user["mbti"] in MBTI_COMPATIBILITY and
        target["mbti"] in MBTI_COMPATIBILITY[user["mbti"]]
    ):
        score += 0.25

    if abs(
        user["experience_years"]
        - target["experience_years"]
    ) <= 3:
        score += 0.15

    if user["location"] == target["location"]:
        score += 0.10

    return score


def generate_feedback(users_df):

    feedback = []

    users = users_df.to_dict("records")

    for user in users:

        candidates = random.sample(users, 8)

        for candidate in candidates:

            if user["user_id"] == candidate["user_id"]:
                continue

            score = calculate_score(
                user,
                candidate
            )
            probability = min(score*1.8, 0.95)
            action = 1 if random.random() < probability else 0

            timestamp = (
                datetime.now()
                - timedelta(
                    days=random.randint(1,365)
                )
            )

            feedback.append({
                "user_id": user["user_id"],
                "matched_user_id":
                    candidate["user_id"],
                "action": action,
                "timestamp": timestamp
            })

    feedback_df = pd.DataFrame(feedback)

    feedback_df.to_csv(
        "feedback.csv",
        index=False
    )

    return feedback_df