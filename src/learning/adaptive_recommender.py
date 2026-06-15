"""
NexMatch AI — Adaptive Recommender
Wraps the core Recommender + FeedbackModel to apply:
  - ML re-ranking (60% hybrid + 40% logistic regression)
  - Phase 7.10: Recency penalty (freshness) — avoids repeatedly showing same users
  - Phase 7.5:  Diversity filter — limits over-representation of any one profession

CRITICAL: Do NOT modify src/matching/recommender.py or src/learning/feedback_model.py.
          These post-processing filters are applied AFTER all scoring is complete.
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7.5 — Diversity Filter Configuration
# ──────────────────────────────────────────────────────────────────────────────
ENABLE_DIVERSITY_FILTER = True     # Set False to disable profession capping
MAX_SAME_PROFESSION = 2            # Max candidates with same profession in final Top-N

# ──────────────────────────────────────────────────────────────────────────────
# Phase 7.10 — Recency Penalty Configuration
# ──────────────────────────────────────────────────────────────────────────────
ENABLE_FRESHNESS_FILTER = True     # Set False to disable recency penalty entirely
RECENCY_WINDOW_DAYS = 7            # How many days back to check for recent shows
RECENCY_PENALTY_WEIGHT = 0.10      # Score fraction deducted per recent appearance
MAX_RECENCY_PENALTY = 0.25         # Maximum penalty cap (never reduce below 75% of original score)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_CSV = PROJECT_ROOT / "data" / "recommendation_history.csv"


def _load_recent_seen(user_id: str) -> set:
    """
    Phase 7.10 helper: load the set of candidate user_ids that were
    recommended to user_id within the last RECENCY_WINDOW_DAYS days.
    Returns an empty set if history CSV doesn't exist or has no recent rows.
    """
    if not ENABLE_FRESHNESS_FILTER or not HISTORY_CSV.exists():
        return set()

    try:
        df = pd.read_csv(HISTORY_CSV)
        if df.empty or "user_id" not in df.columns or "recommended_user_id" not in df.columns:
            return set()
        if "timestamp" not in df.columns:
            return set()

        user_hist = df[df["user_id"] == user_id].copy()
        if user_hist.empty:
            return set()

        user_hist["timestamp"] = pd.to_datetime(user_hist["timestamp"], errors="coerce")
        cutoff = datetime.now() - timedelta(days=RECENCY_WINDOW_DAYS)
        recent = user_hist[user_hist["timestamp"] >= cutoff]
        return set(recent["recommended_user_id"].dropna().unique())
    except Exception:
        return set()


def _apply_recency_penalty(recommendations: list, recently_seen: set) -> list:
    """
    Phase 7.10: Apply a small score penalty to recently shown candidates.
    Modifies final_ranking_score only. Does not alter ML or hybrid scores.
    Penalty is proportional to RECENCY_PENALTY_WEIGHT and capped at MAX_RECENCY_PENALTY.
    """
    if not recently_seen:
        return recommendations

    for rec in recommendations:
        if rec["user_id"] in recently_seen:
            original = rec["final_ranking_score"]
            penalty = min(original * RECENCY_PENALTY_WEIGHT, original * MAX_RECENCY_PENALTY)
            rec["final_ranking_score"] = round(original - penalty, 2)

    return recommendations


def _apply_diversity_filter(recommendations: list, top_n: int) -> list:
    """
    Phase 7.5: Deterministic soft diversity filter.
    Caps the number of candidates with the same profession at MAX_SAME_PROFESSION
    in the final output. Overflow slots are filled from the next-best
    candidates that do not violate the cap.
    Purely rank-preserving — no randomization.
    """
    if not ENABLE_DIVERSITY_FILTER:
        return recommendations[:top_n]

    profession_count: dict = {}
    selected = []
    overflow = []

    for rec in recommendations:
        prof = rec.get("profession", "")
        count = profession_count.get(prof, 0)
        if count < MAX_SAME_PROFESSION:
            selected.append(rec)
            profession_count[prof] = count + 1
        else:
            overflow.append(rec)

        if len(selected) >= top_n:
            break

    # If we still have slots, fill from overflow
    if len(selected) < top_n:
        for rec in overflow:
            selected.append(rec)
            if len(selected) >= top_n:
                break

    return selected[:top_n]


class AdaptiveRecommender:

    def __init__(
        self,
        recommender,
        feedback_model
    ):

        self.recommender = recommender

        self.feedback_model = feedback_model
        
    def predict_match_score(
        self,
        user_id_1,
        user_id_2
    ):

        result = (
            self.recommender.compatibility_score(
                user_id_1,
                user_id_2
            )
        )

        features = [

            result["text_similarity"],

            result["mbti_score"],

            result["profession_score"],

            result["career_goal_score"],

            result["location_score"],

            result["experience_score"],

            result["skills_score"],

            result["networking_intent_score"]
        ]

        probability = (
            self.feedback_model.predict_probability(
                features
            )
        )

        return round(
            probability * 100,
            2
        )
        
    def get_candidate_pool(
        self,
        user_id,
        pool_size=30
    ):
        candidate_pool = (
            self.recommender.get_top_recommendations(
                user_id,
                top_n=pool_size
            )
        )
        return candidate_pool

    def get_top_recommendations(
        self,
        user_id,
        top_n=5
    ):
        candidate_pool = self.get_candidate_pool(
            user_id,
            pool_size=30
        )

        recommendations = []

        for candidate in candidate_pool:
            candidate_user_id = candidate["user_id"]

            ml_score = self.predict_match_score(
                user_id,
                candidate_user_id
            )

            hybrid_score = candidate["final_score"]

            final_ranking_score = 0.60 * hybrid_score + 0.40 * ml_score

            recommendations.append(
                {
                    "user_id":
                        candidate_user_id,

                    "profession":
                        candidate["profession"],

                    "career_goal":
                        candidate["career_goal"],

                    "mbti":
                        candidate["mbti"],

                    "location":
                        candidate["location"],

                    "experience":
                        candidate.get("experience"),

                    "experience_years":
                        candidate.get("experience"),

                    "hybrid_score":
                        round(hybrid_score, 2),

                    "ml_score":
                        round(ml_score, 2),

                    "score":
                        round(ml_score, 2),

                    "final_ranking_score":
                        round(final_ranking_score, 2)
                }
            )

        # Sort by score (pre-penalty)
        recommendations.sort(
            key=lambda x: x["final_ranking_score"],
            reverse=True
        )

        # Phase 7.10 — Apply recency penalty (freshness)
        if ENABLE_FRESHNESS_FILTER:
            recently_seen = _load_recent_seen(user_id)
            recommendations = _apply_recency_penalty(recommendations, recently_seen)
            # Re-sort after penalty adjustment
            recommendations.sort(
                key=lambda x: x["final_ranking_score"],
                reverse=True
            )

        # Phase 7.5 — Apply profession diversity filter
        return _apply_diversity_filter(recommendations, top_n)