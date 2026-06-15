"""
NexMatch AI — Match Explanation Engine
Generates natural language summaries, structured reasons, shared skills,
strengths, and weaknesses for a matched pair of users.

DO NOT modify matching/recommender.py or learning/adaptive_recommender.py here.
This module only post-processes existing feature scores.
"""

from typing import Dict, List, Tuple


def generate_match_explanation(
    user_a: dict,
    user_b: dict,
    feature_scores: dict
) -> dict:
    """
    Given two user profile dicts and their compatibility feature scores,
    returns a structured explanation dict with:
    - summary:       Natural language paragraph
    - reasons:       List of ✓ reasons (strong dimensions)
    - shared_skills: List of overlapping skill names
    - strengths:     List of (label, score) tuples for scores >= 70
    - weaknesses:    List of (label, score) tuples for scores < 40
    """

    # ── Shared Skills ─────────────────────────────────────────────────────────
    skills_a = set(s.strip().lower() for s in str(user_a.get("skills", "")).split(",") if s.strip())
    skills_b = set(s.strip().lower() for s in str(user_b.get("skills", "")).split(",") if s.strip())
    shared = sorted([s.title() for s in (skills_a & skills_b)])

    # ── Dimension Meta ────────────────────────────────────────────────────────
    dimension_labels = {
        "text_similarity":        "Profile Text Similarity",
        "mbti_score":             "MBTI Compatibility",
        "profession_score":       "Profession Match",
        "career_goal_score":      "Career Goal Match",
        "skills_score":           "Skills Overlap",
        "experience_score":       "Experience Level",
        "location_score":         "Location Match",
        "networking_intent_score":"Networking Intent",
    }

    # ── Reasons (✓ checkmarks for strong signals) ─────────────────────────────
    reasons: List[str] = []

    if feature_scores.get("profession_score", 0) == 100:
        reasons.append(f"✓ Same Profession ({user_a.get('profession', '')})")
    elif feature_scores.get("profession_score", 0) >= 70:
        reasons.append(f"✓ Related Profession ({user_a.get('profession', '')} ↔ {user_b.get('profession', '')})")

    if feature_scores.get("career_goal_score", 0) >= 70:
        reasons.append(f"✓ Same Career Goal ({user_a.get('career_goal', '')})")

    if feature_scores.get("skills_score", 0) >= 50 and shared:
        top3 = shared[:3]
        reasons.append(f"✓ Shared Skills: {', '.join(top3)}")
    elif feature_scores.get("skills_score", 0) >= 30:
        reasons.append("✓ Overlapping Technical Skills")

    if feature_scores.get("mbti_score", 0) >= 80:
        reasons.append(f"✓ Highly Compatible MBTI ({user_a.get('mbti', '')} & {user_b.get('mbti', '')})")
    elif feature_scores.get("mbti_score", 0) >= 50:
        reasons.append(f"✓ Compatible MBTI Types ({user_a.get('mbti', '')} & {user_b.get('mbti', '')})")

    if feature_scores.get("experience_score", 0) >= 80:
        reasons.append("✓ Similar Experience Level")

    if feature_scores.get("location_score", 0) == 100:
        reasons.append(f"✓ Same Location ({user_a.get('location', '')})")

    if feature_scores.get("networking_intent_score", 0) >= 70:
        reasons.append(f"✓ Aligned Networking Intent ({user_b.get('networking_intent', '')})")

    if feature_scores.get("text_similarity", 0) >= 40:
        reasons.append("✓ Strong Profile Text Similarity")

    # Guarantee at least one reason
    if not reasons:
        best_dim = max(
            [(k, v) for k, v in feature_scores.items() if k in dimension_labels],
            key=lambda x: x[1],
            default=("profession_score", 0)
        )
        reasons.append(f"✓ Best Alignment: {dimension_labels.get(best_dim[0], best_dim[0])}")

    # ── Strengths and Weaknesses ──────────────────────────────────────────────
    strengths: List[Tuple[str, float]] = []
    weaknesses: List[Tuple[str, float]] = []

    for key, label in dimension_labels.items():
        val = feature_scores.get(key, 0)
        if val >= 70:
            strengths.append((label, round(val, 1)))
        elif val < 40:
            weaknesses.append((label, round(val, 1)))

    strengths.sort(key=lambda x: x[1], reverse=True)
    weaknesses.sort(key=lambda x: x[1])

    # ── Natural Language Summary ───────────────────────────────────────────────
    name_a = user_a.get("name", "You").split()[0]
    name_b = user_b.get("name", "this user").split()[0]
    prof_a = user_a.get("profession", "")
    prof_b = user_b.get("profession", "")
    goal_a = user_a.get("career_goal", "")
    goal_b = user_b.get("career_goal", "")
    loc_a = user_a.get("location", "")
    loc_b = user_b.get("location", "")

    # Build summary paragraph
    parts: List[str] = []

    # Profession line
    if feature_scores.get("profession_score", 0) == 100:
        parts.append(f"{name_a} and {name_b} both work as **{prof_a}**")
    elif feature_scores.get("profession_score", 0) >= 70:
        parts.append(f"{name_a} works as a **{prof_a}** and {name_b} works as a **{prof_b}**, both in closely related domains")
    else:
        parts.append(f"{name_a} ({prof_a}) and {name_b} ({prof_b}) bring complementary professional backgrounds")

    # Career goal line
    if goal_a == goal_b:
        parts.append(f"and are both focused on **{goal_a}** career pathways")
    elif feature_scores.get("career_goal_score", 0) >= 70:
        parts.append(f"with aligned career trajectories toward **{goal_a}** and **{goal_b}**")

    summary_intro = " ".join(parts) + "."

    # Skills line
    skills_line = ""
    if shared:
        top_shared = shared[:3]
        if len(top_shared) == 1:
            skills_line = f"\n\nThey share expertise in **{top_shared[0]}**"
        else:
            skills_line = f"\n\nThey share expertise in **{', '.join(top_shared[:-1])}** and **{top_shared[-1]}**"
        skills_line += " — a strong signal of technical alignment."

    # Networking line
    networking_line = ""
    if feature_scores.get("networking_intent_score", 0) >= 70:
        networking_line = f"\n\nThis recommendation was boosted because their networking objectives are well-aligned."

    # Location line
    location_line = ""
    if feature_scores.get("location_score", 0) == 100:
        location_line = f"\n\nBoth are located in **{loc_a}**, enabling potential in-person collaboration."

    summary = summary_intro + skills_line + networking_line + location_line

    return {
        "summary": summary,
        "reasons": reasons,
        "shared_skills": shared,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
