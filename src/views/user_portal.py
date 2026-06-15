import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
from src.utils.loader import load_system, clear_system_caches, refresh_feedback_views
from src.utils.data_manager import (
    load_user_profiles_raw,
    update_user_profile,
    load_recommendation_history,
    append_recommendation_history_batch,
    append_feedback_batch,
    get_feedback_for_user,
    MBTI_OPTIONS,
    NETWORKING_INTENT_OPTIONS,
    PROFESSION_OPTIONS,
    CAREER_GOAL_OPTIONS
)
from src.utils.validators import (
    validate_profile,
    calculate_profile_completeness
)
from src.utils.explanation_engine import generate_match_explanation

@st.dialog("Match Compatibility Details")
def show_match_details_dialog(recommender, users_df, user_id, target_id):
    """
    Displays a premium pop-up modal showing compatibility score breakdowns,
    natural language explanation, shared skills, strengths, and weaknesses.
    Uses explanation_engine for structured output.
    """
    score = recommender.compatibility_score(user_id, target_id)
    target_row = users_df[users_df["user_id"] == target_id].iloc[0]
    user_row = users_df[users_df["user_id"] == user_id].iloc[0]

    explanation = generate_match_explanation(
        user_a=user_row.to_dict(),
        user_b=target_row.to_dict(),
        feature_scores=score
    )

    st.markdown(f"### 🤝 Match Analysis — {target_row['name']}")
    st.markdown(f"**{target_row['profession']}** · {target_row['location']} · {target_row.get('mbti', '')}")
    st.markdown("---")

    # ── Natural Language Summary ───────────────────────────────────────────────
    st.markdown("#### 💬 Why This Match?")
    st.info(explanation["summary"])

    # ── Shared Skills ─────────────────────────────────────────────────────────
    shared = explanation["shared_skills"]
    if shared:
        st.markdown("#### 🛠️ Shared Skills")
        badges_html = " ".join(
            f'<span style="background: rgba(59,130,246,0.15); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); border-radius: 12px; padding: 3px 10px; font-size: 0.8rem; margin: 2px;">{s}</span>'
            for s in shared[:6]
        )
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

    # ── Compatibility Score Bars ───────────────────────────────────────────────
    st.markdown("#### 📊 Compatibility Breakdown")

    def render_bar(label, key):
        val = score.get(key, 0)
        color = "#10b981" if val >= 70 else ("#f59e0b" if val >= 40 else "#ef4444")
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px; font-size: 0.85rem;">
            <span style="color: #cbd5e1;">{label}</span>
            <span style="font-weight: bold; color: {color};">{val:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(val / 100.0)

    render_bar("📝 Profile Text Similarity", "text_similarity")
    render_bar("🧬 MBTI Compatibility", "mbti_score")
    render_bar("💼 Profession Match", "profession_score")
    render_bar("🎯 Career Goal Match", "career_goal_score")
    render_bar("🛠️ Skills Overlap", "skills_score")
    render_bar("📈 Experience Level", "experience_score")
    render_bar("📍 Location Match", "location_score")
    render_bar("🤝 Networking Intent", "networking_intent_score")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 15px;">
        <span style="color: #34d399; font-weight: bold; font-size: 1.1rem;">Overall Match Score: {score['final_score']:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Strengths & Weaknesses ────────────────────────────────────────────────
    str_col, wk_col = st.columns(2)
    with str_col:
        if explanation["strengths"]:
            st.markdown("#### 💪 Top Strengths")
            for label, val in explanation["strengths"][:3]:
                st.markdown(f'<span style="color:#34d399;">✓</span> {label} **({val:.0f}%)**', unsafe_allow_html=True)
    with wk_col:
        if explanation["weaknesses"]:
            st.markdown("#### ⚠️ Gaps")
            for label, val in explanation["weaknesses"][:3]:
                st.markdown(f'<span style="color:#f87171;">✗</span> {label} **({val:.0f}%)**', unsafe_allow_html=True)


# --- Helper Functions ---

def _get_confidence_badge(final_score: float) -> tuple:
    """
    6-tier confidence label system.
    Returns (label, color_hex, bg_rgba).
    """
    if final_score >= 90:
        return "🌟 Exceptional Match", "#a855f7", "rgba(168, 85, 247, 0.15)"
    elif final_score >= 80:
        return "✨ Excellent Match", "#10b981", "rgba(16, 185, 129, 0.15)"
    elif final_score >= 70:
        return "💪 Strong Match", "#3b82f6", "rgba(59, 130, 246, 0.15)"
    elif final_score >= 60:
        return "👍 Good Match", "#f59e0b", "rgba(245, 158, 11, 0.15)"
    elif final_score >= 50:
        return "🔍 Potential Match", "#f97316", "rgba(249, 115, 22, 0.15)"
    else:
        return "🌱 Explore Match", "#64748b", "rgba(100, 116, 139, 0.15)"


def _get_match_reasons(recommender, users_df, user_id, target_id) -> list:
    """Generate dynamic match reasons from feature scores without hardcoding."""
    score = recommender.compatibility_score(user_id, target_id)
    user_row = users_df[users_df["user_id"] == user_id].iloc[0]
    target_row = users_df[users_df["user_id"] == target_id].iloc[0]
    reasons = []
    
    if score["career_goal_score"] >= 70:
        reasons.append("✓ Same Career Goal")
    
    if score["skills_score"] >= 50:
        reasons.append("✓ Similar Skills")
    
    if score["mbti_score"] >= 60:
        reasons.append("✓ Compatible MBTI")
    
    if score["experience_score"] >= 70:
        reasons.append("✓ Similar Experience")
    
    if score["profession_score"] >= 70:
        reasons.append("✓ Related Profession")
    
    if score["location_score"] == 100:
        reasons.append("✓ Same Location")
    
    if score["networking_intent_score"] >= 70:
        reasons.append("✓ Aligned Intent")
    
    if score["text_similarity"] >= 40:
        reasons.append("✓ Profile Similarity")
    
    # Always show at least one reason
    if not reasons:
        # Find the highest scoring dimension
        dims = [
            ("✓ Profile Similarity", score["text_similarity"]),
            ("✓ Related Profession", score["profession_score"]),
            ("✓ Similar Skills", score["skills_score"]),
        ]
        dims.sort(key=lambda x: x[1], reverse=True)
        reasons.append(dims[0][0])
    
    return reasons[:4]  # Max 4 visible reasons


def render_user_home(system):
    """
    Renders the User Dashboard. Shows active profile information,
    historical metrics, and allows profile searching.
    """
    users_df = system["users_df"]
    current_user = st.session_state.get("current_user")
    
    if not current_user or current_user == "ADMIN":
        st.warning("You are currently acting in Admin Context. Select or register a user profile to view the home dashboard.")
        return

    user_row = users_df[users_df["user_id"] == current_user].iloc[0]
    
    st.markdown(f'<div class="gradient-header">Welcome back, {user_row["name"]}!</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Manage your profile, view match recommendations, and browse professionals</div>', unsafe_allow_html=True)

    # 1. Dashboard Metrics
    st.markdown("### 📊 Your Recommendation Analytics")
    history_df = load_recommendation_history()
    user_history = history_df[history_df["user_id"] == current_user]
    total_viewed = len(user_history["recommended_user_id"].unique())
    
    feedback_df = get_feedback_for_user(current_user)
    accepted = len(feedback_df[feedback_df["action"] == 1])
    rejected = len(feedback_df[feedback_df["action"] == 0])
    skipped = len(user_history) - accepted - rejected if not user_history.empty else 0
    skipped = max(skipped, 0)
    
    acc_rate = (accepted / (accepted + rejected) * 100) if (accepted + rejected) > 0 else 0.0
    
    last_rec_date = "N/A"
    if not user_history.empty:
        last_rec_date = pd.to_datetime(user_history["timestamp"]).max().strftime("%Y-%m-%d %H:%M")
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Matches Seen", f"{total_viewed:,}")
    with m_col2:
        st.metric("✅ Accepted", f"{accepted:,}")
    with m_col3:
        st.metric("❌ Rejected", f"{rejected:,}")
    with m_col4:
        st.metric("Accept Rate", f"{acc_rate:.1f}%")
    with m_col5:
        st.metric("Last Batch", last_rec_date)

    # Phase 7.6 — Profile Completeness on Home Dashboard
    profile_dict = user_row.to_dict()
    completeness, field_status = calculate_profile_completeness(profile_dict)
    missing_fields = [k for k, v in field_status.items() if not v]

    st.markdown("<br/>", unsafe_allow_html=True)
    comp_color = "#10b981" if completeness >= 80 else ("#f59e0b" if completeness >= 50 else "#ef4444")
    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="color: #f1f5f9; font-weight: 600; font-size: 0.9rem;">📋 Profile Completeness</span>
            <span style="color: {comp_color}; font-weight: bold; font-size: 1.1rem;">{completeness:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(completeness / 100.0)
    if missing_fields and completeness < 100:
        st.caption(f"⚠️ Complete these fields to improve match quality: **{', '.join(missing_fields)}**")
    elif completeness == 100:
        st.caption("✅ Your profile is fully complete — great for match quality!")

    st.markdown("<br/>", unsafe_allow_html=True)


    # 2. Profile Search Component
    st.markdown("### 🔍 Search Professional Network")
    st.markdown("Browse other users by profession, location, or career goal.")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    professions = ["All"] + sorted(list(users_df["profession"].dropna().unique()))
    locations = ["All"] + sorted(list(users_df["location"].dropna().unique()))
    goals = ["All"] + sorted(list(users_df["career_goal"].dropna().unique()))
    
    with col_s1:
        sel_prof = st.selectbox("Profession", professions, key="home_search_prof")
    with col_s2:
        sel_loc = st.selectbox("Location", locations, key="home_search_loc")
    with col_s3:
        sel_goal = st.selectbox("Career Goal", goals, key="home_search_goal")
        
    search_df = users_df[users_df["user_id"] != current_user].copy()
    if sel_prof != "All":
        search_df = search_df[search_df["profession"] == sel_prof]
    if sel_loc != "All":
        search_df = search_df[search_df["location"] == sel_loc]
    if sel_goal != "All":
        search_df = search_df[search_df["career_goal"] == sel_goal]
        
    st.markdown(f"**Found {len(search_df)} profiles matching filters:**")
    
    if search_df.empty:
        st.info("No matching profiles found.")
    else:
        # Display as clean table
        display_cols = ["user_id", "name", "profession", "location", "experience_years", "mbti", "career_goal", "skills", "networking_intent"]
        display_df = search_df[display_cols].rename(columns={
            "user_id": "User ID",
            "name": "Name",
            "profession": "Profession",
            "location": "Location",
            "experience_years": "Experience (Yrs)",
            "mbti": "MBTI",
            "career_goal": "Career Goal",
            "skills": "Skills",
            "networking_intent": "Networking Intent"
        })
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        # Download Profile List button
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Profiles to CSV",
            data=csv_data,
            file_name="nexmatch_profiles_export.csv",
            mime="text/csv",
            key="download_profiles_btn"
        )


def render_user_profile(system):
    """
    Renders My Profile page, allowing standard users to view, edit,
    and save their own profile details. Includes profile completeness indicator.
    """
    users_df = system["users_df"]
    current_user = st.session_state.get("current_user")
    
    if not current_user or current_user == "ADMIN":
        st.warning("You are currently acting in Admin Context. Select a user profile to manage.")
        return

    st.markdown('<div class="gradient-header">My Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Manage your professional information, skills, and networking criteria</div>', unsafe_allow_html=True)

    user_row = users_df[users_df["user_id"] == current_user].iloc[0]

    # --- Profile Completeness (Part 3) ---
    profile_dict = user_row.to_dict()
    completeness, field_status = calculate_profile_completeness(profile_dict)
    
    st.markdown("### 📋 Profile Completeness")
    st.progress(completeness / 100.0)
    
    # Build status display
    complete_fields = [k for k, v in field_status.items() if v]
    missing_fields = [k for k, v in field_status.items() if not v]
    
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 10px; padding: 12px;">
            <span style="color: #34d399; font-weight: 700; font-size: 1.6rem;">{completeness:.0f}%</span>
            <span style="color: #94a3b8; margin-left: 8px;">Complete</span>
        </div>
        """, unsafe_allow_html=True)
    with comp_col2:
        if missing_fields:
            missing_str = ", ".join(missing_fields)
            st.warning(f"Missing or incomplete: {missing_str}")
        else:
            st.success("Your profile is fully complete! 🎉")

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- Profile Edit Form ---
    st.markdown("### Profile Settings")
    
    # Resolve safe selectbox defaults
    prof_val = str(user_row.get("profession", ""))
    prof_idx = PROFESSION_OPTIONS.index(prof_val) if prof_val in PROFESSION_OPTIONS else 0
    
    goal_val = str(user_row.get("career_goal", ""))
    goal_idx = CAREER_GOAL_OPTIONS.index(goal_val) if goal_val in CAREER_GOAL_OPTIONS else 0
    
    intent_val = str(user_row.get("networking_intent", ""))
    intent_idx = NETWORKING_INTENT_OPTIONS.index(intent_val) if intent_val in NETWORKING_INTENT_OPTIONS else 0
    
    mbti_val = str(user_row.get("mbti", "INTJ"))
    mbti_idx = MBTI_OPTIONS.index(mbti_val) if mbti_val in MBTI_OPTIONS else 0
    
    with st.form("edit_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=str(user_row.get("name", "")))
            profession = st.selectbox("Profession", PROFESSION_OPTIONS, index=prof_idx)
            location = st.text_input("Location", value=str(user_row.get("location", "")))
            experience_years = st.number_input("Experience Years", min_value=0, max_value=50, value=int(user_row.get("experience_years", 0)))
            mbti = st.selectbox("MBTI Type", MBTI_OPTIONS, index=mbti_idx)
            career_goal = st.selectbox("Career Goal", CAREER_GOAL_OPTIONS, index=goal_idx)
            
        with col2:
            networking_intent = st.selectbox("Networking Intent", NETWORKING_INTENT_OPTIONS, index=intent_idx)
            skills = st.text_area("Skills (comma separated, min 3)", value=str(user_row.get("skills", "")))
            interests = st.text_area("Interests (comma separated, min 2)", value=str(user_row.get("interests", "")))
            professional_summary = st.text_area("Professional Summary (50-500 chars)", value=str(user_row.get("professional_summary", "")))
            about_me = st.text_area("About Me (100-1000 chars)", value=str(user_row.get("about_me", "")))
            
        submitted = st.form_submit_button("Update Profile", type="primary")

    if submitted:
        updated_profile = {
            "name": name.strip(),
            "profession": profession,
            "location": location.strip(),
            "experience_years": int(experience_years),
            "mbti": mbti,
            "career_goal": career_goal,
            "skills": skills.strip(),
            "interests": interests.strip(),
            "networking_intent": networking_intent,
            "professional_summary": professional_summary.strip(),
            "about_me": about_me.strip(),
        }
        
        is_valid, errors = validate_profile(updated_profile)
        if not is_valid:
            for err in errors:
                st.error(f"🔴 {err}")
        else:
            update_user_profile(current_user, updated_profile)
            from src.utils.audit_logger import log_event
            log_event(current_user, "PROFILE_UPDATE", f"Profile updated by {current_user}")
            clear_system_caches()
            st.success("Your profile has been updated successfully!")
            st.rerun()

    # Phase 8.13 — Notification Preferences
    st.markdown("---")
    st.markdown("### 🔔 Notification Preferences")
    st.caption("Control which emails and notifications you receive from NexMatch AI.")

    # Load current preferences from profile (default True if not set)
    pref_welcome = bool(user_row.get("notif_welcome", True))
    pref_digest = bool(user_row.get("notif_digest", True))
    pref_system = bool(user_row.get("notif_system", True))

    with st.form("notif_prefs_form"):
        n_col1, n_col2, n_col3 = st.columns(3)
        with n_col1:
            notif_welcome = st.checkbox(
                "Welcome Emails",
                value=pref_welcome,
                help="Receive a welcome email when you register or re-activate your account."
            )
        with n_col2:
            notif_digest = st.checkbox(
                "Recommendation Digests",
                value=pref_digest,
                help="Receive email digests showing your top new matches."
            )
        with n_col3:
            notif_system = st.checkbox(
                "System Notifications",
                value=pref_system,
                help="Receive feedback confirmation and important platform notifications."
            )
        save_notif = st.form_submit_button("Save Notification Preferences")

    if save_notif:
        notif_prefs = {
            "notif_welcome": notif_welcome,
            "notif_digest": notif_digest,
            "notif_system": notif_system,
        }
        update_user_profile(current_user, notif_prefs)
        st.success("Notification preferences saved.")


def render_user_recs(system):
    """
    Renders My Recommendations page. Handles batch-based recommendation snapshots,
    rendering match cards with Accept/Reject/Skip options, and View Match Details modal.
    """
    users_df = system["users_df"]
    adaptive = system["adaptive"]
    recommender = system["recommender"]
    current_user = st.session_state.get("current_user")
    
    if not current_user or current_user == "ADMIN":
        st.warning("You are currently acting in Admin Context. Select a user profile to receive recommendations.")
        return

    user_row = users_df[users_df["user_id"] == current_user].iloc[0]


    st.markdown('<div class="gradient-header">My Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Highly compatible professionals retrieved by our hybrid engine and ranked by adaptive learning</div>', unsafe_allow_html=True)

    # 1. Snapshot retrieval
    snapshots = st.session_state.setdefault("recommendation_snapshots", {})
    snapshot = snapshots.get(current_user)

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        st.caption("Recommendations are locked into a stable snapshot. Click 'Generate/Refresh Recommendations' to pull a new batch.")
    with btn_col2:
        if st.button("⚡ Generate / Refresh Recommendations", type="primary", key="refresh_recs_batch_btn"):
            with st.spinner("Retrieving candidates and re-ranking matches..."):
                batch_id = f"B-{uuid.uuid4().hex[:8].upper()}"
                timestamp_str = datetime.now().isoformat()
                recs = adaptive.get_top_recommendations(current_user, top_n=5)
                
                snapshot = {
                    "batch_id": batch_id,
                    "timestamp": timestamp_str,
                    "recs": recs,
                    "selections": {}
                }
                snapshots[current_user] = snapshot
                st.session_state["recommendation_snapshots"] = snapshots
                
                # Append to history batch
                append_recommendation_history_batch(current_user, batch_id, recs, timestamp_str)
                st.rerun()

    if not snapshot:
        st.info("No active recommendations snapshot. Click the button above to generate recommendations.")
        return

    st.markdown(f"**Recommendation Batch:** `{snapshot['batch_id']}` | **Generated At:** {pd.to_datetime(snapshot['timestamp']).strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")

    recs = snapshot["recs"]
    selections = snapshot["selections"]

    # --- Part 10: Recommendation Summary ---
    high_conf = sum(1 for r in recs if r.get("final_ranking_score", 0) >= 85)
    med_conf = sum(1 for r in recs if 70 <= r.get("final_ranking_score", 0) < 85)
    expl_conf = sum(1 for r in recs if r.get("final_ranking_score", 0) < 70)
    
    st.markdown("### 📋 Recommendation Summary")
    sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
    with sum_c1:
        st.metric("Total Recommendations", len(recs))
    with sum_c2:
        st.metric("🟢 High Confidence", high_conf)
    with sum_c3:
        st.metric("🟡 Medium Confidence", med_conf)
    with sum_c4:
        st.metric("⚪ Exploratory", expl_conf)
    
    st.markdown("---")

    # --- Render Cards (Part 4, 5, 6, 7) ---
    for idx, rec in enumerate(recs):
        target_id = rec["user_id"]
        target_row = users_df[users_df["user_id"] == target_id].iloc[0]
        
        final_score = rec.get("final_ranking_score", 0)
        hybrid_score = rec.get("hybrid_score", 0)
        ml_score = rec.get("ml_score", 0)
        
        # Confidence badge (6-tier)
        conf_label, conf_color, conf_bg = _get_confidence_badge(final_score)
        
        # Match reasons — dynamically generated
        match_reasons = _get_match_reasons(recommender, users_df, current_user, target_id)
        
        # Phase 7.3: Shared skills (intersection of current user's skills and target user's skills)
        user_skills = set(s.strip().lower() for s in str(user_row.get("skills", "")).split(",") if s.strip())
        target_skills = set(s.strip().lower() for s in str(target_row.get("skills", "")).split(",") if s.strip())
        shared_skills_set = user_skills & target_skills
        shared_skills_top3 = sorted([s.title() for s in shared_skills_set])[:3]
        
        # Shared skills badges HTML (Phase 7.3)
        if shared_skills_top3:
            skills_badges = " ".join([
                f'<span style="background: rgba(59,130,246,0.12); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); border-radius: 10px; padding: 2px 8px; font-size: 0.72rem; margin: 1px;">⚡ {s}</span>'
                for s in shared_skills_top3
            ])
            shared_label_html = '<div style="color: #64748b; font-size: 0.68rem; margin-bottom: 3px;">Shared Skills</div>'
            skills_badges = shared_label_html + skills_badges
        else:
            # Fall back to target's own top skills
            skills_list = [s.strip() for s in str(target_row.get("skills", "")).split(",") if s.strip()][:3]
            skills_badges = " ".join([f'<span class="badge badge-blue">{s}</span>' for s in skills_list])
        
        # Match reason badges HTML
        reasons_html = " ".join([
            f'<span style="display: inline-block; background: rgba(16, 185, 129, 0.1); color: #34d399; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; margin: 2px 3px 2px 0; border: 1px solid rgba(16, 185, 129, 0.2);">{r}</span>'
            for r in match_reasons
        ])

        
        # --- Compact Card Layout: [3, 2, 3, 2] (Part 4) ---
        col1, col2, col3, col4 = st.columns([3, 2, 3, 2])
        
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #3b82f6; margin-bottom: 0px; padding: 15px;">
                <h4 style="margin: 0 0 4px; color: #f8fafc; font-size: 1.1rem;">{target_row['name']}</h4>
                <p style="margin: 0; color: #a855f7; font-weight: 600; font-size: 0.9rem;">{rec['profession']}</p>
                <p style="margin: 4px 0 0; color: #cbd5e1; font-size: 0.82rem;">📍 {rec['location']} | 💼 {target_row['experience_years']} yrs</p>
                <div style="margin-top: 6px;">
                    <span style="background: {conf_bg}; color: {conf_color}; padding: 2px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid {conf_color}40;">{conf_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 0px; padding: 15px; text-align: center;">
                <div style="margin-bottom: 8px;">
                    <span style="font-weight: bold; color: #34d399; font-size: 1.4rem;">{final_score:.1f}%</span>
                    <div style="color: #64748b; font-size: 0.7rem; margin-top: 2px;">Final Match</div>
                </div>
                <div style="display: flex; justify-content: space-around;">
                    <div style="text-align: center;">
                        <span style="color: #60a5fa; font-weight: 600; font-size: 0.9rem;">{hybrid_score:.1f}%</span>
                        <div style="color: #64748b; font-size: 0.65rem;">Hybrid</div>
                    </div>
                    <div style="text-align: center;">
                        <span style="color: #c084fc; font-weight: 600; font-size: 0.9rem;">{ml_score:.1f}%</span>
                        <div style="color: #64748b; font-size: 0.65rem;">ML</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            intent_val = rec.get('networking_intent', target_row.get('networking_intent', 'N/A'))
            st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 0px; padding: 15px;">
                <p style="margin: 0 0 3px; color: #cbd5e1; font-size: 0.82rem;">🎯 {rec['career_goal']}</p>
                <p style="margin: 0 0 6px; color: #94a3b8; font-size: 0.82rem;">🤝 {intent_val}</p>
                <div style="margin-bottom: 6px;">{skills_badges}</div>
                <div style="margin-top: 4px;">{reasons_html}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
            # View Details button
            if st.button("🔍 View Details", key=f"details_btn_{target_id}_{idx}"):
                show_match_details_dialog(recommender, users_df, current_user, target_id)
            
            # Feedback buttons (Part 7)
            current_selection = selections.get(target_id, "Skip")
            
            fb_c1, fb_c2, fb_c3 = st.columns(3)
            with fb_c1:
                accept_type = "primary" if current_selection == "Accept" else "secondary"
                if st.button("👍", key=f"accept_{target_id}_{idx}", type=accept_type):
                    selections[target_id] = "Accept"
                    snapshot["selections"] = selections
                    snapshots[current_user] = snapshot
                    st.session_state["recommendation_snapshots"] = snapshots
                    st.rerun()
            with fb_c2:
                reject_type = "primary" if current_selection == "Reject" else "secondary"
                if st.button("👎", key=f"reject_{target_id}_{idx}", type=reject_type):
                    selections[target_id] = "Reject"
                    snapshot["selections"] = selections
                    snapshots[current_user] = snapshot
                    st.session_state["recommendation_snapshots"] = snapshots
                    st.rerun()
            with fb_c3:
                skip_type = "primary" if current_selection == "Skip" else "secondary"
                if st.button("⏭", key=f"skip_{target_id}_{idx}", type=skip_type):
                    selections[target_id] = "Skip"
                    snapshot["selections"] = selections
                    snapshots[current_user] = snapshot
                    st.session_state["recommendation_snapshots"] = snapshots
                    st.rerun()
        
        st.markdown("")  # Small spacing between cards

    # Persist selections
    snapshot["selections"] = selections
    snapshots[current_user] = snapshot
    st.session_state["recommendation_snapshots"] = snapshots

    st.markdown("---")
    
    # --- Pending Decisions Counter (Part 7) ---
    pending_count = sum(1 for v in selections.values() if v in ("Accept", "Reject"))
    accepted_count = sum(1 for v in selections.values() if v == "Accept")
    rejected_count = sum(1 for v in selections.values() if v == "Reject")
    
    st.markdown(f"""
    <div style="background: rgba(168, 85, 247, 0.06); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 10px; padding: 10px 16px; margin-bottom: 12px;">
        <span style="color: #c084fc; font-weight: 600;">Pending Decisions: {pending_count}</span>
        <span style="color: #64748b; margin-left: 12px; font-size: 0.85rem;">
            (👍 {accepted_count} Accept | 👎 {rejected_count} Reject)
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Bottom Submit and Export Buttons
    col_sub1, col_sub2 = st.columns([2, 2])
    with col_sub1:
        if st.button("📤 Submit Decisions", type="primary", key="batch_feedback_submit_btn"):
            feedback_rows = []
            for rec in recs:
                tid = rec["user_id"]
                sel = selections.get(tid, "Skip")
                if sel == "Accept":
                    feedback_rows.append({
                        "user_id": current_user,
                        "matched_user_id": tid,
                        "action": 1,
                        "timestamp": datetime.now().isoformat()
                    })
                elif sel == "Reject":
                    feedback_rows.append({
                        "user_id": current_user,
                        "matched_user_id": tid,
                        "action": 0,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            if feedback_rows:
                append_feedback_batch(feedback_rows)
                
                # Part 11: Queue notifications for accepts & send feedback confirmation email
                from src.utils.notifications import store_pending_notification
                from src.utils.email_service import send_feedback_confirmation_email
                from src.utils.data_manager import get_email_for_user
                
                user_email = get_email_for_user(current_user)
                user_name = user_row["name"]
                
                accepts_count = 0
                for row in feedback_rows:
                    if row["action"] == 1:
                        accepts_count += 1
                        store_pending_notification(current_user, row["matched_user_id"], "mutual_match")
                
                if user_email:
                    send_feedback_confirmation_email(user_name, user_email, len(feedback_rows))
                    
                refresh_feedback_views()
                st.success(f"Successfully submitted batch feedback ({len(feedback_rows)} decisions)!")
                # Reset selections in snapshot
                snapshot["selections"] = {}
                snapshots[current_user] = snapshot
                st.session_state["recommendation_snapshots"] = snapshots
                st.rerun()
            else:
                st.warning("No feedback items selected. Use 👍 or 👎 buttons to make decisions first.")
                
    with col_sub2:
        # --- Improved CSV Export (Part 9) ---
        export_rows = []
        for rec in recs:
            tid = rec["user_id"]
            target_row = users_df[users_df["user_id"] == tid].iloc[0]
            f_score = rec.get("final_ranking_score", 0)
            conf_label, _, _ = _get_confidence_badge(f_score)
            export_rows.append({
                "Name": target_row["name"],
                "Profession": rec["profession"],
                "Location": rec["location"],
                "Career Goal": rec["career_goal"],
                "Networking Intent": rec.get("networking_intent", target_row.get("networking_intent", "")),
                "Hybrid Score": f"{rec['hybrid_score']:.2f}",
                "ML Score": f"{rec['ml_score']:.2f}",
                "Final Score": f"{f_score:.2f}",
                "Confidence": conf_label
            })
        rec_export_df = pd.DataFrame(export_rows)
        csv_recs = rec_export_df.to_csv(index=False).encode('utf-8')
        date_str = datetime.now().strftime("%Y%m%d")
        st.download_button(
            label="📥 Download Recommendations CSV",
            data=csv_recs,
            file_name=f"recommendations_{current_user}_{date_str}.csv",
            mime="text/csv",
            key="export_recs_csv_btn"
        )


def render_user_feedback(system):
    """
    Renders Feedback Center inside User Portal. Allows filtering
    their feedback history by Date and Action.
    """
    current_user = st.session_state.get("current_user")
    
    if not current_user or current_user == "ADMIN":
        st.warning("You are currently acting in Admin Context. Select a user profile to manage feedback history.")
        return

    st.markdown('<div class="gradient-header">Feedback History</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Manage your historical acceptances, rejections, and view matching records</div>', unsafe_allow_html=True)

    feedback_df = get_feedback_for_user(current_user)
    
    if feedback_df.empty:
        st.info("No feedback has been logged yet for your profile.")
        return

    # Filters
    st.markdown("### 🔍 Filter History")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f_action = st.selectbox("Action", ["All", "Accept", "Reject"])
    with col_f2:
        f_dates = st.date_input("Date Range", [])

    filtered_df = feedback_df.copy()
    if f_action == "Accept":
        filtered_df = filtered_df[filtered_df["action"] == 1]
    elif f_action == "Reject":
        filtered_df = filtered_df[filtered_df["action"] == 0]

    if len(f_dates) == 2:
        start_date, end_date = pd.to_datetime(f_dates[0]), pd.to_datetime(f_dates[1])
        filtered_df["datetime"] = pd.to_datetime(filtered_df["timestamp"])
        filtered_df = filtered_df[(filtered_df["datetime"].dt.date >= start_date.date()) & (filtered_df["datetime"].dt.date <= end_date.date())]
        filtered_df = filtered_df.drop(columns=["datetime"])

    st.markdown(f"**Found {len(filtered_df)} logged feedback entries:**")
    if filtered_df.empty:
        st.info("No logs match the selected filters.")
    else:
        display_df = filtered_df.rename(columns={
            "user_id": "User ID",
            "matched_user_id": "Matched User ID",
            "action": "Action Code",
            "timestamp": "Logged At"
        }).copy()
        display_df["Action"] = display_df["Action Code"].map({1: "Accept", 0: "Reject"})
        # Reorder columns
        display_df = display_df[["User ID", "Matched User ID", "Action", "Logged At"]]
        st.dataframe(display_df, width='stretch', hide_index=True)


def render_user_history(system):
    """
    Renders the recommendation batches log history generated for the user profile.
    """
    current_user = st.session_state.get("current_user")
    
    if not current_user or current_user == "ADMIN":
        st.warning("You are currently acting in Admin Context. Select a user profile to manage batch logs.")
        return

    st.markdown('<div class="gradient-header">Recommendation History</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Browse previous recommendation snapshots generated by the system</div>', unsafe_allow_html=True)

    history_df = load_recommendation_history()
    user_history = history_df[history_df["user_id"] == current_user]

    if user_history.empty:
        st.info("You haven't generated any recommendations batches yet. Go to My Recommendations to generate one.")
        return

    batches = user_history["recommendation_batch_id"].unique().tolist()
    st.markdown(f"Total recommendation batches generated: **{len(batches)}**")

    # Select batch to inspect
    selected_batch = st.selectbox("Select Batch ID to view details", batches)
    batch_records = user_history[user_history["recommendation_batch_id"] == selected_batch]

    display_df = batch_records.rename(columns={
        "recommendation_batch_id": "Batch ID",
        "recommended_user_id": "Recommended User ID",
        "score": "ML Score",
        "timestamp": "Batch Date"
    })
    
    st.dataframe(display_df[["Batch ID", "Recommended User ID", "ML Score", "Batch Date"]], width='stretch', hide_index=True)
