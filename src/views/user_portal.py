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
    NETWORKING_INTENT_OPTIONS
)

@st.dialog("Match Compatibility Details")
def show_match_details_dialog(recommender, users_df, user_id, target_id):
    """
    Displays a premium pop-up modal showing compatibility score breakdowns
    and natural language reasoning for a recommended match.
    """
    score = recommender.compatibility_score(user_id, target_id)
    target_row = users_df[users_df["user_id"] == target_id].iloc[0]
    user_row = users_df[users_df["user_id"] == user_id].iloc[0]
    
    st.markdown(f"### 🤝 Analysis for {target_row['name']}")
    st.markdown(f"**Profession:** {target_row['profession']} | **Location:** {target_row['location']}")
    st.markdown("---")
    
    # Render Dimensions
    st.markdown("#### 📊 Compatibility Scores")
    
    def render_explain_bar(label, key):
        val = score[key]
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px; font-size: 0.85rem;">
            <span style="color: #cbd5e1;">{label}</span>
            <span style="font-weight: bold;">{val:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(val / 100.0)

    render_explain_bar("📝 Profile Text Similarity (TF-IDF Cosine)", "text_similarity")
    render_explain_bar("🧬 MBTI Compatibility", "mbti_score")
    render_explain_bar("💼 Profession Match", "profession_score")
    render_explain_bar("🎯 Career Goal Match", "career_goal_score")
    render_explain_bar("🛠️ Skills Match", "skills_score")
    render_explain_bar("📈 Experience Level Match", "experience_score")
    render_explain_bar("📍 Location Match", "location_score")
    render_explain_bar("🤝 Networking Intent Match", "networking_intent_score")
    
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 15px;">
        <span style="color: #34d399; font-weight: bold; font-size: 1.1rem;">Overall Combined Match Score: {score['final_score']:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic Reasoning Paragraph
    reasons = []
    if score["profession_score"] == 100:
        reasons.append("share the exact same profession of " + user_row["profession"])
    elif score["profession_score"] >= 70:
        reasons.append(f"work in closely related domains ({user_row['profession']} & {target_row['profession']})")
        
    skills_a = set(s.strip().lower() for s in str(user_row["skills"]).split(",") if s.strip())
    skills_b = set(s.strip().lower() for s in str(target_row["skills"]).split(",") if s.strip())
    common = skills_a & skills_b
    if common:
        reasons.append(f"have overlapping technical expertise in {', '.join([s.title() for s in list(common)[:3]])}")
        
    if score["mbti_score"] >= 80:
        reasons.append(f"possess highly compatible psychological types (MBTI: {user_row['mbti']} and {target_row['mbti']})")
    elif score["mbti_score"] >= 50:
        reasons.append(f"have compatible communication styles (MBTI: {user_row['mbti']} and {target_row['mbti']})")
        
    if score["career_goal_score"] >= 70:
        reasons.append(f"are both focused on {user_row['career_goal']} career pathways")
        
    exp_diff = abs(user_row["experience_years"] - target_row["experience_years"])
    if exp_diff <= 2:
        reasons.append("possess similar levels of industry seniority")
    elif exp_diff <= 5:
        reasons.append("are at complementary stages of their career paths")
        
    if score["location_score"] == 100:
        reasons.append(f"are both located in {user_row['location']}, facilitating local in-person meetups")

    if not reasons:
        dynamic_paragraph = "This matching shows mild peripheral alignment across several networking factors."
    elif len(reasons) == 1:
        dynamic_paragraph = f"This recommendation is driven primarily because both users {reasons[0]}."
    else:
        dynamic_paragraph = "This recommendation is strong because both users " + ", ".join(reasons[:-1]) + ", and " + reasons[-1] + "."

    st.markdown("#### 📢 Match Reason")
    st.info(dynamic_paragraph)


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
    
    acc_rate = (accepted / (accepted + rejected) * 100) if (accepted + rejected) > 0 else 0.0
    
    last_rec_date = "N/A"
    if not user_history.empty:
        last_rec_date = pd.to_datetime(user_history["timestamp"]).max().strftime("%Y-%m-%d %H:%M")
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Matches Seen", f"{total_viewed:,}")
    with m_col2:
        st.metric("Accepted", f"{accepted:,}")
    with m_col3:
        st.metric("Rejected", f"{rejected:,}")
    with m_col4:
        st.metric("Accept Rate", f"{acc_rate:.1f}%")
    with m_col5:
        st.metric("Last Match Batch", last_rec_date)

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
        st.dataframe(display_df, width="stretch", hide_index=True)
        
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
    and save their own profile details.
    """
    users_df = system["users_df"]
    current_user = st.session_state.get("current_user")
    
    if not current_user or current_user == "ADMIN":
        st.warning("You are currently acting in Admin Context. Select a user profile to manage.")
        return

    st.markdown('<div class="gradient-header">My Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Manage your professional information, skills, and networking criteria</div>', unsafe_allow_html=True)

    user_row = users_df[users_df["user_id"] == current_user].iloc[0]

    st.markdown("### Profile Settings")
    with st.form("edit_profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=str(user_row.get("name", "")))
            profession = st.text_input("Profession", value=str(user_row.get("profession", "")))
            location = st.text_input("Location", value=str(user_row.get("location", "")))
            experience_years = st.number_input("Experience Years", min_value=0, max_value=50, value=int(user_row.get("experience_years", 0)))
            mbti = st.selectbox("MBTI Type", MBTI_OPTIONS, index=MBTI_OPTIONS.index(user_row.get("mbti", "INTJ")) if user_row.get("mbti") in MBTI_OPTIONS else 0)
            career_goal = st.text_input("Career Goal", value=str(user_row.get("career_goal", "")))
            
        with col2:
            networking_intent = st.selectbox("Networking Intent", NETWORKING_INTENT_OPTIONS, index=NETWORKING_INTENT_OPTIONS.index(user_row.get("networking_intent", "Career Growth")) if user_row.get("networking_intent") in NETWORKING_INTENT_OPTIONS else 0)
            skills = st.text_area("Skills (comma separated)", value=str(user_row.get("skills", "")))
            interests = st.text_area("Interests (comma separated)", value=str(user_row.get("interests", "")))
            professional_summary = st.text_area("Professional Summary", value=str(user_row.get("professional_summary", "")))
            about_me = st.text_area("About Me", value=str(user_row.get("about_me", "")))
            
        submitted = st.form_submit_button("Update Profile", type="primary")

    if submitted:
        if not name.strip() or not profession.strip() or not location.strip() or not professional_summary.strip() or not about_me.strip():
            st.error("Please fill in all required fields (Name, Profession, Location, Summary, About Me).")
        else:
            updated_profile = {
                "name": name.strip(),
                "profession": profession.strip(),
                "location": location.strip(),
                "experience_years": int(experience_years),
                "mbti": mbti,
                "career_goal": career_goal.strip(),
                "skills": skills.strip(),
                "interests": interests.strip(),
                "networking_intent": networking_intent,
                "professional_summary": professional_summary.strip(),
                "about_me": about_me.strip(),
            }
            update_user_profile(current_user, updated_profile)
            clear_system_caches()
            st.success("Your profile has been updated successfully!")
            st.rerun()


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

    # Render Cards
    for idx, rec in enumerate(recs):
        target_id = rec["user_id"]
        
        # Build Badges
        skills_list = [s.strip() for s in str(rec.get("skills", "")).split(",") if s.strip()]
        if not skills_list:
            # fallback loading from users_df
            match_row = users_df[users_df["user_id"] == target_id]
            if not match_row.empty:
                skills_list = [s.strip() for s in str(match_row.iloc[0]["skills"]).split(",") if s.strip()]
        
        skills_badges = " ".join([f'<span class="badge badge-blue">{s}</span>' for s in skills_list[:4]])
        
        card_col, action_col = st.columns([3, 1])
        with card_col:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #3b82f6; margin-bottom: 0px;">
                <h3 style="margin-bottom: 5px; color: #f8fafc;">{users_df[users_df['user_id'] == target_id]['name'].iloc[0]}</h3>
                <p style="margin: 0; color: #a855f7; font-weight: 600;">{rec['profession']}</p>
                <p style="margin: 5px 0 0; color: #cbd5e1; font-size: 0.9rem;">📍 {rec['location']} | 🎯 Career Goal: {rec['career_goal']}</p>
                <p style="margin: 2px 0 5px; color: #94a3b8; font-size: 0.85rem;">🤝 Intent: {rec.get('networking_intent', users_df[users_df['user_id'] == target_id]['networking_intent'].iloc[0])}</p>
                <div style="margin-top: 10px; margin-bottom: 10px;">
                    {skills_badges}
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); border-radius: 6px; padding: 4px 10px; display: inline-block;">
                    <span style="font-weight: bold; color: #34d399; font-size: 0.95rem;">Match: {rec['final_ranking_score']:.2f}%</span>
                    <span style="color: #64748b; font-size: 0.75rem; margin-left: 5px;">(Hybrid: {rec['hybrid_score']:.1f}% | ML: {rec['ml_score']:.1f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with action_col:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            # Match Details dialog button
            if st.button("🔍 Match Details", key=f"details_btn_{target_id}_{idx}"):
                show_match_details_dialog(recommender, users_df, current_user, target_id)
                
            # Radio feedback choices
            current_selection = selections.get(target_id, "Skip")
            sel_options = ["Skip", "Accept", "Reject"]
            choice = st.radio(
                "Feedback Selection",
                sel_options,
                index=sel_options.index(current_selection),
                key=f"feedback_select_{target_id}_{idx}"
            )
            selections[target_id] = choice
            
        st.markdown("<br/>", unsafe_allow_html=True)

    snapshot["selections"] = selections
    snapshots[current_user] = snapshot
    st.session_state["recommendation_snapshots"] = snapshots

    st.markdown("---")
    
    # Bottom Submit and Export Buttons
    col_sub1, col_sub2 = st.columns([2, 2])
    with col_sub1:
        if st.button("📤 Submit Batch Feedback", type="primary", key="batch_feedback_submit_btn"):
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
                refresh_feedback_views()
                st.success(f"Successfully submitted batch feedback ({len(feedback_rows)} rows)!")
                # Reset selections in snapshot
                snapshot["selections"] = {}
                snapshots[current_user] = snapshot
                st.session_state["recommendation_snapshots"] = snapshots
                st.rerun()
            else:
                st.warning("No feedback items selected. Change selections to Accept or Reject to submit feedback.")
                
    with col_sub2:
        # Export Recommendations CSV
        export_rows = []
        for rec in recs:
            export_rows.append({
                "User ID": rec["user_id"],
                "Profession": rec["profession"],
                "Location": rec["location"],
                "Career Goal": rec["career_goal"],
                "Hybrid Score": rec["hybrid_score"],
                "ML Score": rec["ml_score"],
                "Final Ranking Score": rec["final_ranking_score"]
            })
        rec_export_df = pd.DataFrame(export_rows)
        csv_recs = rec_export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Recommendations CSV",
            data=csv_recs,
            file_name=f"nexmatch_recommendations_{snapshot['batch_id']}.csv",
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
        st.dataframe(display_df, width="stretch", hide_index=True)


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
    
    st.dataframe(display_df[["Batch ID", "Recommended User ID", "ML Score", "Batch Date"]], width="stretch", hide_index=True)
