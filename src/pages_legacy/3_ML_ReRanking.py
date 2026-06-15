import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.loader import load_system
from src.utils.loader import refresh_feedback_views
from src.utils.styles import load_css
from src.utils.data_manager import append_feedback

# Page Configuration
st.set_page_config(
    page_title="ML Re-Ranking | NexMatch AI",
    page_icon="🤖",
    layout="wide"
)

# Load CSS
load_css()

# Load system objects
system = load_system()
users_df = system["users_df"]
adaptive = system["adaptive"]

# Page titles
st.markdown('<div class="gradient-header">My Recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Stable recommendations powered by the latest deployed model. Feedback is saved instantly and only affects rankings after an admin retrains.</div>', unsafe_allow_html=True)

# Select Active User
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = users_df["user_id"].iloc[0]

st.sidebar.markdown("### 👤 Select Active Profile")
user_list = users_df["user_id"].tolist()
default_idx = user_list.index(st.session_state["selected_user"]) if st.session_state["selected_user"] in user_list else 0

selected_user_id = st.sidebar.selectbox("Active User ID", user_list, index=default_idx)
st.session_state["selected_user"] = selected_user_id

# Display User Brief Card
user_row = users_df[users_df["user_id"] == selected_user_id].iloc[0]
st.markdown(f"""
<div class="glass-card" style="border-left: 4px solid #3b82f6;">
    <h3 style="margin-bottom: 5px;">Active Profile: {user_row['name']}</h3>
    <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
        Profession: <b>{user_row['profession']}</b> | Location: <b>{user_row['location']}</b> | 
        Experience: <b>{user_row['experience_years']} years</b> | MBTI: <b>{user_row['mbti']}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# Ranking Architecture Formula Display
st.markdown("### ⚖️ Final Scoring Formula")
st.markdown("""
To achieve high-quality matches, we combine static matching similarity and predictive conversion likelihood.
""", unsafe_allow_html=True)
st.markdown("""
<div class="formula-box">
    Final Ranking Score = <span class="formula-highlight">0.60 × Hybrid Score</span> + <span class="formula-highlight">0.40 × ML Score (Acceptance Prob)</span>
</div>
""", unsafe_allow_html=True)

session_recs = st.session_state.get("recommendation_snapshots", {})
snapshot = session_recs.get(selected_user_id)
if snapshot is None:
    with st.spinner("Processing deployed recommendation pipeline..."):
        snapshot = {
            "candidate_pool": adaptive.get_candidate_pool(selected_user_id, pool_size=30),
            "final_recs": adaptive.get_top_recommendations(selected_user_id, top_n=5),
        }
    session_recs[selected_user_id] = snapshot
    st.session_state["recommendation_snapshots"] = session_recs

refresh_col1, refresh_col2 = st.columns([3, 1])
with refresh_col1:
    st.info("Recommendation cards stay fixed after Accept/Reject clicks. Use refresh only when you want to pull a new snapshot from the currently deployed model.")
with refresh_col2:
    if st.button("Refresh Snapshot", type="primary"):
        with st.spinner("Refreshing recommendations from the deployed model..."):
            session_recs[selected_user_id] = {
                "candidate_pool": adaptive.get_candidate_pool(selected_user_id, pool_size=30),
                "final_recs": adaptive.get_top_recommendations(selected_user_id, top_n=5),
            }
            st.session_state["recommendation_snapshots"] = session_recs
        st.rerun()

candidate_pool = snapshot["candidate_pool"]
final_recs = snapshot["final_recs"]

# Columns for Stage 1 and Stage 2 side-by-side
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎯 Stage 1: Candidate Generation (Retrieval)")
    st.markdown(f"Retrieved top **{len(candidate_pool)}** high-potential candidate profiles from the network database.")
    
    # Create DataFrame for display
    pool_rows = []
    for cand in candidate_pool:
        pool_rows.append({
            "User ID": cand["user_id"],
            "Profession": cand["profession"],
            "Hybrid Score (%)": f"{cand['final_score']:.2f}%"
        })
    pool_df = pd.DataFrame(pool_rows)
    
    st.dataframe(pool_df, height=380, use_container_width=True)

with col2:
    st.markdown("### 🤖 Stage 2: ML Re-Ranking (Scoring)")
    st.markdown("Top **5** recommendations sorted by prediction acceptance score from the feedback learner.")
    
    # Create DataFrame for display
    rec_rows = []
    for match in final_recs:
        rec_rows.append({
            "User ID": match["user_id"],
            "Profession": match["profession"],
            "Hybrid (%)": f"{match['hybrid_score']:.2f}%",
            "ML Prob (%)": f"{match['ml_score']:.2f}%",
            "Final Score (%)": f"{match['final_ranking_score']:.2f}%"
        })
    rec_df = pd.DataFrame(rec_rows)
    
    st.dataframe(rec_df, height=220, use_container_width=True)
    
    # Show highest conversion match badge
    if final_recs:
        best_rec = final_recs[0]
        best_name = users_df[users_df["user_id"] == best_rec["user_id"]]["name"].iloc[0]
        st.markdown(f"""
        <div class="glass-card" style="border-color: rgba(16, 185, 129, 0.3); text-align: center; margin-top: 20px; padding: 15px;">
            <h4 style="color: #34d399; margin: 0 0 5px;">🔥 Top Conversion Match</h4>
            <span style="font-size: 1.1rem; font-weight: bold; color: #f8fafc;">{best_name} ({best_rec['user_id']})</span><br/>
            <span style="color: #94a3b8; font-size: 0.85rem;">{best_rec['profession']}</span><br/>
            <span style="font-weight: 800; color: #34d399; font-size: 1.25rem; margin-top: 5px; display: inline-block;">Final Score: {best_rec['final_ranking_score']:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("### Feedback Loop")
st.markdown("Accept/Reject actions are appended to `feedback.csv` immediately. They do not retrain the model or reshuffle this page until an admin manually retrains.")

for idx, match in enumerate(final_recs):
    target_name = users_df[users_df["user_id"] == match["user_id"]]["name"].iloc[0]
    card_col, action_col1, action_col2 = st.columns([4, 1, 1])
    with card_col:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px; margin-bottom: 8px;">
                <b>{target_name}</b> ({match['user_id']})<br/>
                <span style="color: #94a3b8;">{match['profession']} | Final Score: {match['final_ranking_score']:.2f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with action_col1:
        accepted = st.button("Accept", key=f"accept_{selected_user_id}_{match['user_id']}_{idx}")
    with action_col2:
        rejected = st.button("Reject", key=f"reject_{selected_user_id}_{match['user_id']}_{idx}")

    if accepted:
        append_feedback(selected_user_id, match["user_id"], 1)
        refresh_feedback_views()
        st.success(f"Saved accept feedback for {selected_user_id} -> {match['user_id']}.")
        st.rerun()

    if rejected:
        append_feedback(selected_user_id, match["user_id"], 0)
        refresh_feedback_views()
        st.success(f"Saved reject feedback for {selected_user_id} -> {match['user_id']}.")
        st.rerun()

# Explanation panel about why this architecture is standard in production
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card" style="background: rgba(255, 255, 255, 0.01); border-color: rgba(255, 255, 255, 0.05);">
    <h3 style="color: #60a5fa; margin-bottom: 15px;">💡 Industry System Design Insight: Two-Stage Recommendation Architecture</h3>
    <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        NexMatch AI implements a standard industry <b>two-stage recommendation system</b> (similar to architectures deployed at Netflix, YouTube, and LinkedIn):
    </p>
    <ol style="padding-left: 20px; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
        <li style="margin-bottom: 8px;">
            <b>Stage 1: Candidate Generation (Retrieval)</b> - Scans the entire corpus of users using fast vector indexing (TF-IDF cosine similarity) and hard constraints (Profession/Location/MBTI) to filter out millions of unrelated items down to a small, high-potential candidate pool (e.g. top 30 candidates). This stage solves <i>scalability and latency</i>.
        </li>
        <li>
            <b>Stage 2: ML Re-Ranking (Scoring)</b> - Feeds the candidate pool into a heavy Machine Learning classifier (Logistic Regression) trained on historical feedback (Accept/Reject actions). The classifier uses feature intersections as inputs to predict the precise conversion probability (ML Score). The hybrid similarity and ML prediction are fused for final sorting. This stage solves <i>precision and engagement</i>.
        </li>
    </ol>
</div>
""", unsafe_allow_html=True)
