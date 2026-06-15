import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.loader import load_system
from src.utils.styles import load_css

# Page Configuration
st.set_page_config(
    page_title="Hybrid Recommender | NexMatch AI",
    page_icon="⚖️",
    layout="wide"
)

# Load CSS
load_css()

# Load system objects
system = load_system()
users_df = system["users_df"]
recommender = system["recommender"]

# Page titles
st.markdown('<div class="gradient-header">Hybrid Recommender Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Match professional profiles using vector similarity combined with demographic heuristics</div>', unsafe_allow_html=True)

# Select Target User
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = users_df["user_id"].iloc[0]

# Sidebar user selection to synchronize state
st.sidebar.markdown("### 👤 Select Active Profile")
user_list = users_df["user_id"].tolist()
default_idx = user_list.index(st.session_state["selected_user"]) if st.session_state["selected_user"] in user_list else 0

selected_user_id = st.sidebar.selectbox("Active User ID", user_list, index=default_idx)
st.session_state["selected_user"] = selected_user_id

# Target Profile Details Header
user_row = users_df[users_df["user_id"] == selected_user_id].iloc[0]

st.markdown(f"""
<div class="glass-card" style="border-left: 4px solid #a855f7;">
    <h3 style="margin-bottom: 5px;">Active Profile: {user_row['name']}</h3>
    <p style="margin: 0; color: #94a3b8; font-size: 0.95rem;">
        ID: <b>{selected_user_id}</b> | Profession: <b>{user_row['profession']}</b> | Location: <b>{user_row['location']}</b> | 
        Experience: <b>{user_row['experience_years']} years</b> | MBTI: <b>{user_row['mbti']}</b> | Intent: <b>{user_row['networking_intent']}</b>
    </p>
</div>
""", unsafe_allow_html=True)

# Button to trigger recommendation
generate_btn = st.button("⚡ Generate Hybrid Recommendations", type="primary")

# Persist and compute recommendations using session state
if generate_btn or "last_recs" in st.session_state:
    if generate_btn:
        with st.spinner("Executing Stage 1 retrieval and matching weights..."):
            recs = recommender.get_top_recommendations(selected_user_id, top_n=5)
            st.session_state["last_recs"] = recs
            st.session_state["last_rec_user"] = selected_user_id
    else:
        # If user changed from outside or dropdown, re-generate automatically
        if st.session_state.get("last_rec_user") != selected_user_id:
            with st.spinner("Generating recommendations for new profile..."):
                recs = recommender.get_top_recommendations(selected_user_id, top_n=5)
                st.session_state["last_recs"] = recs
                st.session_state["last_rec_user"] = selected_user_id
        else:
            recs = st.session_state["last_recs"]
            
    if recs:
        st.markdown("### 🏆 Top 5 Recommendations")
        
        # Display recommendations in responsive columns
        cols = st.columns(5)
        for idx, rec in enumerate(recs):
            target_id = rec["user_id"]
            target_name = users_df[users_df["user_id"] == target_id]["name"].iloc[0]
            with cols[idx]:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; border-color: rgba(59, 130, 246, 0.15); min-height: 260px;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1.1rem; margin-bottom: 5px;">{target_name}</div>
                    <div style="font-size: 0.75rem; color: #a855f7; margin-bottom: 10px;">ID: {target_id}</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500; min-height: 40px; display: flex; align-items: center; justify-content: center;">{rec["profession"]}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 15px;">📍 {rec["location"]} | 💼 {rec["experience"]} yrs</div>
                    <div style="background: rgba(168, 85, 247, 0.1); border-radius: 8px; padding: 6px 10px; display: inline-block;">
                        <span style="font-weight: 800; color: #c084fc; font-size: 1.1rem;">{rec["final_score"]:.2f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Selectbox to see breakdowns
        rec_ids = [r["user_id"] for r in recs]
        rec_labels = [f"{users_df[users_df['user_id'] == rid]['name'].iloc[0]} ({rid}) - Match: {r['final_score']:.2f}%" for rid, r in zip(rec_ids, recs)]
        
        st.markdown("### 📊 Detail Match Compatibility Breakdown")
        selected_rec_label = st.selectbox("Select profile to inspect breakdown details", rec_labels)
        selected_rec_id = rec_ids[rec_labels.index(selected_rec_label)]
        
        # Calculate compatibility breakdown
        score_details = recommender.compatibility_score(selected_user_id, selected_rec_id)
        rec_row = users_df[users_df["user_id"] == selected_rec_id].iloc[0]
        
        col_b1, col_b2 = st.columns([4, 3])
        
        with col_b1:
            st.markdown("#### Compatibility Dimensions")
            
            def render_factor_bar(label, key, color_class=""):
                val = score_details[key]
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 2px; font-size: 0.85rem;">
                    <span style="color: #cbd5e1;">{label}</span>
                    <span style="font-weight: bold;">{val}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(val / 100.0)
                
            render_factor_bar("📝 Profile Text Similarity (TF-IDF Cosine)", "text_similarity")
            render_factor_bar("🧬 MBTI Compatibility", "mbti_score")
            render_factor_bar("💼 Profession Field Match", "profession_score")
            render_factor_bar("🎯 Career Goal Aligned", "career_goal_score")
            render_factor_bar("🛠️ Technical Skills Intersection", "skills_score")
            render_factor_bar("🤝 Networking Intent Match", "networking_intent_score")
            render_factor_bar("📈 Experience Delta Match", "experience_score")
            render_factor_bar("📍 Geographic Location Match", "location_score")
            
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 12px; margin-top: 15px; text-align: center;">
                <span style="color: #34d399; font-weight: bold; font-size: 1.1rem;">Final Compatibility Score: {score_details['final_score']:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        with col_b2:
            st.markdown("#### 📢 Match Reasons & Explanations")
            
            # Generate reasons
            reasons = []
            if score_details["profession_score"] == 100:
                reasons.append("✓ **Identical Profession**: Both are in the exact same field, sharing direct professional context.")
            elif score_details["profession_score"] >= 70:
                reasons.append("✓ **Related Domain**: Operating in adjacent/related technology or business areas.")
                
            if score_details["skills_score"] >= 50:
                reasons.append("✓ **Strong Skill Overlap**: Significant matching skillsets for peer-to-peer discussions.")
            elif score_details["skills_score"] > 10:
                reasons.append("✓ **Complementary Skills**: Share similar technical languages and development tool sets.")
                
            if score_details["experience_score"] >= 80:
                reasons.append(f"✓ **Balanced Career Levels**: Both possess comparable seniority ({user_row['experience_years']} yrs vs {rec_row['experience_years']} yrs).")
                
            if score_details["mbti_score"] >= 85:
                reasons.append(f"✓ **Highly Compatible Personalities**: MBTI matching shows strong relationship compatibility ({user_row['mbti']} ↔ {rec_row['mbti']}).")
            elif score_details["mbti_score"] >= 60:
                reasons.append(f"✓ **Good MBTI Compatibility**: Personality traits support collaborative communication ({user_row['mbti']} ↔ {rec_row['mbti']}).")
                
            if score_details["career_goal_score"] >= 70:
                reasons.append(f"✓ **Aligned Goals**: Both share a career interest or trajectory direction ({user_row['career_goal']}).")
                
            if score_details["location_score"] == 100:
                reasons.append(f"✓ **Same Location**: Local connection in {user_row['location']} allows easy networking or face-to-face meetups.")
                
            if score_details["networking_intent_score"] >= 70:
                reasons.append(f"✓ **Symmetric Intentions**: Complementary networking goals ({user_row['networking_intent']} ↔ {rec_row['networking_intent']}).")
                
            if not reasons:
                reasons.append("• Minor alignment across multiple peripheral matching categories.")

            reasons_html = "".join([f"<li style='margin-bottom: 10px; color: #e2e8f0; font-size: 0.9rem;'>{r}</li>" for r in reasons])
            
            st.markdown(f"""
            <div class="glass-card" style="border-color: rgba(168, 85, 247, 0.2);">
                <h3 style="color: #a855f7; margin-bottom: 15px;">Why this match makes sense</h3>
                <ul style="padding-left: 15px; margin-top: 0;">
                    {reasons_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("No recommendations could be generated.")
else:
    st.info("Click the **Generate Hybrid Recommendations** button to run the matching engine pipeline.")
