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
    page_title="Match Explainability | NexMatch AI",
    page_icon="📢",
    layout="wide"
)

# Load CSS
load_css()

# Load system objects
system = load_system()
users_df = system["users_df"]
recommender = system["recommender"]

# Page titles
st.markdown('<div class="gradient-header">Recommendation Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Deep-dive comparison: Analyze matching compatibility dimensions between any two user profiles</div>', unsafe_allow_html=True)

# Selection controls
st.markdown("### 🔍 Select User Pair")
col_sel_1, col_sel_2 = st.columns(2)

user_list = users_df["user_id"].tolist()

# Default user A selection
default_user_a = st.session_state.get("selected_user", user_list[0])
if default_user_a not in user_list:
    default_user_a = user_list[0]

with col_sel_1:
    user_a_id = st.selectbox("Select User A (Source Profile)", user_list, index=user_list.index(default_user_a))
    # Update global selection
    st.session_state["selected_user"] = user_a_id

# Remove User A from options for User B
user_list_b = [uid for uid in user_list if uid != user_a_id]
default_b_idx = 0
# If they previously selected a user B, let's keep it
if "selected_user_b" in st.session_state and st.session_state["selected_user_b"] in user_list_b:
    default_b_idx = user_list_b.index(st.session_state["selected_user_b"])

with col_sel_2:
    user_b_id = st.selectbox("Select User B (Target Profile)", user_list_b, index=default_b_idx)
    st.session_state["selected_user_b"] = user_b_id

# Retrieve profiles
user_a = users_df[users_df["user_id"] == user_a_id].iloc[0]
user_b = users_df[users_df["user_id"] == user_b_id].iloc[0]

# Side by side profiles
st.markdown("### 👥 Profile Comparison")
col_p1, col_p2 = st.columns(2)

def render_compact_card(row, title_prefix):
    skills_list = [s.strip() for s in str(row["skills"]).split(",") if s.strip()]
    skills_badges = "".join([f'<span class="badge badge-blue">{s}</span>' for s in skills_list])
    return f"""
    <div class="glass-card" style="min-height: 250px;">
        <h4 style="color: #60a5fa; margin-top:0;">{title_prefix}: {row['name']}</h4>
        <div style="font-size: 0.85rem; color: #a855f7; font-weight: bold; margin-bottom: 10px;">ID: {row['user_id']} | {row['profession']}</div>
        <div class="profile-item"><span class="profile-label" style="width: 100px;">📍 Location:</span><span class="profile-value">{row['location']}</span></div>
        <div class="profile-item"><span class="profile-label" style="width: 100px;">💼 Experience:</span><span class="profile-value">{row['experience_years']} years</span></div>
        <div class="profile-item"><span class="profile-label" style="width: 100px;">🧬 MBTI:</span><span class="badge badge-green" style="margin:0;">{row['mbti']}</span></div>
        <div class="profile-item"><span class="profile-label" style="width: 100px;">🎯 Career Goal:</span><span class="profile-value">{row['career_goal']}</span></div>
        <div style="margin-top: 10px;"><b>Skills:</b> {skills_badges}</div>
    </div>
    """

with col_p1:
    st.markdown(render_compact_card(user_a, "User A"), unsafe_allow_html=True)
    
with col_p2:
    st.markdown(render_compact_card(user_b, "User B"), unsafe_allow_html=True)

# Calculate compatibility score
score = recommender.compatibility_score(user_a_id, user_b_id)

st.markdown("### 📊 Matching Dimension Breakdown")
col_b1, col_b2 = st.columns([1, 1])

with col_b1:
    def render_explain_bar(label, key):
        val = score[key]
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px; font-size: 0.85rem;">
            <span style="color: #cbd5e1;">{label}</span>
            <span style="font-weight: bold;">{val}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(val / 100.0)

    render_explain_bar("📝 Profile Text Similarity (TF-IDF Cosine)", "text_similarity")
    render_explain_bar("🧬 MBTI Compatibility", "mbti_score")
    render_explain_bar("💼 Profession Match", "profession_score")
    render_explain_bar("🎯 Career Goal Match", "career_goal_score")

with col_b2:
    render_explain_bar("🛠️ Skills Match", "skills_score")
    render_explain_bar("📈 Experience Level Match", "experience_score")
    render_explain_bar("📍 Location Match", "location_score")
    render_explain_bar("🤝 Networking Intent Match", "networking_intent_score")

st.markdown("<br/>", unsafe_allow_html=True)

# Generate Dynamic Explanation Paragraph
def generate_dynamic_reasoning(row_a, row_b, score_dict):
    reasons = []
    
    # 1. Profession
    if score_dict["profession_score"] == 100:
        reasons.append("share the exact same profession of " + row_a["profession"])
    elif score_dict["profession_score"] >= 70:
        reasons.append(f"work in closely related domains ({row_a['profession']} & {row_b['profession']})")
        
    # 2. Skills
    skills_a = set(s.strip().lower() for s in str(row_a["skills"]).split(",") if s.strip())
    skills_b = set(s.strip().lower() for s in str(row_b["skills"]).split(",") if s.strip())
    common = skills_a & skills_b
    if common:
        reasons.append(f"have overlapping technical expertise in {', '.join([s.title() for s in list(common)[:3]])}")
        
    # 3. MBTI
    if score_dict["mbti_score"] >= 80:
        reasons.append(f"possess highly compatible psychological types (MBTI: {row_a['mbti']} and {row_b['mbti']})")
    elif score_dict["mbti_score"] >= 50:
        reasons.append(f"have compatible communication styles (MBTI: {row_a['mbti']} and {row_b['mbti']})")
        
    # 4. Career Goal
    if score_dict["career_goal_score"] >= 70:
        reasons.append(f"are both focused on {row_a['career_goal']} career pathways")
        
    # 5. Experience
    exp_diff = abs(row_a["experience_years"] - row_b["experience_years"])
    if exp_diff <= 2:
        reasons.append("possess similar levels of industry seniority")
    elif exp_diff <= 5:
        reasons.append("are at complementary stages of their career paths")
        
    # 6. Location
    if score_dict["location_score"] == 100:
        reasons.append(f"are both located in {row_a['location']}, facilitating in-person collaboration")

    # Combine into a sentence
    if not reasons:
        return "This matching shows mild peripheral alignment across several networking factors."
        
    if len(reasons) == 1:
        return f"This recommendation is driven primarily because both users {reasons[0]}."
        
    intro = "This recommendation is strong because both users "
    body = ", ".join(reasons[:-1]) + ", and " + reasons[-1]
    return intro + body + "."

dynamic_paragraph = generate_dynamic_reasoning(user_a, user_b, score)

st.markdown(f"""
<div class="glass-card" style="border-left: 4px solid #10b981; margin-top: 20px;">
    <h3 style="color: #34d399; margin: 0 0 10px;">📢 Dynamic Match Explanation</h3>
    <p style="color: #f1f5f9; font-size: 1.05rem; line-height: 1.6; margin: 0;">
        "{dynamic_paragraph}"
    </p>
    <div style="margin-top: 15px; font-weight: bold; color: #10b981; font-size: 1.1rem; text-align: right;">
        Overall Match Score: {score['final_score']:.2f}%
    </div>
</div>
""", unsafe_allow_html=True)
