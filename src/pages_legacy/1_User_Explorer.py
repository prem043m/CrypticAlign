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
    page_title="User Explorer | NexMatch AI",
    page_icon="👤",
    layout="wide"
)

# Load CSS
load_css()

# Load system objects
system = load_system()
users_df = system["users_df"]

# Page titles
st.markdown('<div class="gradient-header">User Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Browse, filter, and inspect detailed user profiles in the network</div>', unsafe_allow_html=True)

# Sidebar Filters
st.sidebar.markdown("### 🔍 Filter Profiles")

# Search Input
search_query = st.sidebar.text_input("Search by Name or ID", "")

# Dropdown Filters
professions = ["All"] + sorted(list(users_df["profession"].dropna().unique()))
selected_prof = st.sidebar.selectbox("Profession", professions)

locations = ["All"] + sorted(list(users_df["location"].dropna().unique()))
selected_loc = st.sidebar.selectbox("Location", locations)

mbtis = ["All"] + sorted(list(users_df["mbti"].dropna().unique()))
selected_mbti = st.sidebar.selectbox("MBTI Type", mbtis)

career_goals = ["All"] + sorted(list(users_df["career_goal"].dropna().unique()))
selected_goal = st.sidebar.selectbox("Career Goal", career_goals)

# Apply filters
filtered_df = users_df.copy()

if search_query:
    filtered_df = filtered_df[
        filtered_df["name"].str.contains(search_query, case=False, na=False) |
        filtered_df["user_id"].str.contains(search_query, case=False, na=False)
    ]

if selected_prof != "All":
    filtered_df = filtered_df[filtered_df["profession"] == selected_prof]

if selected_loc != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_loc]

if selected_mbti != "All":
    filtered_df = filtered_df[filtered_df["mbti"] == selected_mbti]

if selected_goal != "All":
    filtered_df = filtered_df[filtered_df["career_goal"] == selected_goal]

# Handle empty filter result
if filtered_df.empty:
    st.sidebar.warning("No users match the selected filters. Showing all users.")
    display_df = users_df
else:
    display_df = filtered_df

user_list = display_df["user_id"].tolist()

# Load or synchronize default selected user ID from Session State
default_idx = 0
if "selected_user" in st.session_state and st.session_state["selected_user"] in user_list:
    default_idx = user_list.index(st.session_state["selected_user"])

selected_user_id = st.sidebar.selectbox("Select User ID", user_list, index=default_idx)

# Save selected user to Session State
st.session_state["selected_user"] = selected_user_id

# Retrieve profile record
user_row = users_df[users_df["user_id"] == selected_user_id].iloc[0]

# Display Profile
col1, col2 = st.columns([1, 2])

with col1:
    # Build badges list
    skills_list = [s.strip() for s in str(user_row["skills"]).split(",") if s.strip()]
    traits_list = [t.strip() for t in str(user_row["traits"]).split(",") if t.strip()]
    
    skills_badges = "".join([f'<span class="badge badge-blue">{s}</span>' for s in skills_list])
    traits_badges = "".join([f'<span class="badge">{t}</span>' for t in traits_list])
    
    st.markdown(f"""
    <div class="glass-card">
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%); margin: 0 auto 10px; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: bold; color: white;">
                {user_row["name"][0] if isinstance(user_row["name"], str) and len(user_row["name"]) > 0 else "U"}
            </div>
            <h2 style="margin: 0; font-size: 1.6rem; color: #f8fafc;">{user_row["name"]}</h2>
            <p style="margin: 5px 0 0; color: #a855f7; font-weight: 500;">{user_row["profession"]}</p>
            <p style="margin: 2px 0 0; font-size: 0.85rem; color: #64748b;">ID: {user_row["user_id"]}</p>
        </div>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;"/>
        <div class="profile-item"><span class="profile-label">📍 Location:</span><span class="profile-value">{user_row["location"]}</span></div>
        <div class="profile-item"><span class="profile-label">💼 Experience:</span><span class="profile-value">{user_row["experience_years"]} years</span></div>
        <div class="profile-item"><span class="profile-label">🧬 MBTI Type:</span><span class="badge badge-green" style="margin: 0;">{user_row["mbti"]}</span></div>
        <div class="profile-item"><span class="profile-label">🎓 Education:</span><span class="profile-value">{user_row["education"]}</span></div>
        <div class="profile-item"><span class="profile-label">🎯 Career Goal:</span><span class="profile-value">{user_row["career_goal"]}</span></div>
        <div class="profile-item"><span class="profile-label">🤝 Intent:</span><span class="profile-value">{user_row["networking_intent"]}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render quick stats
    st.markdown("### Profile Summary Metrics")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.metric("Experience", f"{user_row['experience_years']} yrs")
    with c_m2:
        st.metric("Age", f"{user_row['age']} yrs")

with col2:
    st.markdown(f"""
    <div class="glass-card" style="height: 100%;">
        <h3 style="border-left: 4px solid #a855f7; padding-left: 8px;">📝 Professional Summary</h3>
        <p style="color: #cbd5e1; line-height: 1.6; font-size: 1.05rem; margin-bottom: 20px;">
            {user_row["professional_summary"]}
        </p>
        
        <h3 style="border-left: 4px solid #3b82f6; padding-left: 8px;">👤 About Me</h3>
        <p style="color: #cbd5e1; line-height: 1.6; font-size: 1.05rem; margin-bottom: 20px;">
            {user_row["about_me"]}
        </p>
        
        <h3 style="border-left: 4px solid #10b981; padding-left: 8px;">🛠️ Core Skills</h3>
        <div style="margin-bottom: 20px;">
            {skills_badges}
        </div>
        
        <h3 style="border-left: 4px solid #f59e0b; padding-left: 8px;">🧬 Personality Traits</h3>
        <div style="margin-bottom: 20px;">
            {traits_badges}
        </div>
        
        <h3 style="border-left: 4px solid #ec4899; padding-left: 8px;">✨ Interests & Hobbies</h3>
        <div>
            {"".join([f'<span class="badge badge-orange">{i.strip()}</span>' for i in str(user_row["interests"]).split(",") if i.strip()])}
        </div>
    </div>
    """, unsafe_allow_html=True)
