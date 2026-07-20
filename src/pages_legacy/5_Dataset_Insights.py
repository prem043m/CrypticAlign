import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.loader import load_system
from src.utils.styles import load_css

# Page Configuration
st.set_page_config(
    page_title="Dataset Insights | nextmatchAi",
    page_icon="📊",
    layout="wide"
)

# Load CSS
load_css()

# Load system objects
system = load_system()
users_df = system["users_df"]
feedback_df = system["feedback_df"]

# Page titles
st.markdown('<div class="gradient-header">Dataset Insights & EDA</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Exploratory Data Analysis and distribution statistics of the user network and feedback logs</div>', unsafe_allow_html=True)

# Metrics Row
total_users = len(users_df)
total_feedback = len(feedback_df)
accept_rate = (feedback_df["action"].mean()) * 100

st.markdown("### 📈 Core Dataset Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total User Profiles", f"{total_users:,}", help="Count of registered professional profiles")
with col2:
    st.metric("Total Feedback Logs", f"{total_feedback:,}", help="Total logged matches with interaction results")
with col3:
    st.metric("Acceptance Rate", f"{accept_rate:.2f}%", help="Percentage of recommended pairs accepted by the user")

st.markdown("<br/>", unsafe_allow_html=True)

# Row 1: Profession & MBTI
col_r1_1, col_r1_2 = st.columns([1, 1])

with col_r1_1:
    st.markdown("### 💼 User Profession Distribution")
    prof_counts = users_df["profession"].value_counts().reset_index()
    prof_counts.columns = ["Profession", "Count"]
    prof_counts = prof_counts.sort_values(by="Count", ascending=True)
    
    fig_prof = px.bar(
        prof_counts,
        x="Count",
        y="Profession",
        orientation="h",
        color="Count",
        color_continuous_scale="Purples",
        template="plotly_dark"
    )
    fig_prof.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter", size=10)
    )
    st.plotly_chart(fig_prof, use_container_width=True)

with col_r1_2:
    st.markdown("### 🧬 User MBTI Distribution")
    mbti_counts = users_df["mbti"].value_counts().reset_index()
    mbti_counts.columns = ["MBTI Type", "Count"]
    
    fig_mbti = px.pie(
        mbti_counts,
        values="Count",
        names="MBTI Type",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template="plotly_dark"
    )
    fig_mbti.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter", size=10)
    )
    st.plotly_chart(fig_mbti, use_container_width=True)

# Row 2: Career Goal & Feedback Action
col_r2_1, col_r2_2 = st.columns([1, 1])

with col_r2_1:
    st.markdown("### 🎯 User Career Goal Distribution")
    goal_counts = users_df["career_goal"].value_counts().reset_index()
    goal_counts.columns = ["Career Goal", "Count"]
    goal_counts = goal_counts.sort_values(by="Count", ascending=True)
    
    fig_goal = px.bar(
        goal_counts,
        x="Count",
        y="Career Goal",
        orientation="h",
        color="Count",
        color_continuous_scale="Blues",
        template="plotly_dark"
    )
    fig_goal.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter", size=10)
    )
    st.plotly_chart(fig_goal, use_container_width=True)

with col_r2_2:
    st.markdown("### 🤝 Acceptance vs Rejection Distribution")
    action_counts = feedback_df["action"].value_counts().reset_index()
    action_counts.columns = ["Action", "Count"]
    action_counts["Action"] = action_counts["Action"].map({1: "Accept (1)", 0: "Reject (0)"})
    
    fig_action = px.pie(
        action_counts,
        values="Count",
        names="Action",
        hole=0.4,
        color="Action",
        color_discrete_map={"Accept (1)": "#10b981", "Reject (0)": "#ef4444"},
        template="plotly_dark"
    )
    fig_action.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Inter", size=10)
    )
    st.plotly_chart(fig_action, use_container_width=True)
