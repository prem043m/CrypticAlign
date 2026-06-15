import streamlit as st
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.data_manager import MBTI_OPTIONS
from src.utils.data_manager import NETWORKING_INTENT_OPTIONS
from src.utils.data_manager import append_feedback
from src.utils.data_manager import append_user_profile
from src.utils.loader import load_system
from src.utils.loader import rebuild_vectorizer
from src.utils.loader import refresh_feedback_views
from src.utils.styles import load_css


st.set_page_config(
    page_title="Create Profile | NexMatch AI",
    page_icon="🆕",
    layout="wide"
)

load_css()

st.markdown('<div class="gradient-header">Create Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Register a new professional profile and start collecting recommendation feedback immediately</div>', unsafe_allow_html=True)

with st.form("profile_form"):
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=18, max_value=80, value=25)
        location = st.text_input("Location")
        profession = st.text_input("Profession")
        experience_years = st.number_input("Experience Years", min_value=0, max_value=50, value=1)
        education = st.text_input("Education")
        skills = st.text_area("Skills (comma separated)")
        mbti = st.selectbox("MBTI", MBTI_OPTIONS, index=0)

    with col2:
        traits = st.text_area("Traits (comma separated)")
        career_goal = st.text_input("Career Goal")
        networking_intent = st.selectbox("Networking Intent", NETWORKING_INTENT_OPTIONS, index=0)
        interests = st.text_area("Interests (comma separated)")
        professional_summary = st.text_area("Professional Summary")
        about_me = st.text_area("About Me")

    submitted = st.form_submit_button("Create Profile", type="primary")

if submitted:
    required_fields = {
        "name": name.strip(),
        "location": location.strip(),
        "profession": profession.strip(),
        "education": education.strip(),
        "career_goal": career_goal.strip(),
        "professional_summary": professional_summary.strip(),
        "about_me": about_me.strip(),
    }

    missing_fields = [field for field, value in required_fields.items() if not value]
    if missing_fields:
        st.error("Please fill all required fields: " + ", ".join(missing_fields))
    else:
        profile = {
            "name": name.strip(),
            "age": int(age),
            "location": location.strip(),
            "profession": profession.strip(),
            "experience_years": int(experience_years),
            "education": education.strip(),
            "skills": skills.strip(),
            "mbti": mbti,
            "traits": traits.strip(),
            "career_goal": career_goal.strip(),
            "networking_intent": networking_intent,
            "interests": interests.strip(),
            "professional_summary": professional_summary.strip(),
            "about_me": about_me.strip(),
        }
        new_user_id = append_user_profile(profile)
        rebuild_vectorizer()
        st.session_state["current_user"] = new_user_id
        st.session_state["selected_user"] = new_user_id
        st.session_state.setdefault("recommendation_snapshots", {})
        st.session_state["recommendation_snapshots"].pop(new_user_id, None)
        st.session_state["profile_created_message"] = f"Profile created successfully: {new_user_id}"
        st.rerun()

if "profile_created_message" in st.session_state:
    st.success(st.session_state.pop("profile_created_message"))

current_user = st.session_state.get("current_user")
if current_user:
    system = load_system()
    adaptive = system["adaptive"]
    users_df = system["users_df"]

    current_user_df = users_df[users_df["user_id"] == current_user]
    if current_user_df.empty:
        st.warning("The active created profile could not be found. Please create the profile again.")
    else:
        user_row = current_user_df.iloc[0]
        st.markdown("### Active Registered User")
        st.markdown(
            f"""
            <div class="glass-card">
                <b>{user_row['name']}</b> ({current_user})<br/>
                <span style="color: #94a3b8;">{user_row['profession']} | {user_row['location']} | {user_row['experience_years']} years</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        session_recs = st.session_state.setdefault("recommendation_snapshots", {})
        snapshot = session_recs.get(current_user)
        if snapshot is None:
            snapshot = {
                "candidate_pool": adaptive.get_candidate_pool(current_user, pool_size=30),
                "final_recs": adaptive.get_top_recommendations(current_user, top_n=5),
            }
            session_recs[current_user] = snapshot
            st.session_state["recommendation_snapshots"] = session_recs

        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col1:
            st.info("These recommendations stay fixed after feedback clicks. Refresh only when you want a new snapshot from the currently deployed model.")
        with refresh_col2:
            if st.button("Refresh Snapshot", key=f"refresh_{current_user}", type="primary"):
                session_recs[current_user] = {
                    "candidate_pool": adaptive.get_candidate_pool(current_user, pool_size=30),
                    "final_recs": adaptive.get_top_recommendations(current_user, top_n=5),
                }
                st.session_state["recommendation_snapshots"] = session_recs
                st.rerun()

        recommendations = snapshot["final_recs"]

        st.markdown("### Immediate Recommendations")
        if not recommendations:
            st.info("No recommendations are available yet for this profile.")
        else:
            for idx, rec in enumerate(recommendations):
                target = users_df[users_df["user_id"] == rec["user_id"]].iloc[0]
                info_col, accept_col, reject_col = st.columns([4, 1, 1])
                with info_col:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding: 14px;">
                            <b>{target['name']}</b> ({rec['user_id']})<br/>
                            <span style="color: #94a3b8;">{rec['profession']} | Hybrid: {rec['hybrid_score']:.2f}% | ML: {rec['ml_score']:.2f}% | Final: {rec['final_ranking_score']:.2f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with accept_col:
                    accepted = st.button("Accept", key=f"create_accept_{current_user}_{rec['user_id']}_{idx}")
                with reject_col:
                    rejected = st.button("Reject", key=f"create_reject_{current_user}_{rec['user_id']}_{idx}")

                if accepted:
                    append_feedback(current_user, rec["user_id"], 1)
                    refresh_feedback_views()
                    st.success(f"Saved accept feedback for {current_user} -> {rec['user_id']}.")
                    st.rerun()

                if rejected:
                    append_feedback(current_user, rec["user_id"], 0)
                    refresh_feedback_views()
                    st.success(f"Saved reject feedback for {current_user} -> {rec['user_id']}.")
                    st.rerun()
else:
    st.info("Create a profile to start the end-to-end recommendation and feedback loop.")
