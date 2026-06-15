import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set Streamlit Page Configuration at the very first step
st.set_page_config(
    page_title="NexMatch AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils.loader import load_system, clear_system_caches
from src.utils.styles import load_css
from src.utils.sidebar import render_sidebar
from src.utils.data_manager import (
    authenticate_user,
    register_credentials,
    register_user_profile,
    MBTI_OPTIONS,
    NETWORKING_INTENT_OPTIONS
)
from src.views.user_portal import (
    render_user_home,
    render_user_profile,
    render_user_recs,
    render_user_feedback,
    render_user_history
)
from src.views.admin_portal import (
    render_admin_dashboard,
    render_user_explorer,
    render_model_analytics,
    render_dataset_insights,
    render_explainability,
    render_system_status
)

# Render login / registration screen
def render_login_screen():
    load_css()
    
    st.markdown('<div class="gradient-header">NexMatch AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Intelligent Hybrid Professional Recommendation Platform</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔒 Member Login", "🆕 Register Profile"])
    
    with tab1:
        st.markdown("### Sign In to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username").strip()
            password = st.text_input("Password", type="password").strip()
            login_btn = st.form_submit_button("Sign In", type="primary")
            
        if login_btn:
            if not username or not password:
                st.error("Please fill in both fields.")
            else:
                success, uid, role, msg = authenticate_user(username, password)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["user_id"] = uid
                    st.session_state["role"] = role
                    st.session_state["portal"] = "User Portal" if role == "user" else "Admin Portal"
                    st.success("Welcome back! Loading system...")
                    st.rerun()
                else:
                    st.error(msg)
                    
    with tab2:
        st.markdown("### Create Your Professional Profile")
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                r_username = st.text_input("Account Username (Unique)").strip()
                r_password = st.text_input("Account Password", type="password").strip()
                name = st.text_input("Your Full Name").strip()
                profession = st.text_input("Current Profession").strip()
                location = st.text_input("Geographic Location (City)").strip()
                experience_years = st.number_input("Years of Industry Experience", min_value=0, max_value=50, value=1)
                mbti = st.selectbox("MBTI Personality Type", MBTI_OPTIONS, index=0)
                
            with col2:
                career_goal = st.text_input("Primary Career Goal").strip()
                networking_intent = st.selectbox("Networking Objective", NETWORKING_INTENT_OPTIONS, index=0)
                skills = st.text_area("Core Skills (comma separated)").strip()
                interests = st.text_area("Hobbies / Interests (comma separated)").strip()
                professional_summary = st.text_area("Professional Summary").strip()
                about_me = st.text_area("About Me").strip()
                
            register_btn = st.form_submit_button("Register & Create Profile", type="primary")
            
        if register_btn:
            # Validations
            required = [r_username, r_password, name, profession, location, career_goal, professional_summary, about_me]
            if any(not val for val in required):
                st.error("Please fill in all fields to register.")
            else:
                # Check username availability
                from src.utils.data_manager import load_credentials_raw
                creds = load_credentials_raw()
                if not creds.empty and r_username.lower() in creds["username"].str.lower().tolist():
                    st.error("Username already exists. Please choose a different one.")
                else:
                    profile = {
                        "name": name,
                        "profession": profession,
                        "location": location,
                        "experience_years": int(experience_years),
                        "mbti": mbti,
                        "career_goal": career_goal,
                        "skills": skills,
                        "interests": interests,
                        "networking_intent": networking_intent,
                        "professional_summary": professional_summary,
                        "about_me": about_me
                    }
                    new_user_id = register_user_profile(profile)
                    success, msg = register_credentials(r_username, r_password, new_user_id, role="user")
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = r_username
                        st.session_state["user_id"] = new_user_id
                        st.session_state["role"] = "user"
                        st.session_state["portal"] = "User Portal"
                        clear_system_caches()
                        st.success("Profile created and logged in successfully!")
                        st.rerun()
                    else:
                        st.error(msg)


# --- View Wrapper Functions for st.navigation ---

def user_home_view():
    load_css()
    system = load_system()
    render_user_home(system)

def user_profile_view():
    load_css()
    system = load_system()
    render_user_profile(system)

def user_recs_view():
    load_css()
    system = load_system()
    render_user_recs(system)

def user_feedback_view():
    load_css()
    system = load_system()
    render_user_feedback(system)

def user_history_view():
    load_css()
    system = load_system()
    render_user_history(system)

def admin_dashboard_view():
    load_css()
    system = load_system()
    render_admin_dashboard(system)

def admin_explorer_view():
    load_css()
    system = load_system()
    render_user_explorer(system)

def admin_analytics_view():
    load_css()
    system = load_system()
    render_model_analytics(system)

def admin_insights_view():
    load_css()
    system = load_system()
    render_dataset_insights(system)

def admin_explainability_view():
    load_css()
    system = load_system()
    render_explainability(system)

def admin_status_view():
    load_css()
    system = load_system()
    render_system_status(system)


# Main Routing Loop
def main():
    if not st.session_state.get("authenticated", False):
        render_login_screen()
    else:
        system = load_system()
        users_df = system["users_df"]
        
        # Render common sidebar & retrieve portal
        portal = render_sidebar(users_df)
        
        # Build Navigation Menu
        if portal == "Admin Portal":
            pages_list = [
                st.Page(admin_dashboard_view, title="Dashboard", icon="📊", default=True),
                st.Page(admin_explorer_view, title="User Explorer", icon="🔍"),
                st.Page(admin_analytics_view, title="Model Analytics", icon="📈"),
                st.Page(admin_insights_view, title="Dataset Insights", icon="💡"),
                st.Page(admin_explainability_view, title="Explainability", icon="📢"),
                st.Page(admin_status_view, title="System Status", icon="⚙️")
            ]
        else:
            pages_list = [
                st.Page(user_home_view, title="Home", icon="🏠", default=True),
                st.Page(user_profile_view, title="My Profile", icon="👤"),
                st.Page(user_recs_view, title="My Recommendations", icon="🎯"),
                st.Page(user_feedback_view, title="Feedback History", icon="💬"),
                st.Page(user_history_view, title="Recommendation History", icon="📜")
            ]
            
        pg = st.navigation(pages_list)
        pg.run()


if __name__ == "__main__":
    main()
