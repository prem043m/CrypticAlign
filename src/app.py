import warnings
# Suppress sklearn unpickling version mismatch warnings and Streamlit container width deprecation warnings
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Logo path
_LOGO_PATH = Path(__file__).resolve().parent / "icon.png"

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set Streamlit Page Configuration at the very first step
st.set_page_config(
    page_title="NextMatchAI",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "🦊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils.loader import load_system, clear_system_caches
from src.utils.styles import load_css
from src.utils.sidebar import render_sidebar
from src.utils.config import Config
from src.utils.email_service import send_welcome_email, send_password_reset_email
from src.utils.password_reset import generate_reset_token, validate_reset_token, consume_reset_token
from src.utils.validators import (
    validate_username,
    validate_password,
    validate_profile,
    validate_email
)
from src.utils.data_manager import (
    authenticate_user,
    register_credentials,
    register_user_profile,
    MBTI_OPTIONS,
    NETWORKING_INTENT_OPTIONS,
    PROFESSION_OPTIONS,
    CAREER_GOAL_OPTIONS,
    update_password_hash,
    hash_password,
    disable_account,
    enable_account,
)
from src.utils.audit_logger import log_event
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
    render_system_status,
    render_recommendation_quality,
    render_user_management,
    render_system_monitoring,
    render_data_management,
    render_audit_log_viewer,
)

# Session constants
SESSION_TIMEOUT_MINUTES = 60


# Render login / registration screen
def _check_session_expiry():
    """Auto-logout after SESSION_TIMEOUT_MINUTES of inactivity."""
    from datetime import datetime, timedelta
    login_time = st.session_state.get("login_time")
    if login_time and (datetime.now() - login_time) > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
        uid = st.session_state.get("user_id", "UNKNOWN")
        log_event(uid, "SESSION_EXPIRED", f"Auto-logout after {SESSION_TIMEOUT_MINUTES} min inactivity")
        for key in ["authenticated", "username", "user_id", "role", "portal", "login_time",
                    "current_user", "recommendation_snapshots"]:
            st.session_state.pop(key, None)
        st.warning("Your session has expired. Please sign in again.")
        st.rerun()


def render_login_screen():
    load_css()

    # Logo on login screen
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=80)
    with col_title:
        st.markdown(f'<div class="gradient-header">{Config.APP_NAME}</div>', unsafe_allow_html=True)
        st.markdown('<div class="gradient-sub">Intelligent Hybrid Professional Recommendation Platform</div>', unsafe_allow_html=True)
    
    tabs_to_show = ["🔒 Member Login"]
    if Config.ENABLE_SIGNUPS:
        tabs_to_show.append("🆕 Register Profile")
    if Config.ENABLE_PASSWORD_RESET:
        tabs_to_show.append("🔑 Forgot Password")
        
    tabs = st.tabs(tabs_to_show)
    
    # Render Member Login Tab
    with tabs[0]:
        st.markdown("### Sign In to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username or Email").strip()
            password = st.text_input("Password", type="password").strip()
            login_btn = st.form_submit_button("Sign In", type="primary")
            
        if login_btn:
            if not username or not password:
                st.error("Please fill in both fields.")
            else:
                success, uid, role, msg = authenticate_user(username, password)
                if success:
                    from datetime import datetime
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    st.session_state["user_id"] = uid
                    st.session_state["role"] = role
                    st.session_state["portal"] = "User Portal" if role == "user" else "Admin Portal"
                    st.session_state["login_time"] = datetime.now()
                    log_event(uid, "LOGIN", f"username={username} role={role}")
                    st.success("Welcome back! Loading system...")
                    st.rerun()
                else:
                    log_event(username, "LOGIN_FAILED", msg)
                    st.error(msg)
                    
    current_tab_idx = 1
    # Render Register Profile Tab
    if Config.ENABLE_SIGNUPS:
        with tabs[current_tab_idx]:
            st.markdown("### Create Your Professional Profile")
            with st.form("register_form"):
                col1, col2 = st.columns(2)
                with col1:
                    r_username = st.text_input("Account Username (Unique)").strip()
                    email = st.text_input("Email Address (for notifications/reset)").strip()
                    r_password = st.text_input("Account Password", type="password").strip()
                    name = st.text_input("Your Full Name").strip()
                    profession = st.selectbox("Current Profession", PROFESSION_OPTIONS, index=0)
                    location = st.text_input("Geographic Location (City)").strip()
                    experience_years = st.number_input("Years of Industry Experience", min_value=0, max_value=50, value=1)
                    mbti = st.selectbox("MBTI Personality Type", MBTI_OPTIONS, index=0)
                    
                with col2:
                    career_goal = st.selectbox("Primary Career Goal", CAREER_GOAL_OPTIONS, index=0)
                    networking_intent = st.selectbox("Networking Objective", NETWORKING_INTENT_OPTIONS, index=0)
                    skills = st.text_area("Core Skills (comma separated, min 3)").strip()
                    interests = st.text_area("Hobbies / Interests (comma separated, min 2)").strip()
                    professional_summary = st.text_area("Professional Summary (50-500 chars)").strip()
                    about_me = st.text_area("About Me (100-1000 chars)").strip()
                    
                register_btn = st.form_submit_button("Register & Create Profile", type="primary")
                
            if register_btn:
                # Step 1: Validate username
                u_ok, u_msg = validate_username(r_username)
                if not u_ok:
                    st.error(f"🔴 Username: {u_msg}")
                
                # Step 1b: Validate email
                e_ok, e_msg = validate_email(email)
                if not e_ok:
                    st.error(f"🔴 Email: {e_msg}")
                
                # Step 2: Validate password
                p_ok, p_msg = validate_password(r_password)
                if not p_ok:
                    st.error(f"🔴 Password: {p_msg}")
                
                # Step 3: Validate profile fields
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
                
                prof_ok, prof_errors = validate_profile(profile)
                for err in prof_errors:
                    st.error(f"🔴 {err}")
                
                # Only proceed if ALL validations pass
                if u_ok and e_ok and p_ok and prof_ok:
                    # Check username/email availability
                    from src.utils.data_manager import load_credentials_raw
                    creds = load_credentials_raw()
                    username_taken = (not creds.empty and r_username.lower() in creds["username"].str.lower().tolist())
                    email_taken = (not creds.empty and email.lower() in [str(e).lower() for e in creds["email"].dropna() if str(e).strip()])
                    
                    if username_taken:
                        st.error("Username already exists. Please choose a different one.")
                    elif email_taken:
                        st.error("Email address is already registered.")
                    else:
                        new_user_id = register_user_profile(profile)
                        success, msg = register_credentials(r_username, email, r_password, new_user_id, role="user")
                        if success:
                            # Send welcome email asynchronously/fire-and-forget (logs error internally if fails)
                            send_welcome_email(name, email)
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
        current_tab_idx += 1

    # Render Password Reset Tab
    if Config.ENABLE_PASSWORD_RESET:
        with tabs[current_tab_idx]:
            st.markdown("### Reset Your Password")
            
            sub_step = st.radio("Select Step", ["1. Request Reset Token", "2. Verify Token & Update Password"], horizontal=True)
            
            if sub_step == "1. Request Reset Token":
                with st.form("reset_request_form"):
                    reset_email = st.text_input("Enter Registered Email Address").strip()
                    submit_req = st.form_submit_button("Send Reset Token", type="primary")
                    
                if submit_req:
                    if not reset_email:
                        st.error("Please enter your email address.")
                    else:
                        from src.utils.data_manager import load_credentials_raw
                        creds = load_credentials_raw()
                        match = creds[creds["email"].str.lower() == reset_email.lower()]
                        if match.empty:
                            st.error("This email address is not registered in our system.")
                        else:
                            user_row = match.iloc[0]
                            user_name = str(user_row.get("username", "User"))
                            
                            # Generate token and save
                            token = generate_reset_token(reset_email)
                            
                            # Send email
                            sent = send_password_reset_email(user_name, reset_email, token)
                            if sent:
                                st.success(f"A password reset token has been sent to {reset_email}. Proceed to step 2.")
                            else:
                                st.warning(f"Failed to send email. Since SMTP is disabled/failed, your token is: {token} (Enter this in Step 2)")
                                
            else:
                with st.form("reset_verify_form"):
                    reset_token = st.text_input("Enter Reset Token").strip()
                    new_password = st.text_input("New Password", type="password").strip()
                    confirm_password = st.text_input("Confirm New Password", type="password").strip()
                    submit_reset = st.form_submit_button("Update Password", type="primary")
                    
                if submit_reset:
                    # Validate password
                    p_ok, p_msg = validate_password(new_password)
                    
                    if not reset_token:
                        st.error("Please enter the reset token.")
                    elif not p_ok:
                        st.error(f"🔴 Password: {p_msg}")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        valid, email_address, err_msg = validate_reset_token(reset_token)
                        if not valid:
                            st.error(err_msg)
                        else:
                            # Update credentials database
                            from src.utils.data_manager import load_credentials_raw
                            creds = load_credentials_raw()
                            match = creds[creds["email"].str.lower() == email_address.lower()]
                            if match.empty:
                                st.error("User associated with token not found.")
                            else:
                                user_id = str(match.iloc[0]["user_id"])
                                hashed = hash_password(new_password)
                                if update_password_hash(user_id, hashed):
                                    consume_reset_token(reset_token)
                                    st.success("Password updated successfully! You can now log in using the Login tab.")
                                else:
                                    st.error("Failed to update password. Please contact support.")


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

def admin_quality_view():
    load_css()
    system = load_system()
    render_recommendation_quality(system)

def admin_user_management_view():
    load_css()
    system = load_system()
    render_user_management(system)

def admin_monitoring_view():
    load_css()
    system = load_system()
    render_system_monitoring(system)

def admin_data_view():
    load_css()
    system = load_system()
    render_data_management(system)

def admin_audit_view():
    load_css()
    system = load_system()
    render_audit_log_viewer(system)


# Main Routing Loop
def main():
    if not st.session_state.get("authenticated", False):
        render_login_screen()
    else:
        _check_session_expiry()
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
                st.Page(admin_status_view, title="System Status", icon="⚙️"),
                st.Page(admin_quality_view, title="Rec. Quality Audit", icon="🔬"),
                st.Page(admin_monitoring_view, title="System Monitoring", icon="🖥️"),
                st.Page(admin_user_management_view, title="User Management", icon="👥"),
                st.Page(admin_data_view, title="Data Management", icon="💾"),
                st.Page(admin_audit_view, title="Audit Log", icon="📋"),
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
