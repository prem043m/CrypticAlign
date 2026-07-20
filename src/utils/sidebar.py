import streamlit as st
import pandas as pd
from pathlib import Path
from src.utils.audit_logger import log_event

_LOGO_PATH = Path(__file__).resolve().parent.parent / "icon.png"

def render_sidebar(users_df: pd.DataFrame):
    """
    Renders a unified sidebar for NextMatchAI, managing the portal selector
    and active user context based on user credentials and roles.
    """
    # Logo at top of sidebar
    if _LOGO_PATH.exists():
        st.sidebar.image(str(_LOGO_PATH), width=56)
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='margin-bottom: 0px;'>NextMatchAI</h2>
        <small style='color: #94a3b8;'>v1.0.6 | Production Ready</small>
    </div>
    """, unsafe_allow_html=True)

    # Check session state credentials
    authenticated = st.session_state.get("authenticated", False)
    role = st.session_state.get("role", "user")
    current_uid = st.session_state.get("user_id")

    if not authenticated:
        st.sidebar.info("Please log in to continue.")
        return "User Portal"

    # 1. Portal Selector (Admin Only)
    if "portal" not in st.session_state:
        st.session_state["portal"] = "User Portal"

    if role == "admin":
        portal_list = ["User Portal", "Admin Portal"]
        current_portal_idx = portal_list.index(st.session_state["portal"]) if st.session_state["portal"] in portal_list else 0
        
        portal = st.sidebar.selectbox(
            "Select Portal",
            portal_list,
            index=current_portal_idx,
            key="sidebar_portal_selector"
        )
        
        # If portal changed, update state and rerun
        if portal != st.session_state["portal"]:
            st.session_state["portal"] = portal
            st.rerun()
    else:
        # Standard users are locked into User Portal
        portal = "User Portal"
        st.session_state["portal"] = "User Portal"

    st.sidebar.markdown("---")

    # 2. Active Context Profile Selector
    user_list = users_df["user_id"].tolist()

    if role == "admin":
        st.sidebar.markdown("### ⚙️ Admin Context Selector")
        # Admin can select any user profile to impersonate
        active_user = st.session_state.get("current_user", st.session_state.get("selected_user"))
        if not active_user or active_user not in user_list:
            active_user = user_list[0] if user_list else ""
            
        default_idx = user_list.index(active_user) if active_user in user_list else 0
        
        def get_user_label(uid):
            row = users_df[users_df["user_id"] == uid]
            if not row.empty:
                name = row.iloc[0]["name"]
                prof = row.iloc[0]["profession"]
                return f"{name} ({uid}) - {prof}"
            return uid
            
        selected_uid = st.sidebar.selectbox(
            "Act As Profile",
            user_list,
            index=default_idx,
            format_func=get_user_label,
            key="sidebar_user_selector"
        )
    else:
        # Standard user is locked to their registered profile
        selected_uid = current_uid
        st.sidebar.markdown("### 👤 User Profile Info")

    st.session_state["selected_user"] = selected_uid
    st.session_state["current_user"] = selected_uid

    # Display active profile snapshot card
    match_row = users_df[users_df["user_id"] == selected_uid]
    if not match_row.empty:
        user_row = match_row.iloc[0]
        st.sidebar.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.02); padding: 12px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.85rem; margin-top: 10px;">
            <span style="color: #cbd5e1; font-weight: 600;">{user_row['name']}</span> ({selected_uid})<br/>
            <span style="color: #94a3b8; font-size: 0.8rem;">{user_row['profession']}</span><br/>
            <span style="color: #64748b; font-size: 0.75rem;">📍 {user_row['location']} | MBTI: {user_row['mbti']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning(f"Profile {selected_uid} not found in database.")

    # 3. Log out button
    st.sidebar.markdown("<br/><br/>", unsafe_allow_html=True)
    if st.sidebar.button("Logout", key="sidebar_logout_btn"):
        uid = st.session_state.get("user_id", "UNKNOWN")
        log_event(uid, "LOGOUT", f"User {st.session_state.get('username', '')} logged out")
        st.session_state.clear()
        st.rerun()

    return portal
