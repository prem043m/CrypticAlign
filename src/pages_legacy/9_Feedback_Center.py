import streamlit as st
import pandas as pd
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.data_manager import get_feedback_for_user
from src.utils.loader import load_system
from src.utils.model_manager import ModelManager
from src.utils.styles import load_css


st.set_page_config(
    page_title="Feedback Center | NexMatch AI",
    page_icon="📝",
    layout="wide"
)

load_css()

system = load_system()
users_df = system["users_df"]
feedback_df = system["feedback_df"]
metadata = ModelManager.load_metadata()

st.markdown('<div class="gradient-header">Feedback Center</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Review saved feedback events, understand what is waiting for retraining, and keep the recommendation platform stable between batch model updates.</div>', unsafe_allow_html=True)

user_list = users_df["user_id"].tolist()
default_user = st.session_state.get("current_user", st.session_state.get("selected_user", user_list[0]))
if default_user not in user_list:
    default_user = user_list[0]

selected_user_id = st.selectbox("Select User", user_list, index=user_list.index(default_user))
st.session_state["selected_user"] = selected_user_id

user_feedback = get_feedback_for_user(selected_user_id).sort_values(by="timestamp", ascending=False)
feedback_rows_at_train = int(metadata.get("feedback_rows_at_train", 0))
pending_feedback = max(len(feedback_df) - feedback_rows_at_train, 0)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Feedback for Selected User", f"{len(user_feedback):,}")
with col2:
    st.metric("Platform Feedback Rows", f"{len(feedback_df):,}")
with col3:
    st.metric("Pending Retrain Events", f"{pending_feedback:,}")

st.markdown("### Workflow")
st.markdown(
    """
    - User clicks `Accept` or `Reject`
    - Feedback is appended to `data/feedback.csv` immediately
    - The currently deployed model stays unchanged
    - An admin later retrains the model from **System Status**
    - Users refresh recommendation snapshots after that deployment
    """
)

st.markdown("### Recent Feedback History")
if user_feedback.empty:
    st.info("No feedback has been captured for this user yet.")
else:
    display_df = user_feedback.rename(
        columns={
            "user_id": "User ID",
            "matched_user_id": "Matched User ID",
            "action": "Action",
            "timestamp": "Timestamp",
        }
    ).copy()
    display_df["Action"] = display_df["Action"].map({1: "Accept", 0: "Reject"})
    st.dataframe(display_df, width='stretch', height=400)

st.markdown("### Latest Training Snapshot")
if metadata:
    metadata_df = pd.DataFrame(
        [
            {"Metric": "Last Trained At", "Value": metadata.get("last_trained_at", "Not recorded")},
            {"Metric": "Feedback Rows At Train", "Value": metadata.get("feedback_rows_at_train", 0)},
            {"Metric": "Training Rows At Train", "Value": metadata.get("training_rows_at_train", 0)},
            {"Metric": "Last Accuracy", "Value": f"{metadata.get('accuracy', 0.0) * 100:.2f}%"},
        ]
    )
    st.dataframe(metadata_df, width='stretch', hide_index=True)
else:
    st.warning("No training metadata found yet. Retrain the model once from System Status to initialize deployment tracking.")
