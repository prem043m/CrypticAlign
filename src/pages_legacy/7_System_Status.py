import streamlit as st
from pathlib import Path
import sys


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.loader import load_system
from src.utils.loader import retrain_feedback_model
from src.utils.styles import load_css
from src.utils.model_manager import ModelManager


MODEL_DIR = project_root / "models"


st.set_page_config(
    page_title="System Status | NexMatch AI",
    page_icon="🛠️",
    layout="wide"
)

load_css()

system = load_system()
users_df = system["users_df"]
feedback_df = system["feedback_df"]
metadata = ModelManager.load_metadata()
feedback_rows_at_train = int(metadata.get("feedback_rows_at_train", 0))
pending_feedback = max(len(feedback_df) - feedback_rows_at_train, 0)
last_trained_at = metadata.get("last_trained_at", "Not recorded yet")

st.markdown('<div class="gradient-header">System Status</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Operational health, artifact checks, and model retraining controls</div>', unsafe_allow_html=True)

acceptance_rate = feedback_df["action"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Profiles", f"{len(users_df):,}")
with col2:
    st.metric("Feedback Rows", f"{len(feedback_df):,}")
with col3:
    st.metric("Acceptance Rate", f"{acceptance_rate:.2f}%")
with col4:
    st.metric("Pending Feedback", f"{pending_feedback:,}")

artifact_col1, artifact_col2 = st.columns(2)
with artifact_col1:
    st.markdown("### Saved Artifacts")
    tfidf_exists = (MODEL_DIR / "tfidf_vectorizer.pkl").exists()
    model_exists = (MODEL_DIR / "feedback_model.pkl").exists()
    st.write(f"TF-IDF Vectorizer: {'Available' if tfidf_exists else 'Missing'}")
    st.write(f"Feedback Model: {'Available' if model_exists else 'Missing'}")
    st.write(f"Last Trained At: {last_trained_at}")
    if "accuracy" in metadata:
        st.write(f"Last Recorded Accuracy: {metadata['accuracy'] * 100:.2f}%")

with artifact_col2:
    st.markdown("### Batch Retraining")
    st.markdown("Use this only after enough new feedback has accumulated. This simulates the production-style nightly/manual batch refresh instead of retraining on every click.")
    if st.button("Retrain Feedback Model", type="primary"):
        with st.spinner("Retraining model on the latest feedback..."):
            accuracy = retrain_feedback_model()
        st.success(f"Model retrained successfully. Latest held-out accuracy: {accuracy * 100:.2f}%")
        st.rerun()

st.markdown("### Notes")
st.markdown(
    """
    - New profiles update `data/users.csv` and rebuild the TF-IDF vectorizer immediately.
    - Accept/Reject actions append rows to `data/feedback.csv` immediately.
    - Recommendation snapshots stay stable until users refresh them.
    - The feedback model only changes when an admin clicks retrain here.
    """
)
