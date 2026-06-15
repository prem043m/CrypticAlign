from pathlib import Path
import sys

import pandas as pd
import streamlit as st


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.embeddings.tfidf_encoder import TFIDFEncoder
from src.learning.adaptive_recommender import AdaptiveRecommender
from src.learning.feedback_dataset import FeedbackDatasetBuilder
from src.learning.feedback_model import FeedbackModel
from src.matching.recommender import Recommender
from src.utils.model_manager import ModelManager
from src.utils.nltk_helper import ensure_nltk_resources
from datetime import datetime, timezone


DATA_DIR = project_root / "data"
USERS_PATH = DATA_DIR / "users.csv"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"


ensure_nltk_resources()


@st.cache_data
def get_users() -> pd.DataFrame:
    """Load users and rebuild the derived profile text column."""
    users_df = pd.read_csv(USERS_PATH)
    
    user_profiles_path = DATA_DIR / "user_profiles.csv"
    if user_profiles_path.exists() and user_profiles_path.stat().st_size > 0:
        try:
            profiles_df = pd.read_csv(user_profiles_path)
            for col in users_df.columns:
                if col not in profiles_df.columns:
                    if col == "age":
                        profiles_df[col] = 30
                    elif col == "experience_years":
                        profiles_df[col] = 0
                    else:
                        profiles_df[col] = ""
            profiles_aligned = profiles_df[users_df.columns].copy()
            users_df = pd.concat([users_df, profiles_aligned], ignore_index=True)
        except Exception as e:
            pass
            
    return TFIDFEncoder.build_profile_text(users_df)


@st.cache_data
def get_feedback() -> pd.DataFrame:
    """Load raw feedback records."""
    return pd.read_csv(FEEDBACK_PATH)


@st.cache_resource
def get_vectorizer():
    """Load the persisted TF-IDF vectorizer, training it if missing."""
    vectorizer = ModelManager.load_model("tfidf_vectorizer.pkl")
    if vectorizer is not None:
        return vectorizer

    encoder = TFIDFEncoder()
    encoder.fit(USERS_PATH)
    return encoder.vectorizer


@st.cache_resource
def get_tfidf_matrix():
    """Recreate the sparse TF-IDF matrix from the saved vectorizer."""
    users_df = get_users()
    vectorizer = get_vectorizer()
    return vectorizer.transform(users_df["profile_text"])


@st.cache_resource
def get_recommender() -> Recommender:
    """Build the hybrid recommender from current users and TF-IDF matrix."""
    return Recommender(get_users(), get_tfidf_matrix())


@st.cache_data
def get_training_dataset() -> pd.DataFrame:
    """Build the supervised dataset from the latest feedback log."""
    builder = FeedbackDatasetBuilder(get_recommender(), FEEDBACK_PATH)
    return builder.build()


@st.cache_resource
def get_feedback_model() -> FeedbackModel:
    """Load the trained classifier or fit a fresh one if needed."""
    saved_model = ModelManager.load_model("feedback_model.pkl")
    if saved_model is not None and hasattr(saved_model, "predict_proba"):
        return FeedbackModel(model=saved_model)

    feedback_model = FeedbackModel()
    feedback_model.train(get_training_dataset())
    return feedback_model


@st.cache_resource
def get_adaptive_recommender() -> AdaptiveRecommender:
    """Build the stage-2 adaptive recommender."""
    return AdaptiveRecommender(get_recommender(), get_feedback_model())


def load_system() -> dict:
    """Load all app-ready components in one place for Streamlit pages."""
    return {
        "users_df": get_users(),
        "feedback_df": get_feedback(),
        "vectorizer": get_vectorizer(),
        "tfidf_matrix": get_tfidf_matrix(),
        "recommender": get_recommender(),
        "training_dataset": get_training_dataset(),
        "feedback_model": get_feedback_model(),
        "adaptive": get_adaptive_recommender(),
    }


def clear_system_caches() -> None:
    """Clear cached Streamlit data/resources after profile or feedback writes."""
    st.cache_data.clear()
    st.cache_resource.clear()


def refresh_feedback_views() -> None:
    """Refresh feedback-backed views without invalidating the active model."""
    get_feedback.clear()
    get_training_dataset.clear()


def rebuild_vectorizer():
    """Refit and persist the TF-IDF vectorizer after user profile changes."""
    encoder = TFIDFEncoder()
    users_df, tfidf_matrix = encoder.fit(USERS_PATH)
    clear_system_caches()
    return users_df, tfidf_matrix


def retrain_feedback_model() -> float:
    """Retrain and persist the feedback model from the latest feedback data."""
    clear_system_caches()
    training_dataset = get_training_dataset()
    feedback_model = FeedbackModel()
    accuracy = feedback_model.train(training_dataset)
    ModelManager.save_metadata(
        {
            "last_trained_at": datetime.now(timezone.utc).isoformat(),
            "feedback_rows_at_train": int(len(get_feedback())),
            "training_rows_at_train": int(len(training_dataset)),
            "accuracy": float(accuracy),
        }
    )
    clear_system_caches()
    return accuracy
