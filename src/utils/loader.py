import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.embeddings.tfidf_encoder import TFIDFEncoder
from src.matching.similarity_engine import SimilarityEngine
from src.matching.recommender import Recommender
from src.learning.feedback_dataset import FeedbackDatasetBuilder
from src.learning.feedback_model import FeedbackModel
from src.learning.adaptive_recommender import AdaptiveRecommender

@st.cache_resource
def load_system():
    """
    Initializes and caches the recommendation system components.
    Loads users, feedback, trains the Logistic Regression feedback model,
    and returns a dict of initialized objects.
    """
    users_path = project_root / "data" / "users.csv"
    feedback_path = project_root / "data" / "feedback.csv"
    
    # 1. Initialize TF-IDF encoder
    encoder = TFIDFEncoder()
    users_df, matrix = encoder.fit(users_path)
    
    # 2. Load feedback data
    feedback_df = pd.read_csv(feedback_path)
    
    # 3. Initialize Similarity Engine and Recommender
    engine = SimilarityEngine(users_df, matrix)
    recommender = Recommender(users_df, matrix)
    
    # 4. Build Feedback Dataset
    feedback_builder = FeedbackDatasetBuilder(recommender, feedback_path)
    training_dataset = feedback_builder.build()
    
    # 5. Initialize and train Feedback model (Logistic Regression)
    feedback_model = FeedbackModel()
    accuracy = feedback_model.train(training_dataset)
    
    # 6. Initialize Adaptive Recommender
    adaptive = AdaptiveRecommender(recommender, feedback_model)
    
    return {
        "users_df": users_df,
        "feedback_df": feedback_df,
        "encoder": encoder,
        "engine": engine,
        "recommender": recommender,
        "feedback_model": feedback_model,
        "adaptive": adaptive,
        "training_dataset": training_dataset,
        "accuracy": accuracy
    }
