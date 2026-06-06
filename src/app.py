import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.loader import load_system
from src.utils.styles import load_css

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="NexMatch AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load styling
load_css()

# Load recommendation system components
with st.spinner("Initializing NexMatch AI Recommendation Pipeline..."):
    system = load_system()

# Extract models/data
users_df = system["users_df"]
feedback_df = system["feedback_df"]
training_dataset = system["training_dataset"]
feedback_model = system["feedback_model"]

# Calculate core statistics
total_users = len(users_df)
total_feedback = len(feedback_df)
acceptance_rate = (feedback_df["action"].mean()) * 100

# Compute test metrics
X = training_dataset.drop(columns=["label"])
y = training_dataset["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
predictions = feedback_model.model.predict(X_test)
probabilities = feedback_model.model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions) * 100
try:
    roc_auc = roc_auc_score(y_test, probabilities)
except Exception:
    roc_auc = 0.0

# Sidebar Title
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 20px;'>
    <h2 style='margin-bottom: 0px;'>🧠 NexMatch AI</h2>
    <small style='color: #94a3b8;'>v1.0.0 | Production Ready</small>
</div>
""", unsafe_allow_html=True)

# Main Title and Header
st.markdown('<div class="gradient-header">NexMatch AI</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Intelligent Hybrid Professional Recommendation Platform</div>', unsafe_allow_html=True)

# KPI Cards
st.markdown("### 📊 Platform Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Users", f"{total_users:,}", help="Count of registered professional user profiles")
with col2:
    st.metric("Feedback Records", f"{total_feedback:,}", help="Historical user interactions (Accept/Reject)")
with col3:
    st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%", help="Percentage of feedback records marked as Accept")
with col4:
    st.metric("Model Accuracy", f"{accuracy:.2f}%", help="Classifier accuracy evaluated on held-out test data")
with col5:
    st.metric("ROC AUC Score", f"{roc_auc:.4f}", help="Area under the ROC curve representing ranking power")

st.markdown("<br>", unsafe_allow_html=True)

# Pipeline Architecture Diagram
st.markdown("### 🛠️ Multi-Stage Recommendation Pipeline Architecture")
pipeline_html = """
<div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; background: rgba(255, 255, 255, 0.02); padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 30px;">
    <div style="text-align: center; flex: 1;">
        <div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
            👤 User Profile
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Raw demographic & text data</div>
    </div>
    <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
    <div style="text-align: center; flex: 1;">
        <div style="background: linear-gradient(135deg, #10b981, #047857); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
            📝 TF-IDF Similarity
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Cosine similarity on summaries</div>
    </div>
    <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
    <div style="text-align: center; flex: 1;">
        <div style="background: linear-gradient(135deg, #f59e0b, #b45309); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
            ⚖️ Hybrid Recommender
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">MBTI, Skills, Intent weightings</div>
    </div>
    <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
    <div style="text-align: center; flex: 1;">
        <div style="background: linear-gradient(135deg, #ec4899, #be185d); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
            🎯 Candidate Gen
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Stage 1: Retrieve top 30 pool</div>
    </div>
    <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
    <div style="text-align: center; flex: 1;">
        <div style="background: linear-gradient(135deg, #8b5cf6, #6d28d9); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
            🤖 ML Re-Ranking
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Stage 2: Logistic Regression</div>
    </div>
    <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
    <div style="text-align: center; flex: 1;">
        <div style="background: linear-gradient(135deg, #06b6d4, #0891b2); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
            📢 Explain Match
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Natural language justifications</div>
    </div>
</div>
"""
st.markdown(pipeline_html, unsafe_allow_html=True)

# Feature details in columns
st.markdown("### 🌟 Implemented Capabilities")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="glass-card">
        <h3>🧠 NLP & Profile Vectorization</h3>
        <p>Leverages TF-IDF vectorization with customized text cleaning routines (removing stopwords, punctuation, lemmatization) to represent unstructured professional summaries, bios, and career objectives in high-dimensional vector space.</p>
        <span class="badge">TFIDFEncoder</span>
        <span class="badge badge-blue">Cosine Similarity</span>
    </div>
    <div class="glass-card">
        <h3>⚡ Multi-Stage Candidate Generation</h3>
        <p>Combines text vector similarity, Myers-Briggs (MBTI) compatibility mapping, profession clusters, career goal alignment, location, experience levels, skills intersection, and networking intent into a single candidate retriever.</p>
        <span class="badge badge-green">Candidate Retrieval</span>
        <span class="badge">Hybrid Scoring</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="glass-card">
        <h3>⚖️ Hybrid Matching Engines</h3>
        <p>Analyzes demographic match values alongside text representations, calculating individual matching dimensions (MBTI compatibilities, professional fields similarity) to serve as inputs for machine learning classifiers.</p>
        <span class="badge badge-orange">Recommender</span>
        <span class="badge badge-blue">MBTI Engine</span>
    </div>
    <div class="glass-card">
        <h3>🤖 ML Re-Ranking Model</h3>
        <p>Uses a binary classifier trained on historical feedback (accept vs reject actions) to compute matching probabilities. The final score fuses hybrid similarity and ML probabilities to bubble up profiles with the highest conversion probability.</p>
        <span class="badge">Logistic Regression</span>
        <span class="badge badge-orange">Predictive Scoring</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="glass-card">
        <h3>📢 Explainable AI (XAI)</h3>
        <p>Exposes complete score breakdowns and automatically drafts human-readable explanations. Transparency helps build user trust and lets administrators see exactly which compatibility factors drove the matches.</p>
        <span class="badge badge-blue">Feature Breakdown</span>
        <span class="badge badge-green">Explainability</span>
    </div>
    <div class="glass-card">
        <h3>📊 Performance & Data Analytics</h3>
        <p>Displays classification evaluation metrics (F1-score, Precision, Recall, Confusion Matrix) and exploratory distribution insights, giving developers and evaluators full visibility into dataset and model health.</p>
        <span class="badge">Plotly Heatmaps</span>
        <span class="badge badge-orange">Dataset Insights</span>
    </div>
    """, unsafe_allow_html=True)

# Footer instructions
st.markdown("""
---
<div style='text-align: center; color: #64748b; margin-top: 20px;'>
    <p>Use the sidebar to explore user profiles, review recommendations, and inspect model coefficients and metrics.</p>
</div>
""", unsafe_allow_html=True)
