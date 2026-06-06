import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import plotly.express as px

# Add project root to sys.path to enable imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.loader import load_system
from src.utils.styles import load_css

# Page Configuration
st.set_page_config(
    page_title="Model Analytics | NexMatch AI",
    page_icon="📊",
    layout="wide"
)

# Load CSS
load_css()

# Load system objects
system = load_system()
training_dataset = system["training_dataset"]
feedback_model = system["feedback_model"]

# Title
st.markdown('<div class="gradient-header">Model Analytics & Interpretability</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-sub">Detailed evaluations, coefficients, and classification performance of the feedback learning engine</div>', unsafe_allow_html=True)

# Run verification and calculate test metrics
X = training_dataset.drop(columns=["label"])
y = training_dataset["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
predictions = feedback_model.model.predict(X_test)
probabilities = feedback_model.model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

try:
    roc_auc = roc_auc_score(y_test, probabilities)
    roc_auc_str = f"{roc_auc:.4f}"
except Exception:
    roc_auc = 0.0
    roc_auc_str = "N/A"

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

# Metrics Row
st.markdown("### 📈 Evaluation Metrics (On Held-Out Test Set)")
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric("Accuracy", f"{accuracy * 100:.2f}%", help="Overall percentage of correct predictions")
with m_col2:
    st.metric("Precision", f"{precision * 100:.2f}%", help="True Acceptances / Predicted Acceptances (minimizes false alarms)")
with m_col3:
    st.metric("Recall", f"{recall * 100:.2f}%", help="True Acceptances / Actual Acceptances (minimizes missed matches)")
with m_col4:
    st.metric("F1 Score", f"{f1 * 100:.2f}%", help="Harmonic mean of Precision and Recall")
with m_col5:
    st.metric("ROC AUC", roc_auc_str, help="Ability of the model to distinguish between Accept and Reject actions")

st.markdown("<br/>", unsafe_allow_html=True)

# Visualizations Row
col_vis_1, col_vis_2 = st.columns([1, 1])

with col_vis_1:
    st.markdown("### 📊 Logistic Regression Feature Coefficients")
    
    coefs = feedback_model.model.coef_[0]
    features = X.columns.tolist()
    
    coef_df = pd.DataFrame({
        "Feature Match Heuristics": [f.replace("_", " ").title() for f in features],
        "Coefficient (Weight)": coefs
    }).sort_values(by="Coefficient (Weight)", ascending=True)
    
    fig_coef = px.bar(
        coef_df,
        x="Coefficient (Weight)",
        y="Feature Match Heuristics",
        orientation="h",
        color="Coefficient (Weight)",
        color_continuous_scale="Blues",
        template="plotly_dark"
    )
    fig_coef.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter", size=11)
    )
    
    st.plotly_chart(fig_coef, use_container_width=True)

with col_vis_2:
    st.markdown("### 🔲 Confusion Matrix Heatmap")
    
    # Grid of confusion matrix
    cm_grid = [
        [tn, fp],
        [fn, tp]
    ]
    
    fig_cm = px.imshow(
        cm_grid,
        labels=dict(x="Predicted Action", y="Actual Feedback Action", color="Count"),
        x=["Reject (0)", "Accept (1)"],
        y=["Reject (0)", "Accept (1)"],
        text_auto=True,
        color_continuous_scale="Purples",
        template="plotly_dark"
    )
    
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter", size=11)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

# Details grid
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("### 🔍 Confusion Matrix Detail Values")
c_d1, c_d2, c_d3, c_d4 = st.columns(4)
with c_d1:
    st.markdown(f"""
    <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; padding: 12px; border-radius: 8px;">
        <small style="color: #94a3b8; font-weight: 500;">True Positives (TP)</small>
        <h2 style="color: #34d399; margin: 5px 0 0;">{tp}</h2>
        <span style="font-size: 0.8rem; color: #64748b;">Accurate Accept Matches</span>
    </div>
    """, unsafe_allow_html=True)
with c_d2:
    st.markdown(f"""
    <div style="background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; padding: 12px; border-radius: 8px;">
        <small style="color: #94a3b8; font-weight: 500;">False Positives (FP)</small>
        <h2 style="color: #f87171; margin: 5px 0 0;">{fp}</h2>
        <span style="font-size: 0.8rem; color: #64748b;">Accept matches that failed</span>
    </div>
    """, unsafe_allow_html=True)
with c_d3:
    st.markdown(f"""
    <div style="background: rgba(251, 146, 60, 0.08); border-left: 4px solid #fb923c; padding: 12px; border-radius: 8px;">
        <small style="color: #94a3b8; font-weight: 500;">False Negatives (FN)</small>
        <h2 style="color: #fdba74; margin: 5px 0 0;">{fn}</h2>
        <span style="font-size: 0.8rem; color: #64748b;">Missed Accept matches</span>
    </div>
    """, unsafe_allow_html=True)
with c_d4:
    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.08); border-left: 4px solid #6366f1; padding: 12px; border-radius: 8px;">
        <small style="color: #94a3b8; font-weight: 500;">True Negatives (TN)</small>
        <h2 style="color: #818cf8; margin: 5px 0 0;">{tn}</h2>
        <span style="font-size: 0.8rem; color: #64748b;">Accurate Reject Matches</span>
    </div>
    """, unsafe_allow_html=True)
