import streamlit as st
import pandas as pd
from pathlib import Path
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
from src.utils.loader import load_system, retrain_feedback_model, clear_system_caches
from src.utils.model_manager import ModelManager
from src.utils.audit_logger import log_event, load_audit_log, get_audit_event_types
from src.utils.validators import calculate_profile_completeness
from src.utils.data_manager import (
    load_credentials_raw,
    disable_account,
    enable_account,
    update_password_hash,
    hash_password,
    is_account_locked,
)
from src.utils.password_reset import generate_reset_token

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"



def render_admin_dashboard(system):
    """
    Renders the Admin Dashboard showing system health, KPIs,
    and matching pipeline architecture.
    """
    users_df = system["users_df"]
    feedback_df = system["feedback_df"]
    training_dataset = system["training_dataset"]
    feedback_model = system["feedback_model"]
    metadata = ModelManager.load_metadata()

    st.markdown('<div class="gradient-header">Admin Control Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">System-wide performance monitoring, metrics tracker, and model indicators</div>', unsafe_allow_html=True)

    # Core Stats
    total_users = len(users_df)
    total_feedback = len(feedback_df)
    acceptance_rate = (feedback_df["action"].mean()) * 100

    # Compute Metrics
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

    st.markdown("### 📊 Platform Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Users", f"{total_users:,}", help="Count of registered user profiles")
    with col2:
        st.metric("Feedback Records", f"{total_feedback:,}", help="Logged match interactions")
    with col3:
        st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%", help="Percentage of recommended pairs accepted")
    with col4:
        st.metric("Model Accuracy", f"{accuracy:.2f}%", help="Classifier accuracy evaluated on held-out test data")
    with col5:
        st.metric("ROC AUC Score", f"{roc_auc:.4f}", help="Area under the ROC curve representing ranking power")

    if metadata:
        st.caption(
            "Latest deployed model trained at "
            f"{metadata.get('last_trained_at', 'unknown')} "
            f"with {metadata.get('feedback_rows_at_train', 0)} feedback rows."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline Architecture Diagram
    st.markdown("### 🛠️ Multi-Stage Recommendation Pipeline Architecture")
    pipeline_html = """
    <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; background: rgba(255, 255, 255, 0.02); padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 30px;">
        <div style="text-align: center; flex: 1;">
            <div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
                👤 User Profile
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Raw profile & text inputs</div>
        </div>
        <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
        <div style="text-align: center; flex: 1;">
            <div style="background: linear-gradient(135deg, #10b981, #047857); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
                📝 TF-IDF Similarity
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Text embeddings & Cosine Similarity</div>
        </div>
        <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
        <div style="text-align: center; flex: 1;">
            <div style="background: linear-gradient(135deg, #f59e0b, #b45309); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
                ⚖️ Hybrid Recommender
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Weights: MBTI, Skills, Intent</div>
        </div>
        <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
        <div style="text-align: center; flex: 1;">
            <div style="background: linear-gradient(135deg, #ec4899, #be185d); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
                🎯 Candidate Gen
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Retrieval Stage: top 30 pool</div>
        </div>
        <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
        <div style="text-align: center; flex: 1;">
            <div style="background: linear-gradient(135deg, #8b5cf6, #6d28d9); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
                🤖 ML Re-Ranking
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Ranking Stage: Logistic Regression</div>
        </div>
        <div style="font-size: 1.5rem; color: #a855f7; padding: 0 10px;">➔</div>
        <div style="text-align: center; flex: 1;">
            <div style="background: linear-gradient(135deg, #06b6d4, #0891b2); padding: 12px; border-radius: 8px; font-weight: bold; color: white; font-size: 0.9rem;">
                📢 Explain Match
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Exploratory metrics & popup dialog</div>
        </div>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)


def render_user_explorer(system):
    """
    Renders User Explorer. Browse profiles and inspect details.
    """
    users_df = system["users_df"]
    st.markdown('<div class="gradient-header">User Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Browse, filter, and inspect detailed user profiles in the network</div>', unsafe_allow_html=True)

    st.sidebar.markdown("### 🔍 Filter Profiles")
    search_query = st.sidebar.text_input("Search by Name or ID", "")
    
    professions = ["All"] + sorted(list(users_df["profession"].dropna().unique()))
    selected_prof = st.sidebar.selectbox("Profession", professions)
    
    locations = ["All"] + sorted(list(users_df["location"].dropna().unique()))
    selected_loc = st.sidebar.selectbox("Location", locations)
    
    mbtis = ["All"] + sorted(list(users_df["mbti"].dropna().unique()))
    selected_mbti = st.sidebar.selectbox("MBTI Type", mbtis)
    
    career_goals = ["All"] + sorted(list(users_df["career_goal"].dropna().unique()))
    selected_goal = st.sidebar.selectbox("Career Goal", career_goals)

    filtered_df = users_df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df["name"].str.contains(search_query, case=False, na=False) |
            filtered_df["user_id"].str.contains(search_query, case=False, na=False)
        ]
    if selected_prof != "All":
        filtered_df = filtered_df[filtered_df["profession"] == selected_prof]
    if selected_loc != "All":
        filtered_df = filtered_df[filtered_df["location"] == selected_loc]
    if selected_mbti != "All":
        filtered_df = filtered_df[filtered_df["mbti"] == selected_mbti]
    if selected_goal != "All":
        filtered_df = filtered_df[filtered_df["career_goal"] == selected_goal]

    if filtered_df.empty:
        st.sidebar.warning("No users match filters. Showing all users.")
        display_df = users_df
    else:
        display_df = filtered_df

    user_list = display_df["user_id"].tolist()
    
    default_idx = 0
    selected_user_id = st.sidebar.selectbox("Select User ID", user_list, index=default_idx)
    st.session_state["selected_user"] = selected_user_id

    user_row = users_df[users_df["user_id"] == selected_user_id].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    with col1:
        skills_list = [s.strip() for s in str(user_row["skills"]).split(",") if s.strip()]
        traits_list = [t.strip() for t in str(user_row.get("traits", "")).split(",") if t.strip()]
        skills_badges = "".join([f'<span class="badge badge-blue">{s}</span>' for s in skills_list])
        traits_badges = "".join([f'<span class="badge">{t}</span>' for t in traits_list])
        
        st.markdown(f"""
        <div class="glass-card">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%); margin: 0 auto 10px; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: bold; color: white;">
                    {user_row["name"][0] if isinstance(user_row["name"], str) and len(user_row["name"]) > 0 else "U"}
                </div>
                <h2 style="margin: 0; font-size: 1.6rem; color: #f8fafc;">{user_row["name"]}</h2>
                <p style="margin: 5px 0 0; color: #a855f7; font-weight: 500;">{user_row["profession"]}</p>
                <p style="margin: 2px 0 0; font-size: 0.85rem; color: #64748b;">ID: {user_row["user_id"]}</p>
            </div>
            <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;"/>
            <div class="profile-item"><span class="profile-label">📍 Location:</span><span class="profile-value">{user_row["location"]}</span></div>
            <div class="profile-item"><span class="profile-label">💼 Experience:</span><span class="profile-value">{user_row["experience_years"]} years</span></div>
            <div class="profile-item"><span class="profile-label">🧬 MBTI Type:</span><span class="badge badge-green" style="margin: 0;">{user_row["mbti"]}</span></div>
            <div class="profile-item"><span class="profile-label">🎓 Education:</span><span class="profile-value">{user_row.get("education", "N/A")}</span></div>
            <div class="profile-item"><span class="profile-label">🎯 Career Goal:</span><span class="profile-value">{user_row["career_goal"]}</span></div>
            <div class="profile-item"><span class="profile-label">🤝 Intent:</span><span class="profile-value">{user_row["networking_intent"]}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="height: 100%;">
            <h3 style="border-left: 4px solid #a855f7; padding-left: 8px;">📝 Professional Summary</h3>
            <p style="color: #cbd5e1; line-height: 1.6; font-size: 1.05rem; margin-bottom: 20px;">
                {user_row["professional_summary"]}
            </p>
            
            <h3 style="border-left: 4px solid #3b82f6; padding-left: 8px;">👤 About Me</h3>
            <p style="color: #cbd5e1; line-height: 1.6; font-size: 1.05rem; margin-bottom: 20px;">
                {user_row["about_me"]}
            </p>
            
            <h3 style="border-left: 4px solid #10b981; padding-left: 8px;">🛠️ Core Skills</h3>
            <div style="margin-bottom: 20px;">
                {skills_badges}
            </div>
            
            <h3 style="border-left: 4px solid #f59e0b; padding-left: 8px;">🧬 Personality Traits</h3>
            <div style="margin-bottom: 20px;">
                {traits_badges}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_model_analytics(system):
    """
    Renders Model Analytics view. Heatmaps, coefficients, metrics.
    """
    training_dataset = system["training_dataset"]
    feedback_model = system["feedback_model"]

    st.markdown('<div class="gradient-header">Model Analytics & Interpretability</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Detailed evaluations, coefficients, and classification performance of the feedback learning engine</div>', unsafe_allow_html=True)

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
        roc_auc_str = "N/A"

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    st.markdown("### 📈 Evaluation Metrics (On Held-Out Test Set)")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
    with m_col2:
        st.metric("Precision", f"{precision * 100:.2f}%")
    with m_col3:
        st.metric("Recall", f"{recall * 100:.2f}%")
    with m_col4:
        st.metric("F1 Score", f"{f1 * 100:.2f}%")
    with m_col5:
        st.metric("ROC AUC", roc_auc_str)

    st.markdown("<br/>", unsafe_allow_html=True)

    col_vis_1, col_vis_2 = st.columns(2)
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
        cm_grid = [[tn, fp], [fn, tp]]
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


def render_dataset_insights(system):
    """
    Renders Dataset Insights. Bar plots and pie charts of EDA.
    """
    users_df = system["users_df"]
    feedback_df = system["feedback_df"]

    st.markdown('<div class="gradient-header">Dataset Insights & EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Exploratory Data Analysis and distribution statistics of the user network and feedback logs</div>', unsafe_allow_html=True)

    # Core Stats
    total_users = len(users_df)
    total_feedback = len(feedback_df)
    accept_rate = (feedback_df["action"].mean()) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total User Profiles", f"{total_users:,}")
    with col2:
        st.metric("Total Feedback Logs", f"{total_feedback:,}")
    with col3:
        st.metric("Acceptance Rate", f"{accept_rate:.2f}%")

    st.markdown("<br/>", unsafe_allow_html=True)

    col_r1_1, col_r1_2 = st.columns(2)
    with col_r1_1:
        st.markdown("### 💼 User Profession Distribution")
        prof_counts = users_df["profession"].value_counts().reset_index()
        prof_counts.columns = ["Profession", "Count"]
        prof_counts = prof_counts.sort_values(by="Count", ascending=True)
        fig_prof = px.bar(
            prof_counts,
            x="Count",
            y="Profession",
            orientation="h",
            color="Count",
            color_continuous_scale="Purples",
            template="plotly_dark"
        )
        fig_prof.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=10)
        )
        st.plotly_chart(fig_prof, use_container_width=True)

    with col_r1_2:
        st.markdown("### 🧬 User MBTI Distribution")
        mbti_counts = users_df["mbti"].value_counts().reset_index()
        mbti_counts.columns = ["MBTI Type", "Count"]
        fig_mbti = px.pie(
            mbti_counts,
            values="Count",
            names="MBTI Type",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_dark"
        )
        fig_mbti.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=10)
        )
        st.plotly_chart(fig_mbti, use_container_width=True)

    col_r2_1, col_r2_2 = st.columns(2)
    with col_r2_1:
        st.markdown("### 🎯 User Career Goal Distribution")
        goal_counts = users_df["career_goal"].value_counts().reset_index()
        goal_counts.columns = ["Career Goal", "Count"]
        goal_counts = goal_counts.sort_values(by="Count", ascending=True)
        fig_goal = px.bar(
            goal_counts,
            x="Count",
            y="Career Goal",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        fig_goal.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=10)
        )
        st.plotly_chart(fig_goal, use_container_width=True)

    with col_r2_2:
        st.markdown("### 🤝 Acceptance vs Rejection Distribution")
        action_counts = feedback_df["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        action_counts["Action"] = action_counts["Action"].map({1: "Accept (1)", 0: "Reject (0)"})
        fig_action = px.pie(
            action_counts,
            values="Count",
            names="Action",
            hole=0.4,
            color="Action",
            color_discrete_map={"Accept (1)": "#10b981", "Reject (0)": "#ef4444"},
            template="plotly_dark"
        )
        fig_action.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(family="Inter", size=10)
        )
        st.plotly_chart(fig_action, use_container_width=True)


def render_explainability(system):
    """
    Renders pairwise explainability matching details.
    """
    users_df = system["users_df"]
    recommender = system["recommender"]

    st.markdown('<div class="gradient-header">Recommendation Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Deep-dive comparison: Analyze matching compatibility dimensions between any two user profiles</div>', unsafe_allow_html=True)

    user_list = users_df["user_id"].tolist()
    col_sel_1, col_sel_2 = st.columns(2)
    default_user_a = st.session_state.get("selected_user", user_list[0])
    if default_user_a not in user_list:
        default_user_a = user_list[0]

    with col_sel_1:
        user_a_id = st.selectbox("Select User A (Source Profile)", user_list, index=user_list.index(default_user_a))
        st.session_state["selected_user"] = user_a_id

    user_list_b = [uid for uid in user_list if uid != user_a_id]
    default_b_idx = 0
    if "selected_user_b" in st.session_state and st.session_state["selected_user_b"] in user_list_b:
        default_b_idx = user_list_b.index(st.session_state["selected_user_b"])

    with col_sel_2:
        user_b_id = st.selectbox("Select User B (Target Profile)", user_list_b, index=default_b_idx)
        st.session_state["selected_user_b"] = user_b_id

    user_a = users_df[users_df["user_id"] == user_a_id].iloc[0]
    user_b = users_df[users_df["user_id"] == user_b_id].iloc[0]

    st.markdown("### 👥 Profile Comparison")
    col_p1, col_p2 = st.columns(2)

    def render_compact_card(row, title_prefix):
        skills_list = [s.strip() for s in str(row["skills"]).split(",") if s.strip()]
        skills_badges = "".join([f'<span class="badge badge-blue">{s}</span>' for s in skills_list])
        return f"""
        <div class="glass-card" style="min-height: 250px;">
            <h4 style="color: #60a5fa; margin-top:0;">{title_prefix}: {row['name']}</h4>
            <div style="font-size: 0.85rem; color: #a855f7; font-weight: bold; margin-bottom: 10px;">ID: {row['user_id']} | {row['profession']}</div>
            <div class="profile-item"><span class="profile-label" style="width: 100px;">📍 Location:</span><span class="profile-value">{row['location']}</span></div>
            <div class="profile-item"><span class="profile-label" style="width: 100px;">💼 Experience:</span><span class="profile-value">{row['experience_years']} years</span></div>
            <div class="profile-item"><span class="profile-label" style="width: 100px;">🧬 MBTI:</span><span class="badge badge-green" style="margin:0;">{row['mbti']}</span></div>
            <div class="profile-item"><span class="profile-label" style="width: 100px;">🎯 Career Goal:</span><span class="profile-value">{row['career_goal']}</span></div>
            <div style="margin-top: 10px;"><b>Skills:</b> {skills_badges}</div>
        </div>
        """

    with col_p1:
        st.markdown(render_compact_card(user_a, "User A"), unsafe_allow_html=True)
    with col_p2:
        st.markdown(render_compact_card(user_b, "User B"), unsafe_allow_html=True)

    score = recommender.compatibility_score(user_a_id, user_b_id)

    st.markdown("### 📊 Matching Dimension Breakdown")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        def render_explain_bar(label, key):
            val = score[key]
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin-bottom: 2px; font-size: 0.85rem;">
                <span style="color: #cbd5e1;">{label}</span>
                <span style="font-weight: bold;">{val}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(val / 100.0)

        render_explain_bar("📝 Profile Text Similarity (TF-IDF Cosine)", "text_similarity")
        render_explain_bar("🧬 MBTI Compatibility", "mbti_score")
        render_explain_bar("💼 Profession Match", "profession_score")
        render_explain_bar("🎯 Career Goal Match", "career_goal_score")

    with col_b2:
        render_explain_bar("🛠️ Skills Match", "skills_score")
        render_explain_bar("📈 Experience Level Match", "experience_score")
        render_explain_bar("📍 Location Match", "location_score")
        render_explain_bar("🤝 Networking Intent Match", "networking_intent_score")

    # Dynamic explanation
    reasons = []
    if score["profession_score"] == 100:
        reasons.append("share the exact same profession of " + user_a["profession"])
    elif score["profession_score"] >= 70:
        reasons.append(f"work in closely related domains ({user_a['profession']} & {user_b['profession']})")
        
    skills_a = set(s.strip().lower() for s in str(user_a["skills"]).split(",") if s.strip())
    skills_b = set(s.strip().lower() for s in str(user_b["skills"]).split(",") if s.strip())
    common = skills_a & skills_b
    if common:
        reasons.append(f"have overlapping technical expertise in {', '.join([s.title() for s in list(common)[:3]])}")
        
    if score["mbti_score"] >= 80:
        reasons.append(f"possess highly compatible psychological types (MBTI: {user_a['mbti']} and {user_b['mbti']})")
    elif score["mbti_score"] >= 50:
        reasons.append(f"have compatible communication styles (MBTI: {user_a['mbti']} and {user_b['mbti']})")
        
    if score["career_goal_score"] >= 70:
        reasons.append(f"are both focused on {user_a['career_goal']} career pathways")
        
    exp_diff = abs(user_a["experience_years"] - user_b["experience_years"])
    if exp_diff <= 2:
        reasons.append("possess similar levels of industry seniority")
    elif exp_diff <= 5:
        reasons.append("are at complementary stages of their career paths")
        
    if score["location_score"] == 100:
        reasons.append(f"are both located in {user_a['location']}, facilitating in-person collaboration")

    if not reasons:
        dynamic_paragraph = "This matching shows mild peripheral alignment across several networking factors."
    elif len(reasons) == 1:
        dynamic_paragraph = f"This recommendation is driven primarily because both users {reasons[0]}."
    else:
        dynamic_paragraph = "This recommendation is strong because both users " + ", ".join(reasons[:-1]) + ", and " + reasons[-1] + "."

    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #10b981; margin-top: 20px;">
        <h3 style="color: #34d399; margin: 0 0 10px;">📢 Dynamic Match Explanation</h3>
        <p style="color: #f1f5f9; font-size: 1.05rem; line-height: 1.6; margin: 0;">
            "{dynamic_paragraph}"
        </p>
        <div style="margin-top: 15px; font-weight: bold; color: #10b981; font-size: 1.1rem; text-align: right;">
            Overall Match Score: {score['final_score']:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_system_status(system):
    """
    Renders System Status. Performance indicators, retrain model.
    """
    users_df = system["users_df"]
    feedback_df = system["feedback_df"]
    metadata = ModelManager.load_metadata()

    st.markdown('<div class="gradient-header">System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Operational health, artifact checks, and model retraining controls</div>', unsafe_allow_html=True)

    feedback_rows_at_train = int(metadata.get("feedback_rows_at_train", 0)) if metadata else 0
    pending_feedback = max(len(feedback_df) - feedback_rows_at_train, 0)
    last_trained_at = metadata.get("last_trained_at", "Not recorded yet") if metadata else "Not recorded yet"
    acceptance_rate = feedback_df["action"].mean() * 100 if not feedback_df.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", f"{len(users_df):,}")
    with col2:
        st.metric("Feedback Rows", f"{len(feedback_df):,}")
    with col3:
        st.metric("Acceptance Rate", f"{acceptance_rate:.2f}%")
    with col4:
        st.metric("Pending Feedback", f"{pending_feedback:,}")

    artifact_col1, artifact_col2 = st.columns(2)
    with artifact_col1:
        st.markdown("### Saved Artifacts Check")
        tfidf_exists = (MODEL_DIR / "tfidf_vectorizer.pkl").exists()
        model_exists = (MODEL_DIR / "feedback_model.pkl").exists()
        st.write(f"TF-IDF Vectorizer: **{'Available' if tfidf_exists else 'Missing'}**")
        st.write(f"Feedback Model (Logistic Regression): **{'Available' if model_exists else 'Missing'}**")
        st.write(f"Last Trained At: **{last_trained_at}**")
        if metadata and "accuracy" in metadata:
            st.write(f"Last Recorded Accuracy: **{metadata['accuracy'] * 100:.2f}%**")

    with artifact_col2:
        st.markdown("### Batch Retraining")
        st.markdown("Retrain the ML scoring model on accumulated feedback data to update match predictions.")
        if st.button("Retrain Feedback Model Now", type="primary"):
            with st.spinner("Executing batch training..."):
                accuracy = retrain_feedback_model()
            st.success(f"Model retrained successfully! Latest model accuracy: {accuracy * 100:.2f}%")
            st.rerun()

    # Section 3: Email & Security Status
    st.markdown("---")
    st.markdown("### 📧 Email & Security Status")
    
    from src.utils.config import Config
    from src.utils.email_service import check_smtp_status
    from src.utils.password_reset import get_pending_resets_count
    from src.utils.notifications import get_pending_notifications_count
    
    sec_col1, sec_col2, sec_col3 = st.columns(3)
    
    with sec_col1:
        smtp_ok, smtp_msg = check_smtp_status()
        status_text = "🟢 Connected" if smtp_ok else f"🔴 Offline ({smtp_msg})"
        if not Config.ENABLE_EMAILS:
            status_text = "⚪ Disabled"
        st.markdown(f"**Email Service (SMTP):** {status_text}")
        st.write(f"Environment: `{Config.APP_ENV}`")
        st.write(f"App Version: `{Config.APP_VERSION}`")
        
    with sec_col2:
        pending_resets = get_pending_resets_count()
        st.markdown(f"**Pending Password Resets:** `{pending_resets}`")
        st.write(f"Reset Link Token Expiry: **{Config.PASSWORD_RESET_EXPIRY_MINUTES} minutes**")
        st.write(f"Retrain Threshold: **{Config.FEEDBACK_RETRAIN_THRESHOLD} feedbacks**")
        
    with sec_col3:
        pending_notifs = get_pending_notifications_count()
        st.markdown(f"**Pending Connections Queue:** `{pending_notifs}`")
        st.write("Digest Dispatch: **Manual/Triggered**")
        st.write(f"Auth Protocol: **bcrypt + Salt Hashing**")


def render_recommendation_quality(system):
    """
    Phase 7.9 — Recommendation Quality Audit for Admin.
    Displays:
    - Top recommended professions and career goals (bar charts)
    - Profession Diversity Index (normalized entropy)
    - Recommendation Distribution by Profession
    - Average final recommendation score
    - Acceptance Rate breakdown
    Uses recommendation_history.csv for accurate post-hoc audit.
    """
    import math
    from src.utils.data_manager import load_recommendation_history

    st.markdown('<div class="gradient-header">Recommendation Quality Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Post-hoc analysis of recommendation diversity, quality, and acceptance patterns</div>', unsafe_allow_html=True)

    users_df = system["users_df"]
    feedback_df = system["feedback_df"]

    # Load recommendation history
    history_df = load_recommendation_history()

    if history_df.empty:
        st.warning("No recommendation history found. Ask users to generate recommendations first.")
        return

    # Merge history with user profiles to get profession/career_goal of recommended users
    rec_merged = history_df.merge(
        users_df[["user_id", "profession", "career_goal", "location"]].rename(columns={"user_id": "recommended_user_id"}),
        on="recommended_user_id",
        how="left"
    )

    st.markdown("---")

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total_recs = len(history_df)
    unique_users_served = history_df["user_id"].nunique()
    unique_candidates = history_df["recommended_user_id"].nunique()
    avg_score = history_df["score"].mean() if "score" in history_df.columns else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Recommendations", f"{total_recs:,}")
    with kpi2:
        st.metric("Users Served", f"{unique_users_served:,}")
    with kpi3:
        st.metric("Unique Candidates Shown", f"{unique_candidates:,}")
    with kpi4:
        st.metric("Avg Score", f"{avg_score:.1f}%" if avg_score > 0 else "N/A")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Profession Distribution ───────────────────────────────────────────────
    st.markdown("### 🏢 Recommendation Distribution by Profession")
    if "profession" in rec_merged.columns:
        prof_counts = rec_merged["profession"].value_counts().reset_index()
        prof_counts.columns = ["Profession", "Count"]

        # Profession Diversity Index (normalized Shannon entropy)
        total = prof_counts["Count"].sum()
        probs = prof_counts["Count"] / total
        raw_entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(prof_counts)) if len(prof_counts) > 1 else 1
        diversity_index = raw_entropy / max_entropy if max_entropy > 0 else 0

        div_col, chart_col = st.columns([1, 3])
        with div_col:
            color = "#10b981" if diversity_index >= 0.75 else ("#f59e0b" if diversity_index >= 0.5 else "#ef4444")
            rating = "High" if diversity_index >= 0.75 else ("Moderate" if diversity_index >= 0.5 else "Low")
            st.markdown(f"""
            <div style="background: rgba(15,23,42,0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; text-align: center; margin-top: 10px;">
                <div style="font-size: 2rem; font-weight: bold; color: {color};">{diversity_index:.2f}</div>
                <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">Profession Diversity Index</div>
                <div style="color: {color}; font-size: 0.75rem; margin-top: 6px; font-weight: 600;">{rating} Diversity</div>
                <div style="color: #64748b; font-size: 0.7rem; margin-top: 4px;">(0=concentrated, 1=uniform)</div>
            </div>
            """, unsafe_allow_html=True)

        with chart_col:
            fig_prof = px.bar(
                prof_counts.head(10),
                x="Count",
                y="Profession",
                orientation="h",
                color="Count",
                color_continuous_scale="blues",
                title="Top 10 Recommended Professions"
            )
            fig_prof.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=10, t=40, b=10),
                coloraxis_showscale=False,
            )
            fig_prof.update_traces(marker_line_width=0)
            st.plotly_chart(fig_prof, use_container_width=True)

    # ── Career Goal Distribution ──────────────────────────────────────────────
    st.markdown("### 🎯 Top Recommended Career Goals")
    if "career_goal" in rec_merged.columns:
        goal_counts = rec_merged["career_goal"].value_counts().reset_index()
        goal_counts.columns = ["Career Goal", "Count"]

        fig_goal = px.bar(
            goal_counts.head(8),
            x="Career Goal",
            y="Count",
            color="Count",
            color_continuous_scale="purples",
            title="Distribution of Career Goals in Recommendations"
        )
        fig_goal.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"),
            margin=dict(l=0, r=10, t=40, b=10),
            coloraxis_showscale=False,
        )
        fig_goal.update_traces(marker_line_width=0)
        st.plotly_chart(fig_goal, use_container_width=True)

    # ── Acceptance Rate by Profession ──────────────────────────────────────────
    st.markdown("### ✅ Acceptance Rate by Profession")
    if not feedback_df.empty and "profession" in rec_merged.columns:
        # Merge feedback with profession info
        fb_merged = feedback_df.merge(
            users_df[["user_id", "profession"]].rename(columns={"user_id": "target_user_id"}),
            on="target_user_id",
            how="left"
        ) if "target_user_id" in feedback_df.columns else pd.DataFrame()

        if not fb_merged.empty and "profession" in fb_merged.columns:
            acc_by_prof = fb_merged.groupby("profession")["action"].agg(
                Accepted=lambda x: (x == 1).sum(),
                Total="count"
            ).reset_index()
            acc_by_prof["Acceptance Rate (%)"] = (acc_by_prof["Accepted"] / acc_by_prof["Total"] * 100).round(1)
            acc_by_prof = acc_by_prof.sort_values("Acceptance Rate (%)", ascending=False).head(10)

            fig_acc = px.bar(
                acc_by_prof,
                x="Acceptance Rate (%)",
                y="profession",
                orientation="h",
                color="Acceptance Rate (%)",
                color_continuous_scale="greens",
                title="Acceptance Rate by Recommended Profession"
            )
            fig_acc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=10, t=40, b=10),
                coloraxis_showscale=False,
            )
            fig_acc.update_traces(marker_line_width=0)
            st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("Feedback data does not contain target_user_id column. Skipping per-profession acceptance chart.")
    else:
        st.info("Insufficient feedback data for profession-level acceptance analysis.")

    # ── Recommendation Collapse Detection ─────────────────────────────────────
    st.markdown("### 🔁 Repeated Recommendation Detection")
    if "recommended_user_id" in history_df.columns and "user_id" in history_df.columns:
        repeat_df = history_df.groupby(["user_id", "recommended_user_id"]).size().reset_index(name="Times Shown")
        repeat_df = repeat_df[repeat_df["Times Shown"] > 1].sort_values("Times Shown", ascending=False)
        if repeat_df.empty:
            st.success("✅ No repeated recommendations detected.")
        else:
            st.warning(f"⚠️ {len(repeat_df)} user-candidate pairs have been recommended more than once.")
            st.dataframe(repeat_df.head(20), width='stretch', hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.8 — System Monitoring Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def render_system_monitoring(system):
    """
    Phase 8.8 — System Monitoring Dashboard.
    Shows KPIs, active users, health status (Green/Amber/Red), and recent activity.
    """
    from datetime import datetime, timedelta
    from src.utils.data_manager import load_recommendation_history

    st.markdown('<div class="gradient-header">System Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Real-time platform health, activity metrics, and operational status</div>', unsafe_allow_html=True)

    users_df = system["users_df"]
    feedback_df = system["feedback_df"]
    metadata = ModelManager.load_metadata()
    cred_df = load_credentials_raw()
    history_df = load_recommendation_history()

    total_users = len(users_df)
    total_feedback = len(feedback_df)
    acceptance_rate = feedback_df["action"].mean() * 100 if not feedback_df.empty else 0.0
    total_recs = len(history_df)
    feedback_rows_at_train = int(metadata.get("feedback_rows_at_train", 0)) if metadata else 0
    pending_feedback = max(total_feedback - feedback_rows_at_train, 0)
    last_trained_at = metadata.get("last_trained_at", "Never") if metadata else "Never"

    # Active users: logged in within last 30 days
    active_count = 0
    if "last_login" in cred_df.columns:
        cutoff = datetime.now() - timedelta(days=30)
        try:
            cred_df["_ll"] = pd.to_datetime(cred_df["last_login"], errors="coerce")
            active_count = int((cred_df["_ll"] >= cutoff).sum())
        except Exception:
            active_count = 0

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.metric("Total Users", f"{total_users:,}")
    with k2: st.metric("Active (30d)", f"{active_count:,}")
    with k3: st.metric("Recs Generated", f"{total_recs:,}")
    with k4: st.metric("Feedback Rows", f"{total_feedback:,}")
    with k5: st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
    with k6: st.metric("Pending Feedback", f"{pending_feedback:,}")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── System Health Status ──────────────────────────────────────────────────
    st.markdown("### System Health Status")
    model_ok = (MODEL_DIR / "feedback_model.pkl").exists()
    tfidf_ok = (MODEL_DIR / "tfidf_vectorizer.pkl").exists()
    from src.utils.config import Config
    from src.utils.email_service import check_smtp_status
    smtp_ok, smtp_msg = check_smtp_status()

    issues = []
    warnings = []
    if not model_ok: issues.append("Feedback model file missing")
    if not tfidf_ok: issues.append("TF-IDF vectorizer file missing")
    if pending_feedback > Config.FEEDBACK_RETRAIN_THRESHOLD:
        warnings.append(f"Pending feedback ({pending_feedback}) exceeds retrain threshold ({Config.FEEDBACK_RETRAIN_THRESHOLD})")
    if not Config.ENABLE_EMAILS:
        warnings.append("Email service is disabled in .env")

    if issues:
        health_color = "#ef4444"; health_label = "RED — Critical Issues"; health_icon = "🔴"
    elif warnings:
        health_color = "#f59e0b"; health_label = "AMBER — Warnings"; health_icon = "🟡"
    else:
        health_color = "#10b981"; health_label = "GREEN — All Systems Operational"; health_icon = "🟢"

    st.markdown(f"""
    <div style="background: rgba(15,23,42,0.5); border: 2px solid {health_color}40; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
        <div style="font-size: 1.2rem; font-weight: bold; color: {health_color};">{health_icon} {health_label}</div>
    </div>
    """, unsafe_allow_html=True)

    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown("**Component Status**")
        st.write(f"Feedback Model: **{'OK' if model_ok else 'MISSING'}**")
        st.write(f"TF-IDF Vectorizer: **{'OK' if tfidf_ok else 'MISSING'}**")
        st.write(f"Email (SMTP): **{'Connected' if smtp_ok else 'Offline'}** — {smtp_msg}")
        st.write(f"Last Retrain: **{last_trained_at}**")
    with hc2:
        if issues:
            st.error("Critical Issues:\n" + "\n".join(f"• {i}" for i in issues))
        if warnings:
            st.warning("Warnings:\n" + "\n".join(f"• {w}" for w in warnings))
        if not issues and not warnings:
            st.success("No issues detected.")

    # ── Recent Activity from Audit Log ───────────────────────────────────────
    st.markdown("### Recent Activity (Last 20 Events)")
    audit_df = load_audit_log(limit=20)
    if audit_df.empty:
        st.info("No audit events recorded yet.")
    else:
        st.dataframe(audit_df[["timestamp", "user_id", "event_type", "details"]], width='stretch', hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.6 — Admin User Management
# ─────────────────────────────────────────────────────────────────────────────
def render_user_management(system):
    """
    Phase 8.6 — Admin User Management.
    View all users with Role, Status, Last Login, Registration Date, Completeness.
    Enable/Disable accounts. Reset passwords.
    """
    st.markdown('<div class="gradient-header">User Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Manage user accounts, roles, access control, and profile completeness</div>', unsafe_allow_html=True)

    users_df = system["users_df"]
    feedback_df = system["feedback_df"]
    cred_df = load_credentials_raw()

    if cred_df.empty:
        st.warning("No credentials found.")
        return

    st.markdown("---")

    # ── Search & Filter ───────────────────────────────────────────────────────
    search_col, role_col = st.columns([3, 1])
    with search_col:
        search_q = st.text_input("Search by username or user ID", key="um_search")
    with role_col:
        role_filter = st.selectbox("Filter by Role", ["All", "user", "admin"], key="um_role")

    filtered = cred_df.copy()
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = filtered[
            filtered["username"].str.lower().str.contains(q, na=False) |
            filtered["user_id"].str.lower().str.contains(q, na=False)
        ]
    if role_filter != "All":
        filtered = filtered[filtered["role"] == role_filter]

    st.markdown(f"**Showing {len(filtered)} of {len(cred_df)} accounts**")

    # ── User Table ─────────────────────────────────────────────────────────────
    for _, cred_row in filtered.iterrows():
        uid = str(cred_row["user_id"])
        username = str(cred_row.get("username", ""))
        role = str(cred_row.get("role", "user"))
        last_login = str(cred_row.get("last_login", "Never"))
        created_at = str(cred_row.get("created_at", ""))
        failed_att = int(cred_row.get("failed_attempts", 0) or 0)
        locked, lock_mins = is_account_locked(cred_row)
        is_disabled = str(cred_row.get("locked_until", "")).strip() == "9999-12-31T23:59:59"

        status_label = "Disabled" if is_disabled else ("Locked" if locked else "Active")
        status_color = "#ef4444" if is_disabled else ("#f59e0b" if locked else "#10b981")

        # Profile completeness
        profile_match = users_df[users_df["user_id"] == uid]
        if not profile_match.empty:
            completeness, _ = calculate_profile_completeness(profile_match.iloc[0].to_dict())
        else:
            completeness = 0.0

        # Feedback stats
        user_fb = feedback_df[feedback_df["user_id"] == uid] if not feedback_df.empty else pd.DataFrame()
        fb_count = len(user_fb)
        acc_rate = (user_fb["action"].mean() * 100) if fb_count > 0 else 0.0

        with st.expander(f"{username} ({uid}) — {role.upper()} — {status_label}", expanded=False):
            info_c, action_c = st.columns([3, 2])

            with info_c:
                st.markdown(f"""
                | Field | Value |
                |-------|-------|
                | **Username** | {username} |
                | **User ID** | `{uid}` |
                | **Role** | {role} |
                | **Status** | <span style="color:{status_color};">**{status_label}**</span> |
                | **Last Login** | {last_login if last_login not in ('', 'nan') else 'Never'} |
                | **Registered** | {created_at if created_at not in ('', 'nan') else 'Unknown'} |
                | **Failed Attempts** | {failed_att} |
                | **Profile Completeness** | {completeness:.0f}% |
                | **Feedback Count** | {fb_count} |
                | **Acceptance Rate** | {acc_rate:.1f}% |
                """, unsafe_allow_html=True)

            with action_c:
                st.markdown("**Account Actions**")

                if uid != "ADMIN":
                    if is_disabled:
                        if st.button(f"Enable Account", key=f"enable_{uid}"):
                            enable_account(uid)
                            log_event("ADMIN", "ACCOUNT_ENABLED", f"uid={uid}")
                            st.success(f"Account {uid} enabled.")
                            st.rerun()
                    else:
                        if st.button(f"Disable Account", key=f"disable_{uid}", type="primary"):
                            disable_account(uid)
                            log_event("ADMIN", "ACCOUNT_DISABLED", f"uid={uid}")
                            st.warning(f"Account {uid} disabled.")
                            st.rerun()

                st.markdown("**Reset Password**")
                new_pw = st.text_input("New password", type="password", key=f"newpw_{uid}")
                if st.button("Apply Reset", key=f"reset_pw_{uid}"):
                    if len(new_pw) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        h = hash_password(new_pw)
                        update_password_hash(uid, h)
                        log_event("ADMIN", "PASSWORD_RESET", f"Admin reset password for uid={uid}")
                        st.success("Password updated successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.11 — Data Management (Export / Backup)
# ─────────────────────────────────────────────────────────────────────────────
def render_data_management(system):
    """
    Phase 8.11 — Data Management.
    Admin download buttons for all CSV data files. Read-only. No deletion.
    """
    st.markdown('<div class="gradient-header">Data Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Export and backup system data. All operations are read-only.</div>', unsafe_allow_html=True)
    st.markdown("---")

    files = {
        "feedback.csv": ("Feedback Data", "All user feedback records (Accept/Reject)"),
        "recommendation_history.csv": ("Recommendation History", "Full recommendation batch log"),
        "user_profiles.csv": ("User Profiles", "Registered user profile data"),
        "audit_log.csv": ("Audit Log", "Complete security and activity audit trail"),
        "credentials.csv": ("Credentials (Hashed)", "bcrypt-hashed credentials — no plaintext passwords"),
        "password_reset_tokens.csv": ("Reset Tokens", "Active and expired password reset tokens"),
    }

    for filename, (label, desc) in files.items():
        filepath = DATA_DIR / filename
        with st.container():
            row_c1, row_c2 = st.columns([4, 1])
            with row_c1:
                st.markdown(f"**{label}** — `{filename}`")
                st.caption(desc)
                if filepath.exists():
                    size_kb = filepath.stat().st_size / 1024
                    st.caption(f"Size: {size_kb:.1f} KB")
                else:
                    st.caption("File not yet created.")
            with row_c2:
                if filepath.exists():
                    with open(filepath, "rb") as f:
                        st.download_button(
                            label=f"Download",
                            data=f.read(),
                            file_name=filename,
                            mime="text/csv",
                            key=f"dl_{filename}"
                        )
                else:
                    st.button("Unavailable", key=f"dl_na_{filename}", disabled=True)
            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 8.7 — Audit Log Viewer (Admin)
# ─────────────────────────────────────────────────────────────────────────────
def render_audit_log_viewer(system):
    """
    Phase 8.7 — Audit Log Viewer.
    Paginated, filterable view of data/audit_log.csv for admin review.
    """
    st.markdown('<div class="gradient-header">Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="gradient-sub">Security and activity audit trail — all critical platform events</div>', unsafe_allow_html=True)
    st.markdown("---")

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        event_types = ["All"] + get_audit_event_types()
        selected_event = st.selectbox("Filter by Event Type", event_types, key="audit_event_filter")
    with f2:
        uid_filter = st.text_input("Filter by User ID", key="audit_uid_filter")
    with f3:
        page_size = st.selectbox("Rows per page", [50, 100, 200], key="audit_page_size")

    audit_df = load_audit_log(
        event_type_filter=selected_event,
        user_id_filter=uid_filter,
        limit=page_size
    )

    st.markdown(f"**Showing {len(audit_df)} records** (most recent first)")

    if audit_df.empty:
        st.info("No audit events match the selected filters.")
    else:
        # Color-code event types
        def style_row(row):
            colors = {
                "LOGIN": "background-color: rgba(16,185,129,0.06)",
                "LOGOUT": "background-color: rgba(100,116,139,0.06)",
                "LOGIN_FAILED": "background-color: rgba(239,68,68,0.08)",
                "ACCOUNT_LOCKED": "background-color: rgba(239,68,68,0.12)",
                "REGISTER": "background-color: rgba(59,130,246,0.06)",
                "ADMIN_RETRAIN": "background-color: rgba(168,85,247,0.08)",
                "ERROR": "background-color: rgba(239,68,68,0.10)",
            }
            et = str(row["event_type"])
            color = colors.get(et, "")
            return [color] * len(row)

        st.dataframe(
            audit_df[["timestamp", "user_id", "event_type", "details"]],
            width='stretch',
            hide_index=True,
        )

        # Download audit log
        csv_data = audit_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Log",
            data=csv_data,
            file_name="audit_log_filtered.csv",
            mime="text/csv",
            key="audit_download_btn"
        )
