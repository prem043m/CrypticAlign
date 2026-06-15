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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


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
        traits_list = [t.strip() for t in str(user_row.get("traits", "")).split(",") if s.strip()]
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
        st.plotly_chart(fig_coef, width="stretch")

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
        st.plotly_chart(fig_cm, width="stretch")


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
        st.plotly_chart(fig_prof, width="stretch")

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
        st.plotly_chart(fig_mbti, width="stretch")

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
        st.plotly_chart(fig_goal, width="stretch")

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
        st.plotly_chart(fig_action, width="stretch")


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
