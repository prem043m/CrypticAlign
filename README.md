# � NextMatchAI

> AI-powered professional networking recommender that helps users discover high-fit collaborators, mentors, and peers through intelligent matching.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Table of Contents

| Section | Description |
|---------|-------------|
| [What is NextMatchAI?](#-what-is-nextmatchai) | Project overview and motivation |
| [Key Features](#-key-features) | Capabilities at a glance |
| [Quick Start](#-quick-start) | Get running in 3 minutes |
| [Architecture](#-architecture) | System design and pipeline |
| [How It Works](#-how-it-works) | Step-by-step recommendation flow |
| [Model Training & Evaluation](#-model-training--evaluation) | ML pipeline and metrics |
| [Project Structure](#-project-structure) | Repository layout |
| [Tech Stack](#-tech-stack) | Technologies and libraries |
| [Usage Examples](#-usage-examples) | Python API walkthrough |
| [Deliverables](#-deliverables) | Project evaluation checklist |
| [Performance Results](#-performance-results) | Benchmark metrics and analysis |
| [Strengths & Limitations](#-strengths--limitations) | Honest assessment |
| [Future Improvements](#-future-improvements) | Roadmap |
| [Contributing](#-contributing) | How to contribute |
| [License](#-license) | MIT License |
| [Contact](#-contact) | Author information |

---

## 🧠 What is NextMatchAI?

NextMatchAI is a portfolio-ready recommendation system for professional networking platforms. It combines natural language processing, rule-based compatibility scoring, and machine-learning re-ranking to surface meaningful connections between professionals based on skills, experience, interests, career goals, and personality fit.

The system evaluates multiple compatibility dimensions and continuously improves through feedback, making it suitable for real-world use cases such as mentorship discovery, collaboration matching, and career growth recommendations. Built with a Streamlit interface and modular Python architecture, the project demonstrates a complete end-to-end workflow from synthetic data generation to model evaluation and deployment.

### 🎯 Problem Statement

> How can we help professionals discover relevant connections beyond simple keyword matching, while still keeping the experience explainable, adaptive, and useful?

NextMatchAI addresses this challenge through a data-driven approach that improved recommendation acceptance from roughly 40% to 67% through its adaptive learning loop.

---

## ✨ Key Features

| Category | Feature |
|----------|---------|
| 🤖 **ML Pipeline** | Two-stage hybrid recommender (TF-IDF + Logistic Regression) |
| 📊 **8 Compatibility Dimensions** | Text, MBTI, profession, career goals, location, experience, skills, networking intent |
| 🔄 **Adaptive Learning** | Feedback loop that improves with every accept/reject signal |
| 🧬 **Personality Matching** | MBTI-based compatibility matrix for deeper profiling |
| 📈 **Performance Analytics** | Built-in evaluation dashboard with precision/recall/F1 metrics |
| 💡 **Explainability Engine** | Human-readable explanations for every recommendation |
| 🎨 **Interactive UI** | Streamlit-powered admin portal and user portal |
| 🔐 **Admin Controls** | User management, model retraining, audit logging |
| 📦 **Model Persistence** | Serialized model artifacts with metadata tracking |
| 🚀 **Deploy-Ready** | Render deployment config included |

---

## 🚀 Quick Start

```bash
# 1. Clone and navigate
git clone <repository-url>
cd Intelligent_Recommender_System

# 2. Create virtual environment
python -m venv .ipenv
.ipenv\Scripts\activate         # Windows
# source .ipenv/bin/activate    # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run src/app.py
```

> [!TIP]
> For detailed setup instructions including environment configuration and troubleshooting, see **[quickstart.md](quickstart.md)**. For deployment to Render, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## 🏗️ Architecture

nextmatchAi uses a **two-stage hybrid pipeline** that balances recall (finding relevant candidates) with precision (ranking them accurately):

```
                        ┌──────────────────────────────┐
                        │      USER PROFILES (300)      │
                        │   15 fields per user in CSV   │
                        └──────────────┬───────────────┘
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║    STAGE 1: CANDIDATE GENERATION    ║
                    ╠════════════════════════════════════╣
                    ║                                     ║
                    ║   Profile Text ──► TF-IDF Vectors   ║
                    ║          │                           ║
                    ║          ▼                           ║
                    ║   Cosine Similarity Matrix           ║
                    ║          │                           ║
                    ║          ▼                           ║
                    ║   Top 30 Candidates (recall pool)    ║
                    ║                                     ║
                    ╚══════════════════╤══════════════════╝
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║      STAGE 2: ML RE-RANKING         ║
                    ╠════════════════════════════════════╣
                    ║                                     ║
                    ║   8 Compatibility Features           ║
                    ║   ┌─────────────────────────────┐   ║
                    ║   │ text_similarity     (30%)   │   ║
                    ║   │ skills_score        (15%)   │   ║
                    ║   │ profession_score    (15%)   │   ║
                    ║   │ mbti_score          (10%)   │   ║
                    ║   │ career_goal_score   (10%)   │   ║
                    ║   │ experience_score    (10%)   │   ║
                    ║   │ location_score       (5%)   │   ║
                    ║   │ networking_intent    (5%)   │   ║
                    ║   └─────────────────────────────┘   ║
                    ║          │                           ║
                    ║          ▼                           ║
                    ║   Logistic Regression Classifier     ║
                    ║   (trained on feedback data)         ║
                    ║          │                           ║
                    ║          ▼                           ║
                    ║   Acceptance Probability [0, 1]      ║
                    ║                                     ║
                    ╚══════════════════╤══════════════════╝
                                       │
                    ┌──────────────────┴──────────────────┐
                    │        HYBRID SCORE FUSION           │
                    │                                      │
                    │  final = 0.60 × hybrid_score         │
                    │        + 0.40 × ml_score             │
                    │                                      │
                    │        ▼                              │
                    │   Top 5 Recommendations               │
                    │   (with explainability scores)        │
                    └──────────────────────────────────────┘
```

### Compatibility Feature Breakdown

| # | Feature | Weight | Logic | Source Module |
|---|---------|--------|-------|---------------|
| 1 | `text_similarity` | 30% | TF-IDF cosine similarity between profile embeddings | `similarity_engine.py` |
| 2 | `mbti_score` | 10% | MBTI personality compatibility matrix lookup | `mbti_engine.py` |
| 3 | `profession_score` | 15% | Same or related profession group detection | `recommender.py` |
| 4 | `career_goal_score` | 10% | Alignment of stated career aspirations | `recommender.py` |
| 5 | `location_score` | 5% | Geographic proximity scoring | `recommender.py` |
| 6 | `experience_score` | 10% | Years-of-experience difference buckets | `recommender.py` |
| 7 | `skills_score` | 15% | Skill set intersection ratio (Jaccard-like) | `recommender.py` |
| 8 | `networking_intent_score` | 5% | Intent compatibility rules mapping | `recommender.py` |

---

## ⚙️ How It Works

```
  ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐    ┌───┐
  │ 1 │───►│ 2 │───►│ 3 │───►│ 4 │───►│ 5 │───►│ 6 │───►│ 7 │
  └───┘    └───┘    └───┘    └───┘    └───┘    └───┘    └───┘
```

| Step | Phase | Description |
|------|-------|-------------|
| **1** | 📥 Data Ingestion | Load 300 user profiles (15 fields) and 5,984 feedback records from CSV |
| **2** | 🧹 Text Preprocessing | Clean, tokenize, and normalize profile text (stopword removal, lowercasing) |
| **3** | 📐 Embedding Generation | TF-IDF encoder converts profile text into sparse vector representations |
| **4** | 🔬 Feature Engineering | Compute 8 compatibility features for each candidate user pair |
| **5** | 🤖 ML Prediction | Logistic Regression predicts acceptance probability P(accept \| features) |
| **6** | ⚖️ Hybrid Fusion | Blend rule-based hybrid score (60%) with ML score (40%) for final ranking |
| **7** | 🏆 Top-N Output | Return top 5 recommendations with scores and human-readable explanations |

---

## 📊 Model Training & Evaluation

### Training Pipeline

```bash
# Run the full training pipeline via CLI
python src/main.py
```

The pipeline executes the following sequence:

```
users.csv ──► TF-IDF Encoding ──► Feature Extraction ──► Dataset Build
                                                              │
feedback.csv ──────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                                    Train/Test Split (80/20)
                                                              │
                                                              ▼
                                                  Logistic Regression (balanced)
                                                              │
                                                              ▼
                                              Model Evaluation & Serialization
                                                              │
                                                    ┌─────────┴─────────┐
                                                    ▼                   ▼
                                            feedback_model.pkl   model_metadata.json
```

### Evaluation Metrics

| Metric | Score | Interpretation |
|---|---|---|
| **Accuracy** | 69.03% | Overall correct predictions |
| **Precision** | 44.60% | Of predicted accepts, % actually accepted |
| **Recall** | 57.06% | Of actual accepts, % correctly identified |
| **F1 Score** | 50.07% | Harmonic mean of precision and recall |
| **ROC AUC** | 66.98% | Model's discriminative ability |

### Feedback Loop Impact

| Metric | Before Feedback | After Feedback | Improvement |
|--------|----------------|----------------|-------------|
| Acceptance Rate | 40% | 67% | **+27 pp** ↑ |

> [!NOTE]
> The model uses **balanced class weights** to handle the imbalanced accept/reject distribution in feedback data. Detailed evaluation notebooks are available in `notebook/model_evaluation.ipynb`.

---

## 📁 Project Structure

```
Intelligent_Recommender_System/
├── data/                              # Datasets and data utilities
│   ├── users.csv                      # 300 user profiles with 15 fields
│   ├── feedback.csv                   # 5,984 feedback records (accept/reject)
│   └── src/                           # Data generation scripts
│       ├── dataset_generator.py       # Synthetic profile generation
│       ├── feedback_generator.py      # Feedback data simulation
│       └── dataset_validator.py       # Data integrity checks
│
├── models/                            # Trained model artifacts
│   ├── feedback_model.pkl             # Serialized Logistic Regression model
│   ├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer
│   └── model_metadata.json            # Training metadata and timestamps
│
├── notebook/                          # Jupyter notebooks
│   └── model_evaluation.ipynb         # Model evaluation and analysis
│
├── reports/                           # Generated reports
│   └── project_report.md             # Project analysis report
│
├── src/                               # Main source code
│   ├── app.py                         # Streamlit UI application (entry point)
│   ├── main.py                        # CLI training pipeline
│   │
│   ├── preprocessing/                 # Text preprocessing module
│   │   └── text_preprocessor.py       # Tokenization, cleaning, normalization
│   │
│   ├── embeddings/                    # Text embedding module
│   │   └── tfidf_encoder.py           # TF-IDF vectorization engine
│   │
│   ├── matching/                      # Feature engineering & matching
│   │   ├── recommender.py             # 8-feature compatibility calculator
│   │   ├── similarity_engine.py       # Cosine similarity computation
│   │   └── mbti_engine.py             # MBTI compatibility matrix
│   │
│   ├── learning/                      # ML ranking module
│   │   ├── feedback_model.py          # Logistic Regression classifier
│   │   ├── feedback_dataset.py        # Training dataset builder
│   │   └── adaptive_recommender.py    # ML-based ranking orchestrator
│   │
│   ├── utils/                         # Utility modules (12+ files)
│   │   ├── performance_analyzer.py    # Metrics computation & reporting
│   │   ├── explanation_engine.py      # Human-readable recommendation reasons
│   │   ├── model_manager.py           # Model serialization & loading
│   │   ├── config.py                  # Application configuration
│   │   ├── data_manager.py            # Data persistence layer
│   │   ├── validators.py              # Input validation
│   │   ├── audit_logger.py            # Action audit trail
│   │   ├── email_service.py           # Notification email service
│   │   ├── notifications.py           # In-app notifications
│   │   ├── sidebar.py                 # UI sidebar components
│   │   ├── styles.py                  # UI styling utilities
│   │   └── ...                        # Additional utilities
│   │
│   └── views/                         # Streamlit page views
│       ├── admin_portal.py            # Admin dashboard & controls
│       └── user_portal.py             # User-facing recommendation UI
│
├── quickstart.md                      # Quick start guide
├── DEPLOYMENT.md                      # Render deployment instructions
├── fullmetal.md                       # Comprehensive project documentation
├── render.yaml                        # Render platform config
├── requirements.txt                   # Python dependencies (57 packages)
├── runtime.txt                        # Python runtime version
├── .env.example                       # Environment variable template
└── README.md                          # This file
```

---

## 🛠️ Tech Stack

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Core runtime |
| **ML Framework** | scikit-learn | 1.9.0 | Logistic Regression, TF-IDF, metrics |
| **Data Processing** | pandas | 3.0.3 | DataFrame operations, CSV I/O |
| **Numerical** | NumPy | 2.4.6 | Array operations, linear algebra |
| **Scientific** | SciPy | 1.17.1 | Sparse matrices, distance computations |
| **NLP** | NLTK | 3.9.4 | Tokenization, stopword removal |
| **Web UI** | Streamlit | 1.58.0 | Interactive dashboard application |
| **Visualization** | Plotly | 6.8.0 | Interactive charts and metrics plots |
| **Data Generation** | Faker | 40.21.0 | Synthetic user profile generation |
| **Serialization** | joblib | 1.5.3 | Model persistence (pickle) |
| **Security** | bcrypt | 4.1.2 | Password hashing for admin portal |
| **Environment** | python-dotenv | 1.0.1 | Environment variable management |
| **Deployment** | Render | — | Cloud hosting platform |

---

## 💻 Usage Examples

### CLI — Train the Model

```bash
python src/main.py
```

**Output:**
```
Loading user profiles...
Building TF-IDF embeddings for 300 users...
Training Logistic Regression on 5,986 feedback pairs...
Model Accuracy: 69.03%
Top 5 recommendations for user U005:
  1. U042 — 89.3% compatibility
  2. U118 — 85.7% compatibility
  3. U073 — 82.1% compatibility
  4. U201 — 79.6% compatibility
  5. U156 — 76.4% compatibility
```

### Python API — Step-by-Step

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.embeddings.tfidf_encoder import TFIDFEncoder
from src.matching.recommender import Recommender
from src.learning.feedback_dataset import FeedbackDatasetBuilder
from src.learning.feedback_model import FeedbackModel
from src.learning.adaptive_recommender import AdaptiveRecommender

# ── Step 1: Encode user profiles into TF-IDF vectors ──
encoder = TFIDFEncoder()
users_df, tfidf_matrix = encoder.fit("data/users.csv")

# ── Step 2: Initialize recommender with feature engineering ──
recommender = Recommender(users_df, tfidf_matrix)

# ── Step 3: Build training dataset from historical feedback ──
builder = FeedbackDatasetBuilder(recommender, "data/feedback.csv")
dataset = builder.build()

# ── Step 4: Train the ML model ──
model = FeedbackModel()
model.train(dataset)

# ── Step 5: Create adaptive recommender (ML + rules) ──
adaptive = AdaptiveRecommender(recommender, model)

# ── Get top 5 recommendations for a user ──
recommendations = adaptive.get_top_recommendations(
    user_id="U005",
    top_n=5
)

for rec in recommendations:
    print(f"{rec['user_id']}: {rec['score']:.2f}% compatibility")

# ── Predict match score for a specific pair ──
prob = adaptive.predict_match_score("U005", "U015")
print(f"Acceptance Probability: {prob:.2f}%")
```

### Streamlit UI — Launch the Dashboard

```bash
streamlit run src/app.py
```

This opens a browser-based interface with:

- **User Portal** — Browse recommendations, view explanations, submit feedback
- **Admin Portal** — Manage users, retrain models, view audit logs, analyze performance

---

## ✅ Deliverables

Mapping of project components to evaluation criteria:

| # | Deliverable | Status | Implementation |
|---|-------------|--------|----------------|
| 1 | **Data Pipeline Script** | ✅ Complete | `data/src/dataset_generator.py` — generates 300 synthetic profiles with 15 fields; `data/src/feedback_generator.py` — simulates 5,984 feedback records |
| 2 | **Matching Algorithm** | ✅ Complete | Two-stage hybrid: TF-IDF cosine similarity (Stage 1) + Logistic Regression on 8 features (Stage 2); final hybrid fusion formula |
| 3 | **Performance Analysis** | ✅ Complete | `src/utils/performance_analyzer.py` — computes accuracy, precision, recall, F1, ROC AUC; `notebook/model_evaluation.ipynb` — detailed evaluation |
| 4 | **UI Demo** | ✅ Complete | `src/app.py` — full Streamlit application with user portal, admin portal, and real-time recommendations |

---

## 📈 Performance Results

### Classification Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Accuracy | **69.03%** | > 60% | ✅ Met |
| Precision | **44.60%** | > 35% | ✅ Met |
| Recall | **57.06%** | > 50% | ✅ Met |
| F1 Score | **50.07%** | > 40% | ✅ Met |
| ROC AUC | **66.98%** | > 60% | ✅ Met |

### Adaptive Learning Results

```
Acceptance Rate Over Training Iterations
─────────────────────────────────────────
Iteration 0 (baseline)  │████████░░░░░░░░░░░░░│  40%
Iteration 1             │██████████░░░░░░░░░░░│  48%
Iteration 2             │████████████░░░░░░░░░│  55%
Iteration 3             │██████████████░░░░░░░│  62%
Iteration 4 (final)     │██████████████░░░░░░░│  67%
                         0%                   100%
```

> **+27 percentage point improvement** from baseline to trained model, demonstrating the effectiveness of the feedback-driven adaptive learning loop.

---

## 💪 Strengths & Limitations

### ✅ Strengths

| # | Strength | Details |
|---|----------|---------|
| 1 | **Data-Driven Ranking** | ML model learns from actual user acceptance/rejection patterns |
| 2 | **Multi-Dimensional Matching** | 8 compatibility features capture diverse aspects of professional fit |
| 3 | **Explainability** | Every recommendation comes with human-readable reasoning |
| 4 | **Adaptive Learning** | Feedback loop continuously improves recommendation quality |
| 5 | **Balanced Training** | Class weights prevent bias toward the majority class |
| 6 | **Production-Ready UI** | Full Streamlit app with admin controls, audit logging, and user management |
| 7 | **Clean Architecture** | Modular separation: preprocessing → embeddings → matching → learning → views |

### ⚠️ Limitations

| # | Limitation | Impact | Mitigation Path |
|---|-----------|--------|-----------------|
| 1 | **Limited Training Data** | Model variance with small feedback sets | Collect more feedback; use cross-validation |
| 2 | **TF-IDF Semantics** | Ignores word order and semantic meaning | Migrate to BERT/sentence-transformers |
| 3 | **Cold Start Problem** | Poor recommendations for new users | Use content-based fallback for new profiles |
| 4 | **O(n²) Scaling** | Pairwise scoring bottleneck at 10K+ users | Implement approximate nearest neighbors (ANN) |
| 5 | **Static Features** | No temporal dynamics or recency weighting | Add time-decay factors to feedback |
| 6 | **Binary Feedback** | Only accept/reject — no nuance | Add rating scales and implicit signals |
| 7 | **MBTI Validity** | Personality typing has scientific limitations | Supplement with Big Five traits model |

---

## 🔮 Future Improvements

### Short-Term (Easy)

- [ ] Hyperparameter tuning via grid search (regularization, max iterations)
- [ ] K-fold cross-validation for robust accuracy estimates
- [ ] Feature importance visualization dashboard
- [ ] TF-IDF model caching for faster repeated predictions

### Medium-Term (Moderate)

- [ ] Approximate Nearest Neighbors (ANN) for O(1) candidate retrieval
- [ ] Temporal decay weights for older feedback data
- [ ] A/B testing framework for comparing ranking strategies
- [ ] REST API layer (FastAPI) for microservice deployment

### Long-Term (Complex)

- [ ] Neural embeddings (BERT, Sentence-Transformers) for semantic understanding
- [ ] Collaborative filtering via matrix factorization
- [ ] Implicit feedback signals (clicks, views, message patterns)
- [ ] Learning-to-rank models (LambdaMART, RankNet)
- [ ] Real-time model updating with online learning

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** with clear messages: `git commit -m "Add: feature description"`
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request with a description of your changes

> [!IMPORTANT]
> Please ensure all existing tests pass and add tests for new functionality before submitting a PR.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 nextmatchAi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 📬 Contact

| | |
|---|---|
| **Author** | *Your Name* |
| **Email** | *your.email@example.com* |
| **GitHub** | *github.com/your-username* |
| **Project** | [nextmatchAi — Intelligent Recommender System](https://github.com/your-username/Intelligent_Recommender_System) |

---

<div align="center">

**Built with 🧠 Machine Learning · 📊 Data Science · ❤️ Passion**

*Last Updated: June 2026*

</div>
