# Full Project Architecture and Deep Execution Flow

## 1. Project Identity

This project is an ML-based professional networking recommender system. It is not a pure collaborative-filtering recommender and not a deep learning recommender. Its current design is a **two-stage hybrid recommendation pipeline**:

1. **Stage 1: Hybrid candidate generation**
2. **Stage 2: ML-based re-ranking**

At a high level, the system:

- Reads user profile data from `data/users.csv`
- Builds a text representation called `profile_text`
- Cleans the text with NLTK-based preprocessing
- Converts text into TF-IDF vectors
- Computes multi-factor compatibility features between two users
- Builds a labeled training dataset from `data/feedback.csv`
- Trains a Logistic Regression classifier on those compatibility features
- Uses the classifier to predict acceptance probability
- Combines the hybrid score and ML score into a final ranking score

Core implementation files:

- `src/main.py`
- `src/embeddings/tfidf_encoder.py`
- `src/preprocessing/text_preprocessor.py`
- `src/matching/recommender.py`
- `src/learning/feedback_dataset.py`
- `src/learning/feedback_model.py`
- `src/learning/adaptive_recommender.py`
- `src/utils/model_manager.py`

---

## 2. Actual Architecture in This Repository

### 2.1 Data Layer

The project uses two main CSV files:

- `data/users.csv`
  - 300 user profiles
  - 15 columns
  - Includes structured and unstructured profile data
- `data/feedback.csv`
  - 5984 interaction records
  - Binary label field `action`
  - `0 = reject`, `1 = accept`

Observed dataset schema:

### User profile columns

- `user_id`
- `name`
- `age`
- `location`
- `profession`
- `experience_years`
- `education`
- `skills`
- `mbti`
- `traits`
- `career_goal`
- `networking_intent`
- `interests`
- `professional_summary`
- `about_me`

### Feedback columns

- `user_id`
- `matched_user_id`
- `action`
- `timestamp`

### 2.2 Data Generation Layer

Synthetic data is generated through:

- `data/src/dataset_generator.py`
- `data/src/feedback_generator.py`
- `data/src/dataset_validator.py`

This means the current project does **not** collect production user events from a live application. Instead, it uses:

- synthetic user generation
- synthetic feedback generation
- lightweight validation rules

### 2.3 Preprocessing Layer

Implemented in `src/preprocessing/text_preprocessor.py`.

The cleaning pipeline does:

- lowercase conversion
- removal of non-alphabetic characters
- tokenization by whitespace
- English stopword removal using NLTK
- lemmatization using WordNet

This is applied to a combined text field built from multiple user columns.

### 2.4 Text Representation Layer

Implemented in `src/embeddings/tfidf_encoder.py`.

The project constructs `profile_text` by concatenating:

- `professional_summary`
- `about_me`
- `career_goal`
- `interests`
- `profession`
- `skills`
- `education`
- `traits`
- `networking_intent`

Then it fits:

- `TfidfVectorizer(max_features=5000)`

Outputs:

- enriched `users_df`
- sparse TF-IDF matrix

Persisted artifact:

- `models/tfidf_vectorizer.pkl`

### 2.5 Feature Engineering Layer

Implemented mainly in `src/matching/recommender.py`.

For every user pair, the system computes 8 compatibility features:

1. `text_similarity`
2. `mbti_score`
3. `profession_score`
4. `career_goal_score`
5. `location_score`
6. `experience_score`
7. `skills_score`
8. `networking_intent_score`

The hybrid score is a weighted sum:

```text
final_score =
0.30 * text_similarity +
0.10 * mbti_score +
0.15 * profession_score +
0.10 * career_goal_score +
0.05 * location_score +
0.10 * experience_score +
0.15 * skills_score +
0.05 * networking_intent_score
```

### 2.6 Supervised Learning Layer

Implemented in:

- `src/learning/feedback_dataset.py`
- `src/learning/feedback_model.py`

The feedback dataset builder reads each row of `feedback.csv`, recomputes compatibility features for that user pair, and creates a training row.

Training features:

- the 8 compatibility features above

Target label:

- `label = action`

Model used:

- `LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)`

Persisted artifact:

- `models/feedback_model.pkl`

### 2.7 Ranking Layer

Implemented in `src/learning/adaptive_recommender.py`.

This is the real recommendation serving logic:

1. Use the hybrid recommender to get top 30 candidates
2. For each candidate, predict acceptance probability using the trained ML model
3. Fuse hybrid score and ML score
4. Sort by final ranking score
5. Return top N recommendations

Final ranking formula:

```text
final_ranking_score = 0.60 * hybrid_score + 0.40 * ml_score
```

---

## 3. Deep Execution Flow

## 3.1 Offline / Training Flow

When `python src/main.py` runs, the actual code path is:

### Step 1: Resolve file paths

`src/main.py` builds absolute paths to:

- `data/users.csv`
- `data/feedback.csv`

### Step 2: Build TF-IDF representation

`TFIDFEncoder.fit(users_path)` does:

1. Load `users.csv` into a DataFrame
2. Build `profile_text`
3. Clean each text row with `clean_text()`
4. Fit `TfidfVectorizer`
5. Transform profile text into sparse TF-IDF vectors
6. Save vectorizer to `models/tfidf_vectorizer.pkl`

### Step 3: Create similarity and feature engines

`src/main.py` initializes:

- `SimilarityEngine(users_df, matrix)`
- `Recommender(users_df, matrix)`

`SimilarityEngine` is used for raw cosine similarity inspection.

`Recommender` is used for full compatibility feature calculation.

### Step 4: Build supervised training dataset

`FeedbackDatasetBuilder(recommender, feedback_path).build()` does:

1. Load `feedback.csv`
2. For each `(user_id, matched_user_id)` pair:
   - call `recommender.compatibility_score()`
   - extract the 8 features
   - attach `action` as label
3. Return a DataFrame of shape `(5984, 9)`

Observed training dataset:

- 8 feature columns
- 1 label column
- total rows: 5984

### Step 5: Split train and test sets

Inside `FeedbackModel.train()`:

- `train_test_split(test_size=0.2, random_state=42)`

This means:

- 80% training
- 20% testing

There is no stratification parameter explicitly provided.

### Step 6: Train the classifier

The model is fit on the training split using Logistic Regression.

Because `class_weight="balanced"` is enabled, the algorithm compensates for class imbalance automatically.

Observed feedback label distribution:

- reject (`0`): 72.34%
- accept (`1`): 27.66%

### Step 7: Evaluate the classifier

The model then predicts on the held-out test set and prints:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

Observed runtime metrics from this repository state:

- Accuracy: `0.6759`
- Precision: `0.4215`
- Recall: `0.5607`
- F1 Score: `0.4813`
- ROC AUC: `0.6584`

Observed confusion matrix:

- TP: `180`
- FN: `141`
- FP: `247`
- TN: `629`

### Step 8: Persist trained model

The trained classifier is saved to:

- `models/feedback_model.pkl`

### Step 9: Build adaptive recommender

`AdaptiveRecommender(recommender, model)` is created after training.

This object is the orchestration layer for final recommendation serving.

### Step 10: Generate recommendations

For a sample user like `U005`, the flow is:

1. Generate candidate pool from hybrid recommender
2. Keep top 30 by hybrid score
3. For each candidate, recompute features
4. Predict match acceptance probability
5. Compute final ranking score
6. Sort descending
7. Return top 5

---

## 3.2 Online / Inference Flow

The inference path for one recommendation request is:

### Input

- active user ID, for example `U005`

### Stage 1: Candidate generation

`AdaptiveRecommender.get_candidate_pool(user_id, pool_size=30)` calls:

- `Recommender.get_top_recommendations()`

This scans every other user in the dataset and computes a hybrid score using hand-engineered rules plus TF-IDF similarity.

This is effectively an **all-pairs scoring pass** against the full user table.

### Stage 2: ML prediction

For each candidate:

1. Recompute 8 compatibility features
2. Wrap them into a one-row DataFrame
3. Call `feedback_model.predict_probability()`
4. Convert probability to percentage

### Stage 3: Final fusion

The system combines:

- `hybrid_score`
- `ml_score`

using:

```text
0.60 * hybrid_score + 0.40 * ml_score
```

### Output

Each returned recommendation contains:

- `user_id`
- `profession`
- `career_goal`
- `mbti`
- `location`
- `experience`
- `hybrid_score`
- `ml_score`
- `final_ranking_score`

---

## 4. How the Algorithm Is Used in This Project

## 4.1 TF-IDF Algorithm Usage

TF-IDF is used to convert free-form profile text into numeric vectors.

Why it is used here:

- user summaries and profile text are unstructured
- similarity cannot be computed directly on raw text
- TF-IDF highlights important words for each profile while down-weighting very common words

How it is used:

1. Build one combined profile string
2. Clean the text
3. Fit TF-IDF vectorizer across all users
4. Transform each profile into a sparse vector
5. Use cosine similarity between vectors

What it powers:

- `text_similarity`
- the semantic part of the hybrid recommender

## 4.2 Logistic Regression Usage

Logistic Regression is the supervised ML algorithm used for acceptance prediction.

Why it is used here:

- the problem is binary classification
- output probability is easy to interpret
- coefficients are explainable
- works well with low-dimensional tabular features

How it is used:

1. Input = 8 engineered compatibility features
2. Label = user accepted match or rejected match
3. Train on historical feedback
4. Predict probability that a future match will be accepted

What it powers:

- `ml_score`
- stage-2 reranking

## 4.3 Handcrafted Rule Engine Usage

Not all intelligence comes from ML here. A large part comes from deterministic rules in `recommender.py`.

Examples:

- same location gives `100`, different location gives `0`
- close experience years gives high compatibility
- same or related profession boosts score
- same or grouped career goals boost score
- compatible networking intents boost score

This rule engine acts as:

- feature generator for the ML model
- fallback compatibility layer
- candidate retriever before reranking

---

## 5. Requested ML Lifecycle Mapping

## 5.1 Defining the Objective

### General ML meaning

This stage defines:

- what the model should optimize
- what prediction task is being solved
- what success means

### In this project

The objective is:

- recommend professional connections that are likely to be accepted by the target user

More specifically, the system optimizes for:

- semantic similarity of profiles
- compatibility across profession, MBTI, career goals, skills, experience, location, and networking intent
- probability of acceptance learned from historical feedback

So the actual learning objective is:

- **binary classification**
- predict whether a user will accept a recommended match

Target variable:

- `feedback.action`

Business-level objective:

- surface better networking matches
- improve acceptance rate
- make recommendations explainable

## 5.2 Data Collection & Preprocessing

### General ML meaning

This stage gathers raw data and converts it into a usable form.

### In this project

Data collection is synthetic, not live.

Collected/generated sources:

- `dataset_generator.py` creates user profiles
- `feedback_generator.py` creates feedback interactions

The project currently simulates:

- profile metadata
- interests
- skills
- personality type
- professional summaries
- feedback labels

Preprocessing implementation:

1. concatenate multiple profile fields into `profile_text`
2. lowercase text
3. remove punctuation and non-letter characters
4. remove English stopwords
5. lemmatize words
6. fit TF-IDF on cleaned text

Important note:

- preprocessing is real and implemented
- data collection is not production collection; it is synthetic generation

## 5.3 Feature Engineering

### General ML meaning

Feature engineering transforms raw data into predictive signals.

### In this project

This is one of the strongest implemented parts.

The project engineers 8 pairwise features:

1. `text_similarity`
   - cosine similarity of TF-IDF vectors
2. `mbti_score`
   - rule-based personality compatibility
3. `profession_score`
   - exact or grouped profession similarity
4. `career_goal_score`
   - exact or grouped goal similarity
5. `location_score`
   - same city or not
6. `experience_score`
   - based on years difference buckets
7. `skills_score`
   - overlap of comma-separated skills
8. `networking_intent_score`
   - intent compatibility rule

These features are used twice:

- once for hybrid ranking
- once as ML training input

## 5.4 Train-Test Splitting

### General ML meaning

This stage separates data used for learning from data used for evaluation.

### In this project

Implemented in `FeedbackModel.train()` with:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

What this means:

- 80% of samples are used for training
- 20% are used for testing
- same split is reproducible because of `random_state=42`

Current limitations:

- no explicit `stratify=y`
- no cross-validation
- no validation set separate from test set
- no temporal split using timestamps

## 5.5 Model Training & Tuning

### General ML meaning

This stage selects an algorithm, fits it, evaluates it, and tunes it.

### In this project

Implemented model:

- Logistic Regression

Configuration:

- `class_weight="balanced"`
- `random_state=42`
- `max_iter=1000`

Training flow:

1. build dataset from feedback logs
2. split into train and test
3. fit logistic regression
4. evaluate using classification metrics
5. save trained model artifact

Observed feature influence from the latest run:

- `profession_score`: `0.0157`
- `career_goal_score`: `0.0110`
- `skills_score`: `0.0043`
- `text_similarity`: `0.0041`
- `mbti_score`: `0.0037`
- `location_score`: `0.0016`
- `experience_score`: `0.0009`
- `networking_intent_score`: `0.0002`

Interpretation:

- profession and career-goal alignment are the strongest positive learned signals in this current synthetic dataset
- networking intent currently contributes very little to the trained classifier

Tuning status:

- model training is implemented
- hyperparameter tuning is **not** implemented
- grid search, random search, Bayesian optimization, or threshold tuning are absent

## 5.6 Production Deployment

### General ML meaning

This stage serves the model in an application where users can consume predictions.

### In this project

There is a partial local deployment story, but not a full production deployment.

Implemented pieces:

- `src/app.py` Streamlit UI shell
- multiple Streamlit pages under `src/pages/`
- model persistence in `models/`
- vectorizer persistence in `models/`

Current deployment reality:

- this project is mainly a **local demo / prototype**
- recommendations can be served inside a Streamlit-style interface
- artifacts are stored locally using `joblib`

Not fully implemented:

- no complete REST inference API implementation
- no containerization workflow
- no CI/CD deployment pipeline
- no cloud serving configuration
- no real-time feature store
- no online feedback ingestion pipeline

Important codebase gap found during evaluation:

- `src/app.py` imports `load_system` from `src/utils/loader.py`
- `src/utils/loader.py` in the current repository state does not define `load_system`
- it also imports `src.utils.nltk_helper`, which is not present in the scanned files

So the intended frontend architecture exists, but the current app path appears incomplete or broken in this snapshot.

## 5.7 Continuous Monitoring

### General ML meaning

Monitoring checks whether the deployed system remains healthy and useful over time.

### In this project

There is **evaluation visibility**, but not true production monitoring.

Implemented monitoring-like elements:

- accuracy display
- precision display
- recall display
- F1 score
- ROC AUC
- confusion matrix
- dataset distribution views
- feature coefficient visualization

These appear in:

- `src/pages/4_Model_Analytics.py`
- `src/pages/5_Dataset_Insights.py`

What is missing for real continuous monitoring:

- automated scheduled retraining
- drift detection
- live acceptance-rate tracking
- feature distribution drift alerts
- model version registry
- latency/error monitoring
- online A/B testing
- user behavior logging pipeline

So this project has **manual analytics**, not full MLOps monitoring.

---

## 6. End-to-End Execution Summary

The complete end-to-end flow in one sequence is:

1. Generate or load user profiles
2. Generate or load feedback labels
3. Combine profile text fields into a single textual representation
4. Clean text with NLTK preprocessing
5. Fit TF-IDF vectorizer and encode user profiles
6. Compute pairwise compatibility features
7. Build supervised feedback dataset
8. Split data into train/test
9. Train Logistic Regression model
10. Save vectorizer and classifier
11. Retrieve top hybrid candidates for a user
12. Predict acceptance probability for each candidate
13. Fuse hybrid score and ML score
14. Return top ranked recommendations
15. Show analytics and explainability views in the UI layer

---

## 7. Strengths of the Current Design

- clean separation between preprocessing, feature engineering, learning, and ranking
- explainable features and explainable classifier
- persisted vectorizer and model artifacts
- good educational example of a hybrid recommender
- uses both content-based similarity and supervised learning
- includes analytics and explainability pages

---

## 8. Current Limitations and Gaps

- user and feedback data are synthetic, not real production data
- candidate generation scans all users and does not scale well
- no hyperparameter tuning pipeline
- no cross-validation
- no stratified split
- no temporal evaluation using feedback timestamps
- no collaborative filtering component
- no deep learning embeddings
- frontend loader path appears incomplete in current repo state

---

## 9. Final Evaluation

This project is now a **hardened, deployment-ready hybrid ML recommender prototype** equipped with robust security, trust indicators, error handling, administrative auditing, and deployment files.

What is fully implemented and production-ready:

- **Objective definition & preprocessing**: Structured profiling with NLTK token cleaning.
- **Supervised classification & feature engineering**: Logistic Regression with compatibility metrics.
- **Two-stage recommendation flow**: TF-IDF retrieval candidate generation + ML classifier re-ranking.
- **Explainability & Diversity (Phase 7)**: Clear textual reasons, strengths/weaknesses, profession diversity filtering, and recommendation freshness decay using recent match logs.
- **Production Hardening (Phase 8)**: Bcrypt password hashing, session expiry (60m inactivity), audit log writing (LOGIN, PROFILE_UPDATE, RETRAIN, etc.), system monitoring dashboards (KPI metrics, components check, SMTP tests), user preferences, and Render.com configuration.

---

## 10. Phase 7 — Recommendation Quality, Explainability & User Trust Enhancement

Phase 7 addressed the black-box nature of the machine learning recommendations by introducing trust elements, explainable metrics, diversity controls, and candidate freshness:

### 10.1 Explanation Engine
- Implemented in `src/utils/explanation_engine.py`.
- Generates a textual summary, clear reasons for the recommendation (e.g. TF-IDF threshold match, MBTI compatibility), shared skills list, and specific strengths/weaknesses.

### 10.2 Recommendation Freshness
- Uses `data/recommendation_history.csv` to track what profiles a user has seen.
- Applies a decay penalty to recently recommended candidates to prevent recommendation fatigue and presentation collapse.

### 10.3 Diversity Filtering
- Restricts recommendable professionals to a maximum of `MAX_SAME_PROFESSION` per batch.
- Configurable through toggles (`ENABLE_DIVERSITY_FILTER`).
- Audits diversity via the **Profession Diversity Index** and distribution bar chart inside the admin recommendations quality dashboard.

---

## 11. Phase 8 — Production Hardening, Authentication & Email Workflows

Phase 8 transformed the application into a SaaS-ready platform by implementing enterprise-grade security and administrative tools:

### 11.1 Security & Authentication Hardening
- **Bcrypt Hashing**: Credentials stored in `data/credentials.csv` using bcrypt instead of legacy SHA-256.
- **Account Lockouts**: Temporarily locks account for 15 minutes after 5 consecutive failed login attempts.
- **Session Expiry**: Automates logout by tracking last active timestamp and checking age (60 minutes inactivity threshold).

### 11.2 Centralized Logging & Audit Trail
- Log records appended to `data/audit_log.csv`.
- Captures login actions, profile changes, model retraining events, and errors.
- Includes a dedicated Admin Audit Log Viewer with pagination and filter support.

### 11.3 System Monitoring & Data Management
- Admin dashboard displaying platform KPIs (acceptance rates, active users, pending feedback count).
- Color-coded System Health status indicator (Green/Amber/Red) running automated SMTP connection checks and model existence checks.
- Granular User Management view (enable/disable account, force password updates, profile completeness ratios).
- Safe, read-only Data Management portal for exporting CSV backups.

### 11.4 Notification Preferences
- Allows users to opt in/out of Welcome Emails, Recommendation Digests, and System Notifications.
- Saves preferences directly in `data/user_profiles.csv` and honors them across email send routines.

### 11.5 Deployment readiness
- Render.com blueprints (`render.yaml`), python runtime specs (`runtime.txt`), streamlit settings (`.streamlit/config.toml`), and a detailed setup guide (`DEPLOYMENT.md`).

