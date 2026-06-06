# CrypticAlign : Intelligent Recommender System

An intelligent recommendation engine that uses TF-IDF embeddings and similarity matching to provide personalized recommendations. The system processes user data, generates embeddings for text content, and matches users with relevant items based on MBTI profiling and similarity scores.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Components](#components)
- [Development](#development)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Intelligent_Recommender_System
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .ipenv
   .ipenv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .ipenv
   source .ipenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate enhanced dataset** (optional - skip if users.csv exists)
   ```bash
   python data/src/dataset_generator.py
   ```

5. **Run the main application**
   ```bash
   python src/main.py
   ```

## Project Architecture

### ML-Based Ranking Pipeline

```
┌─────────────────────────────────────────────┐
│         Data Layer (data/)                   │
│  • users.csv, feedback.csv                   │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│    Preprocessing Layer (src/preprocessing/)  │
│  • TextPreprocessor: Clean & normalize text  │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│      Embeddings Layer (src/embeddings/)      │
│  • TFIDFEncoder: Convert text to vectors     │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│      Feature Engineering (src/matching/)     │
│  • text_similarity                           │
│  • mbti_score                                │
│  • profession_score                          │
│  • career_goal_score                         │
│  • location_score                            │
│  • experience_score                          │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│       ML Model (src/learning/)               │
│  • Logistic Regression with balanced weights │
│  • Trained on feedback data                  │
│  • Output: Acceptance Probability (0-1)      │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│    Adaptive Ranking (src/learning/)          │
│  • ML Probability × 100 = PRIMARY SIGNAL     │
│  • Rank candidates by probability            │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│    Top Recommendations (top_n sorted)        │
│  • User ID, Profile, Score (ML Probability) │
└─────────────────────────────────────────────┘
```

### How It Works

1. **Data Ingestion**: User profiles and feedback data loaded from CSV
2. **Text Preprocessing**: Profile text cleaned and normalized
3. **Embedding Generation**: TF-IDF encoder creates dense vectors
4. **Feature Engineering**: Six key features extracted per user pair
5. **ML Model Prediction**: Logistic Regression predicts acceptance probability
6. **Ranking**: Users ranked by ML-predicted probability (primary signal)
7. **Top Recommendations**: Return top-N candidates sorted by ML score

## Installation

### Step-by-Step for Beginners

#### 1. Install Python

Download from [python.org](https://www.python.org/downloads/) (version 3.8+)

#### 2. Open Terminal/Command Prompt

Navigate to your project folder:
```bash
cd path\to\Intelligent_Recommender_System
```

#### 3. Create Virtual Environment

A virtual environment keeps project dependencies isolated:

**Windows:**
```bash
python -m venv .ipenv
.ipenv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .ipenv
source .ipenv/bin/activate
```

You should see `(.ipenv)` at the start of your terminal line when activated.

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Verify Installation

```bash
python -c "import pandas; print('Installation successful!')"
```

## Usage

### Running the ML-Based Recommendation System

```bash
python src/main.py
```

Output:
- Model accuracy on test set
- Feature coefficients (shows importance of each dimension)
- Top 5 recommendations for a sample user with ML probability scores

### Using Components Individually

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.embeddings.tfidf_encoder import TFIDFEncoder
from src.matching.recommender import Recommender
from src.learning.feedback_dataset import FeedbackDatasetBuilder
from src.learning.feedback_model import FeedbackModel
from src.learning.adaptive_recommender import AdaptiveRecommender

# Step 1: Encode user profiles
encoder = TFIDFEncoder()
users_df, matrix = encoder.fit("data/users.csv")

# Step 2: Create recommender with feature engineering
recommender = Recommender(users_df, matrix)

# Step 3: Build training dataset from feedback
builder = FeedbackDatasetBuilder(recommender, "data/feedback.csv")
dataset = builder.build()

# Step 4: Train ML model
model = FeedbackModel()
model.train(dataset)

# Step 5: Use adaptive recommender with ML ranking
adaptive = AdaptiveRecommender(recommender, model)

# Get recommendations
recommendations = adaptive.get_top_recommendations(
    user_id="U005",
    top_n=5
)

# Get acceptance probability for specific pair
prob = adaptive.predict_match_score("U005", "U015")
print(f"Acceptance Probability: {prob:.2f}%")

# Print recommendations
for rec in recommendations:
    print(f"{rec['user_id']}: {rec['score']:.2f}% compatibility")
```

### Model Training & Evaluation

The system automatically:
1. Loads user profiles and builds TF-IDF embeddings
2. Calculates 6 compatibility features per user pair
3. Trains Logistic Regression on feedback data (80/20 split)
4. Reports accuracy and displays feature importance
5. Makes predictions using ML probabilities

## Project Structure

```
Intelligent_Recommender_System/
├── api/                          # API endpoints (Flask/FastAPI)
├── app/                          # Application UI/logic
├── data/                         # Datasets and data utilities
│   ├── users.csv                 # User profiles (500+ users)
│   ├── feedback.csv              # Acceptance/rejection labels
│   ├── dataset_statistics.csv    # Data statistics
│   └── src/                      # Data generation scripts
│       ├── dataset_generator.py  # Generate synthetic datasets
│       ├── dataset_validator.py  # Validate data integrity
│       └── feedback_generator.py # Generate feedback data
├── models/                       # Trained models storage
├── notebooks/                    # Jupyter notebooks for exploration
├── reports/                      # Generated reports & outputs
├── src/                          # Main source code
│   ├── main.py                   # Application entry point
│   ├── embeddings/               # Text embedding module
│   │   └── tfidf_encoder.py      # TF-IDF vectorization
│   ├── matching/                 # Feature engineering module
│   │   ├── similarity_engine.py  # Compute text similarity
│   │   ├── mbti_engine.py        # MBTI personality matching
│   │   └── recommender.py        # Calculate 6 compatibility features
│   ├── preprocessing/            # Data preprocessing module
│   │   └── text_preprocessor.py  # Text cleaning & normalization
│   └── learning/                 # ML ranking module
│       ├── feedback_model.py      # Logistic Regression classifier
│       ├── feedback_dataset.py    # Dataset builder from feedback
│       └── adaptive_recommender.py # ML-based ranking orchestrator
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── DATASET_ENHANCEMENTS.md       # Dataset enhancement documentation
```

## 🆕 Enhanced Dataset (v2.0)

The synthetic dataset has been upgraded with **4 new fields** for richer professional profiles while maintaining full backward compatibility:

| Field | Type | Examples | Purpose |
|-------|------|----------|---------|
| **education** | String | B.Tech CS, MBA, MBBS, M.Tech AI | Realistic education based on profession |
| **skills** | CSV | Python,ML,SQL, React,AWS | Profession-specific technical skills (3 per user) |
| **traits** | CSV | Leadership,Analytical | Personality traits (2-3 per user) |
| **networking_intent** | String | Research Collaboration, Startup Partner | Professional networking objective |

**Enhanced TF-IDF Profile**:
The profile text now includes all new fields, resulting in richer semantic embeddings for better similarity matching.

See [DATASET_ENHANCEMENTS.md](DATASET_ENHANCEMENTS.md) for comprehensive details, examples, and future enhancement opportunities.

## Components

### 1. Text Preprocessor (`src/preprocessing/text_preprocessor.py`)
- Cleans and normalizes user profile text data
- Removes stopwords, handles tokenization
- Prepares text for embedding generation

### 2. TF-IDF Encoder (`src/embeddings/tfidf_encoder.py`)
- Converts text into TF-IDF (Term Frequency-Inverse Document Frequency) vectors
- Creates numerical representations for similarity calculations
- Handles dimensionality reduction

### 3. Similarity Engine (`src/matching/similarity_engine.py`)
- Calculates cosine similarity between user profile embeddings
- Returns similarity scores on a 0-100% scale
- Supports multiple similarity metrics

### 4. Recommender (`src/matching/recommender.py`)
- Orchestrates feature engineering across multiple dimensions
- Calculates six compatibility scores:
  - **text_similarity**: Profile text overlap (TF-IDF cosine similarity)
  - **mbti_score**: Personality type compatibility
  - **profession_score**: Similar professional backgrounds
  - **career_goal_score**: Aligned career aspirations
  - **location_score**: Geographic proximity compatibility
  - **experience_score**: Similar years of experience

### 5. Feedback Model (`src/learning/feedback_model.py`)
- Trains Logistic Regression on historical feedback data
- Uses balanced class weights to handle imbalanced datasets
- Predicts acceptance probability: P(user accepts match) ∈ [0, 1]
- Features: [text_similarity, mbti_score, profession_score, career_goal_score, location_score, experience_score]

### 6. Adaptive Recommender (`src/learning/adaptive_recommender.py`)
- Integrates ML model predictions into ranking
- **predict_match_score()**: Returns ML-predicted acceptance probability × 100
- **get_top_recommendations()**: Ranks users by ML probability (primary signal)
- Returns top-N candidates with full profile information and ML score

## Features

### ✅ Strengths

1. **Data-Driven Ranking**: ML model learns from actual acceptance patterns in feedback data
2. **Multi-Dimensional Compatibility**: Considers 6 different compatibility dimensions beyond text similarity
3. **Balanced Learning**: Class weights prevent bias toward majority class (improves minority detection)
4. **Explainable**: Feature coefficients show which factors most influence acceptance
5. **Probabilistic Output**: Returns interpretable probability scores (0-100%)
6. **Personality Matching**: Integrates MBTI personality types for deeper compatibility
7. **Scalable Architecture**: Clean separation of concerns (preprocessing, embeddings, ML, ranking)

### ⚠️ Weaknesses & Limitations

1. **Limited Training Data**
   - Quality of predictions depends on feedback dataset size
   - Small dataset = high variance in model coefficients
   - **Impact**: May overfit to limited user patterns

2. **Binary Classification Bias**
   - Model only learns "accepted" vs "rejected" patterns
   - Cannot capture neutral or uncertain matches
   - **Impact**: Extreme probability outputs (near 0 or 100%)

3. **Static Feature Engineering**
   - Features are pre-computed and don't adapt to user context
   - No temporal dynamics (recency bias not modeled)
   - **Impact**: Older user data weighted equally with recent data

4. **TF-IDF Limitations**
   - Ignores word order and semantic relationships
   - Poor performance on short profiles or slang
   - High-dimensional sparse vectors (memory intensive)
   - **Impact**: May miss nuanced compatibility signals

5. **Personality Type Oversimplification**
   - MBTI reduces complex personalities to 4-letter types
   - Scientific validity of MBTI questioned by researchers
   - **Impact**: False compatibility matches between different types

6. **Location & Experience Binary Scoring**
   - No gradient: either "compatible" or "not"
   - Doesn't account for willingness to relocate
   - No learning curve for career growth
   - **Impact**: Rigid matching criteria

7. **Cold Start Problem**
   - New users with no feedback history → unpredictable model behavior
   - New user profiles untrained in TF-IDF space
   - **Impact**: Poor recommendations for onboarded users

8. **Feedback Data Quality**
   - Feedback may be noisy or contain strategic responses
   - Selection bias: users only give feedback on viewed matches
   - **Impact**: Model learns from biased subset of potential matches

9. **Scalability Concerns**
   - O(n²) complexity: must score every user pair
   - Inefficient for systems with 10K+ users
   - **Recommendation**: Use approximate nearest neighbors (ANN) for production

10. **No Personalization Over Time**
    - Model coefficients frozen after training
    - Cannot adapt to changing user preferences
    - **Impact**: Recommendations become stale over weeks/months

11. **Feature Correlation Issues**
    - Some features may be correlated (text_similarity & career_goal_score)
    - Logistic Regression assumes feature independence
    - **Impact**: Model may assign inflated weights to correlated features

12. **Missing Context Features**
    - No user activity history
    - No interaction patterns (clicks, messages, views)
    - No demographic diversity factors
    - **Impact**: Homogeneous recommendations

## Recommended Improvements

### Short-term (Easy)
- Add model hyperparameter tuning (grid search over max_iter, regularization)
- Implement k-fold cross-validation for robust accuracy estimates
- Add feature importance visualization
- Cache TF-IDF models to speed up repeated predictions

### Medium-term (Moderate)
- Implement approximate nearest neighbors (ANN) for O(1) candidate retrieval
- Add temporal dynamics (decay weights for old feedback)
- Implement user feedback loop to track recommendation acceptance
- Add A/B testing framework to compare ranking strategies

### Long-term (Complex)
- Migrate to neural network embeddings (BERT, Word2Vec)
- Implement collaborative filtering (matrix factorization)
- Add interaction tracking and implicit feedback
- Build ranking-specific loss functions (LambdaMART, RankNet)

## Development

### ML Model Training Pipeline

**The main.py flow:**
```
1. Load users → TF-IDF encode profiles
2. Calculate compatibility scores (recommender)
3. Build training dataset from feedback.csv
4. Train Logistic Regression on [features] → labels
5. Create adaptive recommender (ml model + feature engine)
6. Generate top recommendations (sorted by ML probability)
```

### Retraining the Model

To update the ML model after collecting new feedback:

```bash
# 1. Ensure feedback.csv has new user pair acceptance/rejection data
# 2. Run main.py (automatically retrains on feedback.csv)
python src/main.py

# 3. Check new model accuracy and feature coefficients
```

### Feature Importance Analysis

After training, examine model coefficients:
```
python -c "
from src.learning.feedback_model import FeedbackModel
import pandas as pd

model = FeedbackModel()
# Fit model...
features = ['text_similarity', 'mbti_score', 'profession_score', 
            'career_goal_score', 'location_score', 'experience_score']
for feat, coef in zip(features, model.model.coef_[0]):
    print(f'{feat}: {coef:.4f}')
"
```

**Positive coefficient** = increases acceptance probability
**Negative coefficient** = decreases acceptance probability
**Magnitude** = strength of influence

### Adding New Features

To add a new compatibility dimension:

1. **Add calculation in `recommender.py`** → new_score()
2. **Include in compatibility_score()** return dict
3. **Update FeedbackDatasetBuilder** to include new column
4. **Update FeedbackModel.predict_probability()** column names
5. **Retrain model** with new feedback dataset

### Data Utilities

```bash
# Generate synthetic user profiles & feedback
python data/src/dataset_generator.py

# Check data statistics
python data/src/dataset_statistics.py

# Validate data integrity
python data/src/dataset_validator.py
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

## Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **Faker** - Generate fake data for testing
- **python-dateutil** - Date/time utilities
- **tzdata** - Timezone data

See `requirements.txt` for versions.

## Troubleshooting

### "ModuleNotFoundError" when running main.py

Make sure you:
1. Activated the virtual environment: `.ipenv\Scripts\activate`
2. Installed dependencies: `pip install -r requirements.txt`
3. Are in the project root directory

### "No such file or directory: data/users.csv"

Generate test data first:
```bash
python data/src/dataset_generator.py
```

### Model Accuracy is Low (< 50%)

Possible causes:
- **Insufficient feedback data**: Collect more acceptance/rejection examples
- **Class imbalance**: Check `dataset["label"].value_counts()` - add more minority class examples
- **Poor feature engineering**: Some features may not correlate with acceptance
- **Hyperparameter tuning needed**: Adjust `max_iter`, `class_weight` in feedback_model.py

### Virtual Environment Not Working

Try removing and recreating it:
```bash
# Remove old environment
rmdir .ipenv /s /q

# Create new one
python -m venv .ipenv
.ipenv\Scripts\activate
pip install -r requirements.txt
```

### Memory Issues with Large Datasets

TF-IDF creates sparse matrices that consume memory:
- For 10K+ users: Consider dimensionality reduction or sparse matrix optimization
- Use `sparse=True` parameter in TFIDFEncoder if available
- Or implement approximate nearest neighbors (ANN) for scalability

### Recommendations All Have Same Score

- Check if feedback.csv has sufficient data (> 100 examples)
- Verify feature engineering in recommender.compatibility_score()
- Ensure model trained successfully (check training accuracy)

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and test
3. Commit with clear messages: `git commit -m "Add feature description"`
4. Push to branch: `git push origin feature/my-feature`
5. Open a Pull Request

## License

[Add your license here]

## Contact

[Add contact information here]

---

**Last Updated**: June 2026
