# Intelligent Recommender System

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

4. **Run the main application**
   ```bash
   python src/main.py
   ```

## Project Architecture

```
┌─────────────────────────────────────────────┐
│         Data Layer (data/)                   │
│  • users.csv, feedback.csv, datasets         │
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
│       Matching Layer (src/matching/)         │
│  • SimilarityEngine: Calculate similarities  │
│  • MBTIEngine: MBTI-based matching           │
│  • Recommender: Final recommendations        │
└─────────────────────────────────────────────┘
            ↓
┌─────────────────────────────────────────────┐
│         Output (recommendations)             │
│  • API responses or reports                  │
└─────────────────────────────────────────────┘
```

### How It Works

1. **Data Ingestion**: User profiles and items are loaded from CSV files
2. **Text Preprocessing**: Text data is cleaned and normalized
3. **Embedding Generation**: TF-IDF encoder converts text into numerical vectors
4. **Similarity Calculation**: Similarity engine computes distances between user and item embeddings
5. **MBTI Matching**: MBTI engine applies personality-based filtering
6. **Ranking & Recommendation**: Top recommendations are returned based on similarity scores

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

### Running the Recommender

```bash
python src/main.py
```

### Using Components Individually

```python
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.embeddings.tfidf_encoder import TFIDFEncoder
from src.matching.similarity_engine import SimilarityEngine
from src.matching.recommender import Recommender

# Initialize components
preprocessor = TextPreprocessor()
encoder = TFIDFEncoder()
similarity_engine = SimilarityEngine()
recommender = Recommender()

# Use them
cleaned_text = preprocessor.preprocess("Your text here")
embeddings = encoder.encode(cleaned_text)
scores = similarity_engine.calculate_similarity(embeddings)
recommendations = recommender.get_recommendations(user_id, top_n=10)
```

### Generating Test Data

```bash
python data/src/dataset_generator.py
```

## Project Structure

```
Intelligent_Recommender_System/
├── api/                          # API endpoints (Flask/FastAPI)
├── app/                          # Application UI/logic
├── data/                         # Datasets and data utilities
│   ├── users.csv                 # User profiles
│   ├── feedback.csv              # User feedback data
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
│   │   └── tfidf_encoder.py      # TF-IDF implementation
│   ├── matching/                 # Recommendation matching module
│   │   ├── similarity_engine.py  # Compute similarity scores
│   │   ├── mbti_engine.py        # MBTI personality matching
│   │   └── recommender.py        # Main recommender engine
│   └── preprocessing/            # Data preprocessing module
│       └── text_preprocessor.py  # Text cleaning & normalization
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Components

### 1. Text Preprocessor (`src/preprocessing/text_preprocessor.py`)
- Cleans and normalizes user and item text data
- Removes stopwords, handles tokenization
- Prepares text for embedding generation

### 2. TF-IDF Encoder (`src/embeddings/tfidf_encoder.py`)
- Converts text into TF-IDF (Term Frequency-Inverse Document Frequency) vectors
- Creates numerical representations for similarity calculations
- Handles dimensionality reduction

### 3. Similarity Engine (`src/matching/similarity_engine.py`)
- Calculates similarity scores between user embeddings and item embeddings
- Supports multiple similarity metrics (cosine, euclidean, etc.)
- Returns ranked similarity scores

### 4. MBTI Engine (`src/matching/mbti_engine.py`)
- Applies Myers-Briggs Type Indicator personality matching
- Filters recommendations based on personality compatibility
- Provides personality-based insights

### 5. Recommender (`src/matching/recommender.py`)
- Orchestrates the entire recommendation pipeline
- Combines similarity scores with MBTI filtering
- Returns top-N personalized recommendations

## Development

### Adding New Features

1. **Create a new module** in the appropriate package
2. **Add imports** to `__init__.py` files
3. **Update requirements.txt** if adding dependencies:
   ```bash
   pip freeze > requirements.txt
   ```
4. **Test** using Jupyter notebooks in `notebooks/`

### Running Tests

```bash
# Install test dependencies (if added)
pip install pytest pytest-cov

# Run tests
pytest tests/
```

### Data Utilities

Generate test data:
```bash
python data/src/dataset_generator.py
```

Check data statistics:
```bash
python data/src/dataset_statistics.py
```

Validate data:
```bash
python data/src/dataset_validator.py
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

Make sure to generate test data first:
```bash
python data/src/dataset_generator.py
```

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
