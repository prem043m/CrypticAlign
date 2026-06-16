# 🚀 Quick Start Guide — CrypticAlign (NexMatch AI)

> **Get the Intelligent Hybrid Professional Recommendation System running locally in under 5 minutes.**

---

## 📋 Prerequisites

Before you begin, make sure the following are installed on your machine:

| Requirement | Minimum Version | Check Command           |
|-------------|-----------------|-------------------------|
| **Python**  | 3.11+           | `python --version`      |
| **pip**     | 23.0+           | `pip --version`         |
| **Git**     | 2.30+           | `git --version`         |

> [!NOTE]
> Python 3.11.9 is the tested runtime (see `runtime.txt`). Python 3.12+ should also work, but 3.11 is recommended for full compatibility.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/prem043m/CrypticAlign.git
cd Intelligent_Recommender_System
```

---

## 2️⃣ Create a Virtual Environment

Create an isolated Python environment named `.ipenv`:

**Windows (PowerShell)**
```powershell
python -m venv .ipenv
.ipenv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```cmd
python -m venv .ipenv
.ipenv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .ipenv
source .ipenv/bin/activate
```

> [!TIP]
> You should see `(.ipenv)` in your terminal prompt once the virtual environment is active.

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including:

| Package          | Purpose                                 |
|------------------|-----------------------------------------|
| `streamlit`      | Interactive web UI                      |
| `scikit-learn`   | TF-IDF vectorization & Logistic Regression |
| `nltk`           | Natural language text preprocessing     |
| `pandas`         | Data manipulation & CSV handling        |
| `plotly`         | Interactive charts & analytics          |
| `bcrypt`         | Secure password hashing                 |
| `python-dotenv`  | Environment variable management         |
| `Faker`          | Synthetic dataset generation            |

---

## 4️⃣ Configure Environment Variables

Copy the example environment file and edit it with your own values:

**Windows**
```cmd
copy .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

Open `.env` in your editor and set at minimum:

```dotenv
ADMIN_NAME=admin
ADMIN_EMAIL=admin@nexmatch.ai
ADMIN_PASSWORD=YourSecurePassword123!
SECRET_KEY=your-random-hex-key-32-chars
```

> [!IMPORTANT]
> The `.env` file contains secrets (admin credentials, SMTP passwords). It is listed in `.gitignore` — **never commit it to version control**.

---

## 5️⃣ Generate the Dataset

Generate the synthetic user and feedback datasets:

```bash
python data/src/dataset_generator.py
```

This creates the following data files in the `data/` directory:

| File              | Description                          |
|-------------------|--------------------------------------|
| `users.csv`       | 200+ synthetic user profiles         |
| `feedback.csv`    | User interaction feedback records    |
| `user_profiles.csv` | Extended profile metadata          |
| `credentials.csv` | Authentication credentials store     |

> [!NOTE]
> If data files already exist, they will be used as-is. To regenerate from scratch, delete the existing CSV files and re-run the command.

---

## 6️⃣ Train the ML Model

Run the full training pipeline — TF-IDF encoding → similarity computation → feedback model training:

```bash
python src/main.py
```

**What happens under the hood:**

1. **TF-IDF Encoding** — Converts user profile text into feature vectors (`tfidf_vectorizer.pkl`)
2. **Similarity Engine** — Computes cosine similarity between user embeddings
3. **Feedback Model** — Trains a Logistic Regression classifier on interaction data (`feedback_model.pkl`)
4. **Adaptive Recommender** — Combines content-based similarity with ML re-ranking

**Expected output:**
```
======= Training Feedback Shape =======
 (N, M)

======= training label check =======
 ...

====================================================
STAGE 1: Candidate Generation
====================================================
Candidate Pool Size: 30
...
```

Trained model artifacts are saved to the `models/` directory:

| File                    | Description                              |
|-------------------------|------------------------------------------|
| `feedback_model.pkl`    | Trained Logistic Regression model        |
| `tfidf_vectorizer.pkl`  | Fitted TF-IDF vectorizer                |
| `model_metadata.json`   | Model version, timestamp, and metrics    |

---

## 7️⃣ Launch the Web UI

Start the Streamlit application:

```bash
streamlit run src/app.py
```

The app will open automatically in your browser at:

```
http://localhost:8501
```

### 🔑 First Login

| Field    | Value                                  |
|----------|----------------------------------------|
| Username | Value of `ADMIN_NAME` in `.env`        |
| Password | Value of `ADMIN_PASSWORD` in `.env`    |

### 🧭 Available Portals

| Portal           | Access       | Features                                                              |
|------------------|--------------|-----------------------------------------------------------------------|
| **User Portal**  | All users    | Home, My Profile, Recommendations, Feedback History, Recommendation History |
| **Admin Portal** | Admin only   | Dashboard, User Explorer, Model Analytics, Dataset Insights, Explainability, System Status, Quality Audit, Monitoring, User Management, Data Management, Audit Log |

---

## 8️⃣ Run the Evaluation Notebook

For detailed model performance analysis, open the Jupyter notebook:

```bash
jupyter notebook notebook/model_evaluation.ipynb
```

Or if you use VS Code, simply open the file:
```
notebook/model_evaluation.ipynb
```

The notebook contains:
- Model accuracy & classification metrics
- Confusion matrix visualization
- Feature importance analysis
- ROC curve and AUC evaluation

---

## ✅ Quick Verification Commands

Run these commands to verify everything is set up correctly:

```bash
# Check Python version (should be 3.11+)
python --version

# Verify virtual environment is active
pip --version    # Should show path inside .ipenv/

# Confirm key packages are installed
python -c "import streamlit; print('Streamlit', streamlit.__version__)"
python -c "import sklearn; print('scikit-learn', sklearn.__version__)"
python -c "import nltk; print('NLTK', nltk.__version__)"
python -c "import pandas; print('Pandas', pandas.__version__)"
python -c "import bcrypt; print('bcrypt', bcrypt.__version__)"

# Verify dataset files exist
python -c "from pathlib import Path; [print(f'  ✓ {f.name} ({f.stat().st_size:,} bytes)') for f in Path('data').glob('*.csv')]"

# Verify trained model files exist
python -c "from pathlib import Path; [print(f'  ✓ {f.name} ({f.stat().st_size:,} bytes)') for f in Path('models').iterdir()]"
```

---

## ❓ Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

You're running a script from the wrong directory. Always run commands from the **project root**:
```bash
# ✗ Wrong
cd src && python main.py

# ✓ Correct
python src/main.py
```

### `streamlit: command not found`

Your virtual environment is not activated. Activate it first:
```bash
# Windows PowerShell
.ipenv\Scripts\Activate.ps1

# macOS / Linux
source .ipenv/bin/activate
```

### NLTK data not found

If you see errors about missing NLTK resources (e.g., `punkt`, `stopwords`), download them manually:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Port 8501 already in use

Another Streamlit instance may be running. Kill it or specify a different port:
```bash
streamlit run src/app.py --server.port 8502
```

### `FileNotFoundError` for CSV files

The dataset hasn't been generated yet. Run:
```bash
python data/src/dataset_generator.py
```

### Model files not found / stale model warnings

Retrain the model from scratch:
```bash
python src/main.py
```

### Virtual environment not recognized (Windows)

If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then re-activate:
```powershell
.ipenv\Scripts\Activate.ps1
```

---

## 📂 Project Structure at a Glance

```
Intelligent_Recommender_System/
├── src/
│   ├── app.py                  # Streamlit web application entry point
│   ├── main.py                 # ML pipeline: train & evaluate
│   ├── embeddings/             # TF-IDF encoder
│   ├── matching/               # Similarity engine & recommender
│   ├── learning/               # Feedback model & adaptive recommender
│   ├── views/                  # User portal & admin portal UI views
│   └── utils/                  # Config, auth, email, data management
├── data/
│   ├── src/                    # Dataset & feedback generators
│   ├── users.csv               # Generated user profiles
│   └── feedback.csv            # Generated user feedback
├── models/
│   ├── feedback_model.pkl      # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│   └── model_metadata.json     # Model metadata & metrics
├── notebook/
│   └── model_evaluation.ipynb  # Performance evaluation notebook
├── reports/
│   └── project_report.md       # Project report
├── .env.example                # Environment variable template
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render.com deployment config
├── runtime.txt                 # Python runtime version
└── quickstart.md               # ← You are here
```

---

## 🔗 Next Steps

| Action                    | Resource                          |
|---------------------------|-----------------------------------|
| Deploy to production      | [`DEPLOYMENT.md`](DEPLOYMENT.md)  |
| Review model performance  | [`notebook/model_evaluation.ipynb`](notebook/model_evaluation.ipynb) |
| View project report       | [`reports/project_report.md`](reports/project_report.md) |
| Full project overview     | [`README.md`](README.md)         |

---

<p align="center">
  <strong>CrypticAlign (NexMatch AI)</strong> — Intelligent Hybrid Professional Recommendation Platform<br/>
  <em>Built with Streamlit · scikit-learn · NLTK · Pandas</em>
</p>