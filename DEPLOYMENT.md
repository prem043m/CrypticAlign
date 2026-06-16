# 🚀 CrypticAlign — Deployment Guide

> **Complete deployment reference for CrypticAlign (NexMatch AI), the Intelligent Hybrid Professional Recommendation Platform.**

---

## 📋 Table of Contents

1. [Local Development Setup](#-1-local-development-setup)
2. [Environment Variables](#-2-environment-variables)
3. [Streamlit Community Cloud](#-3-streamlit-community-cloud)
4. [Render.com Deployment](#-4-rendercom-deployment)
5. [Docker Deployment](#-5-docker-deployment)
6. [Gmail SMTP Setup](#-6-gmail-smtp-setup)
7. [Security Notes](#-7-security-notes)
8. [Production Checklist](#-8-production-checklist)
9. [Health Check & Monitoring](#-9-health-check--monitoring)

---

## 🖥️ 1. Local Development Setup

### Prerequisites

| Requirement | Version | Check Command       |
|-------------|---------|---------------------|
| Python      | 3.11+   | `python --version`  |
| pip         | 23.0+   | `pip --version`     |
| Git         | 2.30+   | `git --version`     |

### Clone & Install

**Windows (PowerShell)**
```powershell
git clone https://github.com/prem043m/CrypticAlign.git
cd Intelligent_Recommender_System
python -m venv .ipenv
.ipenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
git clone https://github.com/prem043m/CrypticAlign.git
cd Intelligent_Recommender_System
python3 -m venv .ipenv
source .ipenv/bin/activate
pip install -r requirements.txt
```

### Configure Environment

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` in your editor and set the required values — see [Environment Variables](#-2-environment-variables) below.

### Generate Data & Train Model

```bash
# Generate synthetic dataset
python data/src/dataset_generator.py

# Train the ML pipeline (TF-IDF + Logistic Regression)
python src/main.py
```

### Run the Application

```bash
streamlit run src/app.py
```

Access the app at: **http://localhost:8501**

> [!TIP]
> Login with the `ADMIN_NAME` and `ADMIN_PASSWORD` values from your `.env` file to access the Admin Portal.

---

## ⚙️ 2. Environment Variables

All configuration is managed through environment variables. Copy `.env.example` to `.env` for local development, or set them in your deployment platform's secrets/config panel.

### Core Application

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `APP_NAME` | `NexMatch AI` | No | Display name shown in the UI header |
| `APP_ENV` | `development` | No | Environment identifier (`development` / `production`) |
| `APP_VERSION` | `1.0.0` | No | Application version string |
| `SECRET_KEY` | `auto` | **Yes** | Secret key for token signing & session security. Use a random 32+ character hex string in production |

### Admin Credentials

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `ADMIN_NAME` | `admin` | **Yes** | Admin account username |
| `ADMIN_EMAIL` | — | **Yes** | Admin email address (used for password reset) |
| `ADMIN_PASSWORD` | — | **Yes** | Admin account password (stored as bcrypt hash) |

### SMTP / Email

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SMTP_HOST` | `smtp.gmail.com` | No | SMTP server hostname |
| `SMTP_PORT` | `587` | No | SMTP server port (TLS) |
| `SMTP_EMAIL` | — | If emails enabled | Sender email address |
| `SMTP_PASSWORD` | — | If emails enabled | Gmail App Password (see [Gmail SMTP Setup](#-6-gmail-smtp-setup)) |

### Feature Flags

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `ENABLE_EMAILS` | `false` | No | Enable/disable outbound email notifications (welcome emails, password resets) |
| `ENABLE_SIGNUPS` | `true` | No | Allow new user registration via the UI |
| `ENABLE_PASSWORD_RESET` | `true` | No | Enable the password reset flow |

### Limits & Thresholds

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PASSWORD_RESET_EXPIRY_MINUTES` | `30` | No | Time (in minutes) before a password reset token expires |
| `FEEDBACK_RETRAIN_THRESHOLD` | `50` | No | Number of new feedback entries before triggering a retrain alert |

### Paths

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `MODEL_PATH` | `models` | No | Directory for saved model artifacts |
| `DATA_PATH` | `data` | No | Directory for CSV data files |

---

## ☁️ 3. Streamlit Community Cloud

The simplest way to deploy CrypticAlign for free.

### Step-by-Step

1. **Push your code** to a public or private GitHub repository

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Create a new app:**
   - Select your repository
   - Branch: `main`
   - Main file path: `src/app.py`

4. **Configure Secrets** (Advanced Settings → Secrets):
   ```toml
   APP_NAME = "CrypticAlign"
   APP_ENV = "production"
   SECRET_KEY = "your-random-hex-key-at-least-32-characters"

   ADMIN_NAME = "admin"
   ADMIN_EMAIL = "admin@yourdomain.com"
   ADMIN_PASSWORD = "YourSecurePassword123!"

   ENABLE_EMAILS = "false"
   ENABLE_SIGNUPS = "true"
   ENABLE_PASSWORD_RESET = "true"

   SMTP_HOST = "smtp.gmail.com"
   SMTP_PORT = "587"
   SMTP_EMAIL = "your_email@gmail.com"
   SMTP_PASSWORD = "your_gmail_app_password"
   ```

5. **Click Deploy** 🎉

> [!WARNING]
> **Ephemeral Storage**: Streamlit Community Cloud's free tier uses ephemeral file storage. CSV data files (users, feedback, credentials) will reset on every redeploy or app restart. This tier is best suited for demos and evaluations, not production use.

### Custom Domain (Optional)

After deployment, you can configure a custom subdomain under `*.streamlit.app` from the app settings panel.

---

## 🌐 4. Render.com Deployment

The repository includes a pre-configured `render.yaml` blueprint for one-click deployment.

### Blueprint Deployment (Recommended)

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New → Blueprint**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure the service
5. Set the following **secret** environment variables in the Render dashboard:

   | Variable | Notes |
   |----------|-------|
   | `ADMIN_NAME` | Your admin username |
   | `ADMIN_EMAIL` | Your admin email |
   | `ADMIN_PASSWORD` | Strong admin password |
   | `SMTP_EMAIL` | Gmail address (if enabling emails) |
   | `SMTP_PASSWORD` | Gmail App Password (if enabling emails) |

6. Click **Apply** to deploy

### Manual Deployment

If you prefer manual setup instead of the blueprint:

| Setting | Value |
|---------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |

> [!NOTE]
> The `render.yaml` blueprint auto-generates a `SECRET_KEY` and sets sensible defaults for all non-secret variables.

---

## 🐳 5. Docker Deployment

Use Docker for consistent, reproducible deployments on any infrastructure.

### Sample Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
# ──────────────────────────────────────────────
# CrypticAlign (NexMatch AI) — Production Image
# ──────────────────────────────────────────────
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Copy application source
COPY . .

# Generate dataset & train model during build
RUN python data/src/dataset_generator.py && \
    python src/main.py

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
```

### Build & Run

```bash
# Build the image
docker build -t crypticalign:latest .

# Run with environment variables
docker run -d \
  --name crypticalign \
  -p 8501:8501 \
  -e SECRET_KEY="your-production-secret-key" \
  -e ADMIN_NAME="admin" \
  -e ADMIN_EMAIL="admin@yourdomain.com" \
  -e ADMIN_PASSWORD="YourSecurePassword123!" \
  -e ENABLE_EMAILS="false" \
  crypticalign:latest
```

Access at: **http://localhost:8501**

### Docker Compose (Optional)

Create a `docker-compose.yml` for easier management:

```yaml
version: "3.9"

services:
  crypticalign:
    build: .
    container_name: crypticalign
    ports:
      - "8501:8501"
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
```

```bash
docker-compose up -d
```

> [!TIP]
> For persistent data across container restarts, mount the `data/` and `models/` directories as Docker volumes:
> ```bash
> docker run -d \
>   -v $(pwd)/data:/app/data \
>   -v $(pwd)/models:/app/models \
>   ...
> ```

---

## 📧 6. Gmail SMTP Setup

CrypticAlign uses SMTP to send welcome emails and password reset tokens. Gmail is the default and recommended provider.

### Step-by-Step Configuration

1. **Enable 2-Step Verification** on your Google Account
   - Go to: [myaccount.google.com/security](https://myaccount.google.com/security)
   - Under "Signing in to Google" → enable **2-Step Verification**

2. **Generate an App Password**
   - Go to: [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
   - Select app: **Mail**
   - Select device: **Other** (enter "CrypticAlign")
   - Click **Generate** — copy the 16-character password

3. **Update your `.env` file:**
   ```dotenv
   ENABLE_EMAILS=true
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_EMAIL=your_email@gmail.com
   SMTP_PASSWORD=abcd efgh ijkl mnop
   ```

4. **For cloud deployments**, set `SMTP_EMAIL` and `SMTP_PASSWORD` as secrets in your platform's dashboard (Streamlit Cloud → Secrets, Render → Environment Variables, Docker → `-e` flags).

> [!IMPORTANT]
> **Do NOT use your regular Gmail password.** Google requires App Passwords for third-party SMTP access when 2-Step Verification is enabled. Regular passwords will be rejected.

### Alternative SMTP Providers

| Provider | Host | Port | Notes |
|----------|------|------|-------|
| Gmail | `smtp.gmail.com` | `587` | Requires App Password |
| Outlook | `smtp.office365.com` | `587` | Microsoft 365 account |
| SendGrid | `smtp.sendgrid.net` | `587` | API key as password |
| Mailgun | `smtp.mailgun.org` | `587` | Domain verification required |

---

## 🔒 7. Security Notes

CrypticAlign implements multiple layers of security. Review and enforce these in production:

### Authentication & Password Security

| Feature | Implementation |
|---------|---------------|
| **Password Hashing** | All passwords are hashed with **bcrypt** before storage — plaintext passwords are never persisted |
| **Session Expiry** | User sessions automatically expire after **60 minutes** of inactivity |
| **Account Lockout** | Accounts are locked after **5 failed login attempts** within a **15-minute** window |
| **Password Reset Tokens** | Tokens expire after **30 minutes** and are single-use (consumed after successful reset) |
| **Input Validation** | Usernames, passwords, emails, and profile fields are validated server-side before processing |

### Secrets Management

| Practice | Details |
|----------|---------|
| **`.env` file** | Listed in `.gitignore` — never committed to version control |
| **`SECRET_KEY`** | Used for session token signing. Must be a random, unpredictable string (32+ chars) in production |
| **Cloud Secrets** | Always use platform-native secrets management (Streamlit Secrets, Render Environment Variables, Docker secrets) |

### Production Security Checklist

- [ ] Generate a **cryptographically random** `SECRET_KEY` (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`)
- [ ] Set **strong** `ADMIN_PASSWORD` (12+ chars, mixed case, numbers, symbols)
- [ ] Set `APP_ENV=production`
- [ ] Enable HTTPS (via reverse proxy — Nginx, Caddy, or platform-provided TLS)
- [ ] Review and restrict `ENABLE_SIGNUPS` if not needed in production
- [ ] Set up **log monitoring** for `LOGIN_FAILED` and `SESSION_EXPIRED` audit events
- [ ] Regularly rotate the `SECRET_KEY` and `SMTP_PASSWORD`
- [ ] Ensure the `data/credentials.csv` file is not publicly accessible

> [!CAUTION]
> Never deploy with the default `SECRET_KEY` from `.env.example`. Always generate a unique, random key for every deployment environment.

---

## ✅ 8. Production Checklist

Use this checklist before going live:

### Infrastructure

- [ ] Python 3.11+ runtime confirmed
- [ ] All dependencies installed from `requirements.txt`
- [ ] Dataset generated (`data/users.csv`, `data/feedback.csv`)
- [ ] Model trained and saved (`models/feedback_model.pkl`, `models/tfidf_vectorizer.pkl`)
- [ ] `.env` configured with production values

### Security

- [ ] Unique, random `SECRET_KEY` set
- [ ] Strong `ADMIN_PASSWORD` configured
- [ ] `APP_ENV` set to `production`
- [ ] `.env` file excluded from Git (verify `.gitignore`)
- [ ] HTTPS/TLS enabled via reverse proxy or platform

### Email (if enabled)

- [ ] Gmail App Password generated and configured
- [ ] `ENABLE_EMAILS=true` set
- [ ] `SMTP_EMAIL` and `SMTP_PASSWORD` configured
- [ ] Test email sent successfully

### Monitoring

- [ ] Health check endpoint verified (`/_stcore/health`)
- [ ] Audit log (`data/audit_log.csv`) accessible for review
- [ ] Error alerts configured (platform-dependent)

### Data Persistence

- [ ] Data directory (`data/`) is on persistent storage (not ephemeral)
- [ ] Backup strategy in place for `credentials.csv` and `feedback.csv`
- [ ] Model artifacts (`models/`) backed up

---

## 📊 9. Health Check & Monitoring

### Health Check Endpoint

Streamlit exposes a built-in health check at:

```
GET http://<host>:<port>/_stcore/health
```

**Healthy response:**
```
"ok"
```

Use this for load balancer health probes, Docker `HEALTHCHECK`, and uptime monitors.

### Audit Logging

CrypticAlign logs security-relevant events to `data/audit_log.csv`:

| Event Type | Description |
|------------|-------------|
| `LOGIN` | Successful user login |
| `LOGIN_FAILED` | Failed login attempt (wrong credentials) |
| `SESSION_EXPIRED` | Auto-logout after timeout |
| `REGISTER` | New user registration |
| `PASSWORD_RESET` | Password reset completed |

Access the audit log via:
- **Admin Portal → Audit Log** (in the UI)
- Direct file: `data/audit_log.csv`

### Key Metrics to Monitor

| Metric | Where to Check | Alert Threshold |
|--------|----------------|-----------------|
| App health | `/_stcore/health` | Response ≠ `"ok"` |
| Failed logins | `data/audit_log.csv` | > 10 per hour |
| Session expirations | `data/audit_log.csv` | Unusual spikes |
| Feedback volume | `data/feedback.csv` row count | Approaching `FEEDBACK_RETRAIN_THRESHOLD` |
| Disk usage | OS-level monitoring | > 90% capacity |
| Model staleness | `models/model_metadata.json` | > 7 days since last training |

### Recommended Uptime Monitors

| Tool | Free Tier | Setup |
|------|-----------|-------|
| [UptimeRobot](https://uptimerobot.com) | 50 monitors | HTTP check on `/_stcore/health` |
| [Better Uptime](https://betteruptime.com) | 10 monitors | HTTP check with Slack/email alerts |
| [Render Health Checks](https://render.com) | Built-in | Auto-configured via `render.yaml` |

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| [`quickstart.md`](quickstart.md) | Get running locally in 5 minutes |
| [`README.md`](README.md) | Full project overview & architecture |
| [`notebook/model_evaluation.ipynb`](notebook/model_evaluation.ipynb) | Model performance analysis |
| [`reports/project_report.md`](reports/project_report.md) | Detailed project report |

---

<p align="center">
  <strong>CrypticAlign (NexMatch AI)</strong> — Intelligent Hybrid Professional Recommendation Platform<br/>
  <em>Built with Streamlit · scikit-learn · NLTK · Pandas · bcrypt</em>
</p>
