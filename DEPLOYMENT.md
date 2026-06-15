# NexMatch AI - Deployment Guide

## Prerequisites
- Python 3.11+
- Git

## 1. Local Development

### Setup
```bash
git clone <your-repo-url>
cd Intelligent_Recommender_System
python -m venv .ipenv
.ipenv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment
```bash
copy .env.example .env
```

Edit .env with your values (ADMIN_NAME, ADMIN_PASSWORD, SMTP settings).

### Run
```bash
streamlit run src/app.py
```

Access at: http://localhost:8501

---

## 2. Streamlit Community Cloud (Free)

1. Push code to GitHub
2. Go to share.streamlit.io
3. New app -> select repo -> Main file: src/app.py
4. Advanced settings -> add Secrets (TOML format):
   APP_NAME = "NexMatch AI"
   ADMIN_NAME = "admin"
   ADMIN_PASSWORD = "your-password"
   SECRET_KEY = "random-hex-32-chars"
5. Deploy

NOTE: Storage is ephemeral on free tier. CSV files reset on redeploy.

---

## 3. Render.com

The repo includes render.yaml.
1. Go to dashboard.render.com
2. New -> Blueprint -> connect GitHub repo
3. Set secret env vars: ADMIN_NAME, ADMIN_EMAIL, ADMIN_PASSWORD
4. Deploy

Manual:
- Build: pip install -r requirements.txt
- Start: streamlit run src/app.py --server.port  --server.address 0.0.0.0 --server.headless true

---

## 4. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| APP_NAME | NexMatch AI | Display name |
| SECRET_KEY | auto | Token signing |
| ADMIN_NAME | admin | Admin username |
| ADMIN_EMAIL | - | Admin email |
| ADMIN_PASSWORD | - | Admin password |
| SMTP_HOST | smtp.gmail.com | SMTP host |
| SMTP_PORT | 587 | SMTP port |
| SMTP_EMAIL | - | Sender email |
| SMTP_PASSWORD | - | Gmail App Password |
| ENABLE_EMAILS | false | Toggle emails |
| ENABLE_SIGNUPS | true | Allow registration |
| PASSWORD_RESET_EXPIRY_MINUTES | 30 | Token expiry |
| FEEDBACK_RETRAIN_THRESHOLD | 50 | Retrain alert |

---

## 5. Gmail SMTP Setup

1. Enable 2-Step Verification on Google account
2. Go to: myaccount.google.com/apppasswords
3. Create App Password for Mail
4. Set ENABLE_EMAILS=true, SMTP_EMAIL, SMTP_PASSWORD in .env

---

## 6. Security Notes

- Passwords stored with bcrypt (never plaintext)
- Session expires after 60 minutes
- Account locks after 5 failed logins (15 min)
- Reset tokens expire in 30 minutes, one-time use
- Never commit .env to git
