import os
import secrets
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Config:
    # App General settings
    APP_NAME = os.getenv("APP_NAME", "NextMatchAI")
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    
    # Session / Token Security
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        # Generate a secure secret key if none is provided
        SECRET_KEY = secrets.token_hex(32)
        
    # Admin Credentials
    ADMIN_NAME = os.getenv("ADMIN_NAME", "admin")
    ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@nextmatchai.ai")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "AdminSecurePassword2026!")
    
    # SMTP Service Settings
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    try:
        SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        SMTP_PORT = 587
        
    SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    
    # Features Enabled
    ENABLE_EMAILS = os.getenv("ENABLE_EMAILS", "false").lower() in ("true", "1", "yes")
    ENABLE_SIGNUPS = os.getenv("ENABLE_SIGNUPS", "true").lower() in ("true", "1", "yes")
    ENABLE_PASSWORD_RESET = os.getenv("ENABLE_PASSWORD_RESET", "true").lower() in ("true", "1", "yes")
    
    # Thresholds / Expirations
    try:
        PASSWORD_RESET_EXPIRY_MINUTES = int(os.getenv("PASSWORD_RESET_EXPIRY_MINUTES", "30"))
    except ValueError:
        PASSWORD_RESET_EXPIRY_MINUTES = 30
        
    try:
        FEEDBACK_RETRAIN_THRESHOLD = int(os.getenv("FEEDBACK_RETRAIN_THRESHOLD", "50"))
    except ValueError:
        FEEDBACK_RETRAIN_THRESHOLD = 50
        
    # File / Folder Paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models")
    DATA_PATH = os.getenv("DATA_PATH", "data")
