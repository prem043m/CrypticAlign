import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from src.utils.config import Config

def get_user_id_by_email(email: str) -> str:
    try:
        from src.utils.data_manager import load_credentials_raw
        df = load_credentials_raw()
        match = df[df["email"].str.lower() == email.lower()]
        if not match.empty:
            return str(match.iloc[0]["user_id"])
    except Exception:
        pass
    return ""

def get_notification_preference(email: str, pref_key: str) -> bool:
    try:
        user_id = get_user_id_by_email(email)
        if not user_id:
            return True
        from src.utils.data_manager import load_user_profiles_raw
        df = load_user_profiles_raw()
        match = df[df["user_id"] == user_id]
        if not match.empty:
            val = match.iloc[0].get(pref_key, True)
            if pd.notna(val):
                return bool(val)
    except Exception:
        pass
    return True

def _send_email(to: str, subject: str, html_body: str) -> bool:
    """Internal SMTP sender helper, logging failures and respecting Config settings."""
    if not Config.ENABLE_EMAILS:
        # Silent return if emails are disabled
        return True
        
    if not Config.SMTP_EMAIL or not Config.SMTP_PASSWORD:
        print("SMTP credentials are not configured. Cannot send email.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{Config.APP_NAME} <{Config.SMTP_EMAIL}>"
        msg["To"] = to

        # Create html mime part
        html_part = MIMEText(html_body, "html")
        msg.attach(html_part)

        # Connect and send
        server = smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT)
        server.starttls()
        server.login(Config.SMTP_EMAIL, Config.SMTP_PASSWORD)
        server.sendmail(Config.SMTP_EMAIL, to, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email to {to}: {str(e)}")
        return False

def send_welcome_email(name: str, email: str) -> bool:
    """Sends a welcome message to a newly registered user if preferred."""
    if not get_notification_preference(email, "notif_welcome"):
        return True
    subject = f"Welcome to {Config.APP_NAME}!"
    html_body = f"""
    <html>
      <body style="font-family: sans-serif; color: #333;">
        <h2>Welcome to {Config.APP_NAME}, {name}!</h2>
        <p>Thank you for signing up on our platform.</p>
        <p>You can now log in, complete your profile, and receive smart partner recommendations powered by our AI hybrid engine.</p>
        <br>
        <p>Best regards,</p>
        <p>The {Config.APP_NAME} Team</p>
      </body>
    </html>
    """
    return _send_email(email, subject, html_body)

def send_password_reset_email(name: str, email: str, token: str) -> bool:
    """Sends a password reset token to a user."""
    subject = f"[{Config.APP_NAME}] Password Reset Request"
    html_body = f"""
    <html>
      <body style="font-family: sans-serif; color: #333;">
        <h2>Password Reset Request</h2>
        <p>Hello {name},</p>
        <p>We received a request to reset your password. Use the following security token to complete your reset:</p>
        <p style="font-size: 18px; font-weight: bold; background: #f0f0f0; padding: 10px; display: inline-block; letter-spacing: 2px;">
          {token}
        </p>
        <p>This token is valid for {Config.PASSWORD_RESET_EXPIRY_MINUTES} minutes.</p>
        <p>If you did not request this reset, please ignore this email or contact support.</p>
        <br>
        <p>Best regards,</p>
        <p>The {Config.APP_NAME} Team</p>
      </body>
    </html>
    """
    return _send_email(email, subject, html_body)

def send_match_notification_email(name: str, email: str, matched_name: str) -> bool:
    """Sends a notification email when a new match recommendation is accepted/mutual if preferred."""
    if not get_notification_preference(email, "notif_digest"):
        return True
    subject = f"New Connection recommendation on {Config.APP_NAME}!"
    html_body = f"""
    <html>
      <body style="font-family: sans-serif; color: #333;">
        <h2>Great News!</h2>
        <p>Hello {name},</p>
        <p>You have a new accepted connection match recommendation with <strong>{matched_name}</strong>!</p>
        <p>Log back into nextmatchAi to view contact details or exchange messages.</p>
        <br>
        <p>Best regards,</p>
        <p>The {Config.APP_NAME} Team</p>
      </body>
    </html>
    """
    return _send_email(email, subject, html_body)

def send_feedback_confirmation_email(name: str, email: str, count: int) -> bool:
    """Sends feedback recorded confirmation email if preferred."""
    if not get_notification_preference(email, "notif_system"):
        return True
    subject = f"Your Feedback Recorded - {Config.APP_NAME}"
    html_body = f"""
    <html>
      <body style="font-family: sans-serif; color: #333;">
        <h2>Feedback Recorded</h2>
        <p>Hello {name},</p>
        <p>We've successfully saved {count} match recommendation feedback decision(s).</p>
        <p>Your inputs are queued for the adaptive recommendation model. The system will retrain periodically to align recommendations with your preferences.</p>
        <br>
        <p>Best regards,</p>
        <p>The {Config.APP_NAME} Team</p>
      </body>
    </html>
    """
    return _send_email(email, subject, html_body)

def check_smtp_status() -> tuple[bool, str]:
    """Tests SMTP connection status without sending an email."""
    if not Config.ENABLE_EMAILS:
        return False, "Disabled in environment configuration (.env)"
    if not Config.SMTP_EMAIL or not Config.SMTP_PASSWORD:
        return False, "SMTP credentials missing"
    try:
        server = smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT, timeout=5)
        server.starttls()
        server.login(Config.SMTP_EMAIL, Config.SMTP_PASSWORD)
        server.quit()
        return True, "Connected successfully"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
