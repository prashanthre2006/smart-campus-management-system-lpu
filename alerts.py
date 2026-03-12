import base64
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

from config import (
    ALERTS_ENABLED,
    ALERTS_ON_MARKED,
    ALERTS_ON_UNKNOWN,
    SMTP_FROM,
    SMTP_HOST,
    SMTP_PASS,
    SMTP_PORT,
    SMTP_TO,
    SMTP_USER,
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_FROM,
    TWILIO_TO,
)


def _send_email(subject: str, body: str) -> None:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_TO):
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = SMTP_TO
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


def _send_sms(body: str) -> None:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM and TWILIO_TO):
        return

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = urllib.parse.urlencode(
        {"From": TWILIO_FROM, "To": TWILIO_TO, "Body": body}
    ).encode("utf-8")

    auth = f"{TWILIO_ACCOUNT_SID}:{TWILIO_AUTH_TOKEN}".encode("utf-8")
    headers = {
        "Authorization": "Basic " + base64.b64encode(auth).decode("utf-8")
    }

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=10):
        pass


def send_alert(event: str, message: str) -> None:
    if not ALERTS_ENABLED:
        return

    if event == "unknown" and not ALERTS_ON_UNKNOWN:
        return
    if event == "marked" and not ALERTS_ON_MARKED:
        return

    subject = f"Attendance Alert: {event.title()}"
    _send_email(subject, message)
    _send_sms(message)
