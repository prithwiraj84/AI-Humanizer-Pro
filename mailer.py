"""
Transactional email: verification links, password-reset codes, receipts, plan reminders.

Messages are rendered from templates/emails/<name>.html and .txt and sent over SMTP
(Gmail by default). Sending happens on a background thread so requests don't wait on the
mail server; set MAIL_SYNC=1 to send inline (useful in tests and scripts).
"""
import logging
import os
import smtplib
import threading
from email.message import EmailMessage
from email.utils import formataddr

from flask import render_template

logger = logging.getLogger(__name__)

SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_TIMEOUT = 10


def is_configured():
    return bool(os.environ.get('SENDER_EMAIL') and os.environ.get('SENDER_PASSWORD'))


def deliver(message):
    """Send a prepared EmailMessage now. Returns True on success."""
    if not is_configured():
        logger.error('Email not sent (%s): SENDER_EMAIL and SENDER_PASSWORD must be set', message['Subject'])
        return False
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=SMTP_TIMEOUT) as server:
            server.starttls()
            server.login(os.environ['SENDER_EMAIL'], os.environ['SENDER_PASSWORD'])
            server.send_message(message)
        return True
    except Exception as e:
        logger.error('Email not sent (%s): %s', message['Subject'], type(e).__name__)
        return False


def build(to, subject, template, attachments=(), **context):
    """Render a template pair into an EmailMessage. `attachments`: (filename, bytes, mime type)."""
    message = EmailMessage()
    message['From'] = formataddr((os.environ.get('BUSINESS_NAME', 'AI Humanizer'), os.environ.get('SENDER_EMAIL', '')))
    message['To'] = to
    message['Subject'] = subject
    context.setdefault('subject', subject)
    context.setdefault('business_name', os.environ.get('BUSINESS_NAME', 'AI Humanizer'))
    message.set_content(render_template(f'emails/{template}.txt', **context))
    message.add_alternative(render_template(f'emails/{template}.html', **context), subtype='html')
    for filename, data, mime in attachments:
        maintype, subtype = mime.split('/')
        message.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return message


def send(to, subject, template, attachments=(), **context):
    """Render and send. Returns True if sent (inline) or queued (background)."""
    message = build(to, subject, template, attachments, **context)
    if os.environ.get('MAIL_SYNC') == '1':
        return deliver(message)
    if not is_configured():
        return deliver(message)   # logs the configuration problem
    threading.Thread(target=deliver, args=(message,), daemon=True).start()
    return True
