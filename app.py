import os
import math
import re
import hmac
import time
import json
import logging
import secrets
import hashlib
import threading
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from functools import wraps
from urllib.parse import urlparse
from xml.sax.saxutils import escape as xml_escape
import requests
from flask import (Flask, render_template, request, jsonify, session, redirect, url_for, flash,
                   Response, send_from_directory, has_app_context, has_request_context)
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv
import db
import billing
import invoices
import mailer
import admin
import humanizer
from humanizer import TONES, perform_nuclear_chaos, calculate_diff_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SETUP: LOAD SECRETS FROM .env (on Render, set them as environment variables instead) ---
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- SETUP: LOGGING (stdout/stderr, collected by Render's log viewer) ---
logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('ai_humanizer')

app = Flask(__name__)

# Behind Render's proxy, trust its X-Forwarded-* headers so external URLs use https
ON_RENDER = bool(os.environ.get("RENDER"))
if ON_RENDER or os.environ.get("TRUST_PROXY") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# --- CRITICAL CONFIGURATION ---
# 1. Static Secret Key (from .env): Prevents logouts on server restart
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    logger.warning("SECRET_KEY is not set; using a temporary key (users will be logged out on restart)")
    app.secret_key = secrets.token_hex(32)

# 2. Strict Slashes False: Makes /dashboard and /dashboard/ work the same (canonical links pick one)
app.url_map.strict_slashes = False

# 3. Template Folder: Ensure it looks in current directory/templates
app.template_folder = 'templates'

# 4. Request Size & Cookies: Reject oversized bodies, keep the session cookie same-site
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = ON_RENDER

# 5. Static files are fingerprinted (?v=hash), so browsers may cache them for a year
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(days=365)

# 6. Public address used in canonical links, the sitemap and social previews (e.g. https://example.com).
#    Falls back to the address the request came in on.
SITE_URL = os.environ.get("SITE_URL", "").rstrip('/')
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")

# 7. Business details shown on the contact and legal pages (Razorpay checks these before going live)
BUSINESS_NAME = os.environ.get("BUSINESS_NAME", "AI Humanizer")
BUSINESS_ADDRESS = os.environ.get("BUSINESS_ADDRESS", "")
CONTACT_PHONE = os.environ.get("CONTACT_PHONE", "")
GRIEVANCE_OFFICER = os.environ.get("GRIEVANCE_OFFICER", "")   # name required by Indian IT/data protection rules
JURISDICTION_CITY = os.environ.get("JURISDICTION_CITY", "")   # courts named in the terms, e.g. "Kolkata"

# 8. Country used for pricing when the visitor's IP can't be located (e.g. "IN" for local testing)
GEO_DEFAULT_COUNTRY = os.environ.get("GEO_DEFAULT_COUNTRY", "").upper() or None

# 9. Bot protection on sign-up with Cloudflare Turnstile (free). Enabled when both keys are set.
TURNSTILE_SITE_KEY = os.environ.get("TURNSTILE_SITE_KEY", "")
TURNSTILE_SECRET_KEY = os.environ.get("TURNSTILE_SECRET_KEY", "")
TURNSTILE_VERIFY_URL = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'

# 10. Shared secret for the daily scheduled tasks (POST /internal/tasks, called by GitHub Actions)
CRON_SECRET = os.environ.get("CRON_SECRET", "")

# --- LIMITS ---
MAX_TEXT_LENGTH = 20000
MAX_BATCH_ITEMS = 25
MAX_BATCH_CHARS = 50000
MIN_PASSWORD_LENGTH = 8
VERIFY_LINK_LIFETIME = timedelta(days=3)
OTP_LIFETIME = timedelta(minutes=10)
OTP_RESEND_COOLDOWN = timedelta(seconds=60)
MAX_OTP_ATTEMPTS = 5
MAX_LOGIN_FAILURES = 5
LOGIN_LOCKOUT_WINDOW = timedelta(minutes=15)

# Retention promised in the privacy policy; enforced by housekeeping() below
ABANDONED_ORDER_RETENTION = timedelta(days=30)
HOUSEKEEPING_INTERVAL = 600  # seconds between clean-up runs

# Requests allowed per window (seconds), per client IP, or per user/API key for /humanize
RATE_LIMITS = {
    'login': (20, 600),
    'signup': (5, 3600),
    'send_otp': (5, 3600),
    'reset_password': (20, 3600),
    'humanize': (30, 60),
    'billing': (20, 3600),
    'coupon': (30, 3600),
    'verify_email': (5, 3600),
}

# --- EMAIL ---
# Sent over SMTP with SENDER_EMAIL / SENDER_PASSWORD from .env (for Gmail, an app password:
# Google Account -> Security -> 2-Step Verification -> App passwords). See mailer.py.

# --- SOCIAL LOGIN (GOOGLE & GITHUB) ---
# Each provider is enabled only when its client ID and secret are set in .env
oauth = OAuth(app)
OAUTH_NAMES = {'google': 'Google', 'github': 'GitHub'}
OAUTH_PROVIDERS = set()
OAUTH_TIMEOUT = 10

# Accounts created through social login get this marker instead of a password hash, so
# password login never matches. The owner can still set a password via "Forgot password".
OAUTH_ONLY_PASSWORD = '!oauth'

if os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"):
    oauth.register(
        name='google',
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile', 'default_timeout': OAUTH_TIMEOUT},
    )
    OAUTH_PROVIDERS.add('google')

if os.environ.get("GITHUB_CLIENT_ID") and os.environ.get("GITHUB_CLIENT_SECRET"):
    oauth.register(
        name='github',
        client_id=os.environ["GITHUB_CLIENT_ID"],
        client_secret=os.environ["GITHUB_CLIENT_SECRET"],
        authorize_url='https://github.com/login/oauth/authorize',
        access_token_url='https://github.com/login/oauth/access_token',
        api_base_url='https://api.github.com/',
        client_kwargs={'scope': 'read:user user:email', 'default_timeout': OAUTH_TIMEOUT},
    )
    OAUTH_PROVIDERS.add('github')

# --- DATABASE SETUP ---
# SQLite locally; set DATABASE_URL for PostgreSQL in production (see db.py)

def hash_api_key(api_key):
    # API keys are long random tokens, so a plain SHA-256 is enough for lookups
    return hashlib.sha256(api_key.encode()).hexdigest()

def init_db():
    conn = db.connect()
    c = conn

    # Users table. email_verified defaults to 1 so accounts from before verification existed keep
    # working; sign-ups insert 0 until the emailed link is opened.
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  email_verified INTEGER NOT NULL DEFAULT 1,
                  referral_code TEXT,
                  referred_by INTEGER,
                  avatar_url TEXT)'''))

    # API Keys table (api_key holds a SHA-256 hash; the raw key is only shown once at creation)
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS api_keys
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  key_name TEXT NOT NULL,
                  api_key TEXT UNIQUE NOT NULL,
                  key_prefix TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  is_active INTEGER DEFAULT 1,
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))

    # API Usage table
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS api_usage
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  api_key_id INTEGER NOT NULL,
                  endpoint TEXT NOT NULL,
                  status TEXT NOT NULL,
                  response_time REAL,
                  error_message TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (api_key_id) REFERENCES api_keys (id))'''))

    # Password Resets (OTP) Table
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS password_resets
                 (email TEXT PRIMARY KEY,
                  otp TEXT NOT NULL,
                  expires_at TIMESTAMP NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0)'''))

    # Failed Logins Table (for locking an account after repeated wrong passwords)
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS login_failures
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT NOT NULL,
                  failed_at TIMESTAMP NOT NULL)'''))
    c.execute('CREATE INDEX IF NOT EXISTS idx_login_failures_email ON login_failures (email)')

    # Social Login Table (links a Google/GitHub identity to a user)
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS oauth_accounts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  provider TEXT NOT NULL,
                  provider_user_id TEXT NOT NULL,
                  email TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE (provider, provider_user_id),
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))

    c.execute('CREATE INDEX IF NOT EXISTS idx_api_usage_key ON api_usage (api_key_id)')

    # API plan per user: requests allowed and used in the current period, plus top-up credits
    # that never expire (see billing.py)
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS api_subscriptions
                 (user_id INTEGER PRIMARY KEY,
                  plan_id TEXT NOT NULL,
                  requests_limit INTEGER NOT NULL,
                  requests_used INTEGER NOT NULL DEFAULT 0,
                  period_start TIMESTAMP NOT NULL,
                  period_end TIMESTAMP NOT NULL,
                  cycle TEXT NOT NULL DEFAULT 'monthly',
                  credits INTEGER NOT NULL DEFAULT 0,
                  warned_80 INTEGER NOT NULL DEFAULT 0,
                  reminder_sent INTEGER NOT NULL DEFAULT 0,
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))

    # Razorpay orders, auto-renewal charges and their outcome (amounts in paise / cents).
    # plan_id holds the product bought: 'pro', 'pro_yearly', 'topup', ...
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS payments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  plan_id TEXT NOT NULL,
                  country TEXT,
                  currency TEXT NOT NULL,
                  amount INTEGER NOT NULL,
                  razorpay_order_id TEXT UNIQUE NOT NULL,
                  razorpay_payment_id TEXT,
                  status TEXT NOT NULL DEFAULT 'created',
                  created_at TIMESTAMP NOT NULL,
                  paid_at TIMESTAMP,
                  coupon_code TEXT,
                  discount INTEGER NOT NULL DEFAULT 0,
                  refunded_amount INTEGER NOT NULL DEFAULT 0,
                  refund_id TEXT,
                  razorpay_subscription_id TEXT,
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))
    c.execute('CREATE INDEX IF NOT EXISTS idx_payments_user ON payments (user_id)')

    # Columns added since the first release (databases created by older versions get them here)
    for table, column, definition in [
        ('api_keys', 'key_prefix', 'TEXT'),
        ('password_resets', 'attempts', 'INTEGER NOT NULL DEFAULT 0'),
        ('users', 'email_verified', 'INTEGER NOT NULL DEFAULT 1'),
        ('users', 'referral_code', 'TEXT'),
        ('users', 'referred_by', 'INTEGER'),
        ('users', 'avatar_url', 'TEXT'),
        ('api_subscriptions', 'cycle', "TEXT NOT NULL DEFAULT 'monthly'"),
        ('api_subscriptions', 'credits', 'INTEGER NOT NULL DEFAULT 0'),
        ('api_subscriptions', 'warned_80', 'INTEGER NOT NULL DEFAULT 0'),
        ('api_subscriptions', 'reminder_sent', 'INTEGER NOT NULL DEFAULT 0'),
        ('payments', 'coupon_code', 'TEXT'),
        ('payments', 'discount', 'INTEGER NOT NULL DEFAULT 0'),
        ('payments', 'refunded_amount', 'INTEGER NOT NULL DEFAULT 0'),
        ('payments', 'refund_id', 'TEXT'),
        ('payments', 'razorpay_subscription_id', 'TEXT'),
    ]:
        db.ensure_column(c, table, column, definition)
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_code ON users (referral_code)')
    c.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_payment_id ON payments (razorpay_payment_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_api_usage_time ON api_usage (timestamp)')

    # Discount codes created in the admin panel (percent off; applies_to empty = everything)
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS coupons
                 (code TEXT PRIMARY KEY,
                  percent_off INTEGER NOT NULL,
                  max_redemptions INTEGER,
                  redeemed_count INTEGER NOT NULL DEFAULT 0,
                  expires_at TIMESTAMP,
                  active INTEGER NOT NULL DEFAULT 1,
                  applies_to TEXT,
                  created_at TIMESTAMP NOT NULL)'''))

    # Who referred whom; rewarded on the referred account's first purchase
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS referrals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  referrer_id INTEGER NOT NULL,
                  referred_id INTEGER UNIQUE NOT NULL,
                  status TEXT NOT NULL DEFAULT 'pending',
                  referrer_credits INTEGER NOT NULL DEFAULT 0,
                  referred_credits INTEGER NOT NULL DEFAULT 0,
                  created_at TIMESTAMP NOT NULL,
                  rewarded_at TIMESTAMP,
                  FOREIGN KEY (referrer_id) REFERENCES users (id),
                  FOREIGN KEY (referred_id) REFERENCES users (id))'''))
    c.execute('CREATE INDEX IF NOT EXISTS idx_referrals_referrer ON referrals (referrer_id)')

    # Invoices: one per paid purchase, with the seller's and buyer's details as they were when issued
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS invoices
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  payment_id INTEGER UNIQUE NOT NULL,
                  user_id INTEGER NOT NULL,
                  number TEXT UNIQUE NOT NULL,
                  fy TEXT NOT NULL,
                  seq INTEGER NOT NULL,
                  kind TEXT NOT NULL,
                  issued_at TIMESTAMP NOT NULL,
                  currency TEXT NOT NULL,
                  total INTEGER NOT NULL,
                  taxable INTEGER NOT NULL,
                  cgst INTEGER NOT NULL DEFAULT 0,
                  sgst INTEGER NOT NULL DEFAULT 0,
                  igst INTEGER NOT NULL DEFAULT 0,
                  place_of_supply TEXT,
                  sac TEXT,
                  description TEXT NOT NULL,
                  seller_name TEXT,
                  seller_address TEXT,
                  seller_gstin TEXT,
                  buyer_name TEXT,
                  buyer_email TEXT,
                  buyer_business TEXT,
                  buyer_gstin TEXT,
                  buyer_state TEXT,
                  buyer_address TEXT,
                  UNIQUE (fy, seq),
                  FOREIGN KEY (payment_id) REFERENCES payments (id),
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))
    c.execute('CREATE INDEX IF NOT EXISTS idx_invoices_user ON invoices (user_id)')
    # Last invoice number used in each financial year (see invoices.create_invoice)
    c.execute('''CREATE TABLE IF NOT EXISTS invoice_counters
                 (fy TEXT PRIMARY KEY,
                  last_seq INTEGER NOT NULL)''')

    # Details printed on invoices (business name, GSTIN, state for the place of supply)
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS billing_profiles
                 (user_id INTEGER PRIMARY KEY,
                  business_name TEXT,
                  gstin TEXT,
                  state_code TEXT,
                  address TEXT,
                  updated_at TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))

    # Audit trail of everything done in the admin panel
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS admin_actions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  admin_id INTEGER NOT NULL,
                  action TEXT NOT NULL,
                  target_user_id INTEGER,
                  details TEXT,
                  created_at TIMESTAMP NOT NULL)'''))

    # Auto-renewal: Razorpay plan objects (one per product and price) and customers' subscriptions
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS razorpay_plans
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  product_id TEXT NOT NULL,
                  currency TEXT NOT NULL,
                  amount INTEGER NOT NULL,
                  razorpay_plan_id TEXT NOT NULL,
                  UNIQUE (product_id, currency, amount))'''))
    c.execute(db.ddl('''CREATE TABLE IF NOT EXISTS autopay_subscriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  product_id TEXT NOT NULL,
                  country TEXT,
                  currency TEXT NOT NULL,
                  amount INTEGER NOT NULL,
                  razorpay_subscription_id TEXT UNIQUE NOT NULL,
                  razorpay_plan_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TIMESTAMP NOT NULL,
                  updated_at TIMESTAMP NOT NULL,
                  FOREIGN KEY (user_id) REFERENCES users (id))'''))
    c.execute('CREATE INDEX IF NOT EXISTS idx_autopay_user ON autopay_subscriptions (user_id)')

    # Replace any plaintext API keys with their hashes
    for row in c.execute('SELECT id, api_key FROM api_keys').fetchall():
        if row['api_key'].startswith('hmnz_'):
            c.execute('UPDATE api_keys SET api_key = ?, key_prefix = ? WHERE id = ?',
                      (hash_api_key(row['api_key']), row['api_key'][:12], row['id']))

    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()
logger.info('Humanizer engine: %s', 'spaCy' if humanizer.NLP is not None else 'basic fallback (spaCy not installed)')

# --- HELPER FUNCTIONS ---
utcnow = db.utcnow   # naive UTC, matching how the database stores timestamps

def hash_password(password):
    return generate_password_hash(password)

def is_legacy_password_hash(stored_hash):
    # Accounts created before the upgrade use unsalted SHA-256, which has no '$' separator
    return '$' not in stored_hash

def verify_password(stored_hash, password):
    if not stored_hash or stored_hash == OAUTH_ONLY_PASSWORD:
        return False
    if is_legacy_password_hash(stored_hash):
        return hmac.compare_digest(stored_hash, hashlib.sha256(password.encode()).hexdigest())
    return check_password_hash(stored_hash, password)

def generate_api_key():
    return 'hmnz_' + secrets.token_hex(24)

def get_json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}

def get_str(data, key, strip=True):
    value = data.get(key)
    if not isinstance(value, str):
        return ''
    return value.strip() if strip else value

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # If accessing via API, return 401 JSON
            if request.path.startswith('/api/'):
                return jsonify({"error": "Unauthorized"}), 401
            # If accessing via Browser, redirect to login
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def get_db():
    return db.connect()

def verify_api_key(api_key):
    conn = get_db()
    try:
        return conn.execute('''SELECT k.*, u.email_verified FROM api_keys k JOIN users u ON u.id = k.user_id
                               WHERE k.api_key = ? AND k.is_active = 1''', (hash_api_key(api_key),)).fetchone()
    finally:
        conn.close()

def log_api_usage(api_key_id, endpoint, status, response_time, error_message=None):
    log_api_usage_many([(api_key_id, endpoint, status, response_time, error_message)])

def log_api_usage_many(rows):
    """rows: (api_key_id, endpoint, status, response_time, error_message)"""
    conn = get_db()
    try:
        for row in rows:
            conn.execute('''INSERT INTO api_usage (api_key_id, endpoint, status, response_time, error_message)
                            VALUES (?, ?, ?, ?, ?)''', row)
        conn.commit()
    except Exception as e:
        logger.error('Usage logging failed: %s', type(e).__name__)
    finally:
        conn.close()

def error_location(e):
    """Where an exception happened, without its message (messages can quote user text)."""
    frames = traceback.extract_tb(e.__traceback__)
    return ' <- '.join(f'{os.path.basename(f.filename)}:{f.lineno} {f.name}' for f in reversed(frames[-4:]))

def send_otp_email(to_email, otp):
    # Sent inline (not queued) so the caller can tell the user if it failed
    return mailer.deliver(mailer.build(to_email, f'Your {BUSINESS_NAME} reset code', 'reset_code', otp=otp))

def get_user(user_id):
    conn = get_db()
    try:
        return conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    finally:
        conn.close()

# --- EMAIL VERIFICATION ---
# Sign-ups get a signed link (no database table needed). Unverified accounts can use the free
# editor, but need a verified address for API keys, the free API trial and purchases.

def verification_required():
    """REQUIRE_EMAIL_VERIFICATION=1/0; by default on whenever email sending is configured."""
    setting = os.environ.get('REQUIRE_EMAIL_VERIFICATION', '')
    return setting == '1' if setting in ('0', '1') else mailer.is_configured()

def needs_verification(user):
    return verification_required() and user is not None and not user['email_verified']

def verify_serializer():
    return URLSafeTimedSerializer(app.secret_key, salt='verify-email')

def send_verification_email(user):
    token = verify_serializer().dumps({'uid': user['id'], 'email': user['email']})
    return mailer.send(user['email'], f'Confirm your email for {BUSINESS_NAME}', 'verify_email',
                       username=user['username'], verify_url=f'{public_base_url()}/verify-email?token={token}')

def unverified_response():
    return jsonify({"error": "Verify your email address first. We sent you a link when you signed up; "
                             "you can send a new one from the dashboard.", "code": "email_unverified"}), 403

# --- BOT PROTECTION (CLOUDFLARE TURNSTILE) ---

def turnstile_enabled():
    return bool(TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY)

def turnstile_passed(token):
    """Check a Turnstile token with Cloudflare. Fails closed: no answer means no sign-up."""
    if not turnstile_enabled():
        return True
    if not token or len(token) > 2048:
        return False
    try:
        response = requests.post(TURNSTILE_VERIFY_URL, timeout=10, data={
            'secret': TURNSTILE_SECRET_KEY, 'response': token, 'remoteip': client_ip()})
        return bool(response.json().get('success'))
    except (requests.RequestException, ValueError) as e:
        logger.error('Turnstile check failed: %s', type(e).__name__)
        return False

# ==========================================
#   PRODUCTION SAFEGUARDS
# ==========================================

class RateLimiter:
    """In-memory sliding-window limiter. Exact with one gunicorn worker (the default config);
    with several workers each one keeps its own counts."""

    def __init__(self):
        self.enabled = os.environ.get('RATE_LIMITS', 'on') != 'off'
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, bucket, limit, window, cost=1):
        """Count a request (as `cost` requests); return the seconds to wait if the limit is exceeded, otherwise 0."""
        if not self.enabled:
            return 0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[bucket]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) + cost > limit:
                # Wait until enough earlier requests have left the window
                index = min(len(hits) - 1, max(0, len(hits) + cost - limit - 1))
                return max(1, math.ceil(hits[index] + window - now)) if hits else window
            hits.extend([now] * cost)
            if len(self._hits) > 10000:
                stale = [k for k, v in self._hits.items() if not v or v[-1] <= now - 86400]
                for k in stale:
                    del self._hits[k]
            return 0

limiter = RateLimiter()

def client_ip():
    return request.remote_addr or 'unknown'

JSON_PATHS = ('/api/', '/humanize', '/auth/', '/admin/api/', '/internal/')

def wants_json():
    return request.path.startswith(JSON_PATHS) or request.method != 'GET'

def error_response(status, message):
    if wants_json():
        response = jsonify({"success": False, "error": message, "message": message})
        response.status_code = status
        return response
    return render_template('error.html', code=status, message=message), status

def too_many_requests(retry_after):
    wait = f'{math.ceil(retry_after / 60)} minute(s)' if retry_after >= 60 else f'{retry_after} seconds'
    response = error_response(429, f'Too many requests. Please try again in {wait}.')
    response = response if isinstance(response, Response) else app.make_response(response)
    response.headers['Retry-After'] = str(retry_after)
    return response

def rate_limited(name):
    """Limit POST requests to a route per client IP."""
    limit, window = RATE_LIMITS[name]

    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if request.method == 'POST':
                retry_after = limiter.hit(f'{name}:{client_ip()}', limit, window)
                if retry_after:
                    return too_many_requests(retry_after)
            return f(*args, **kwargs)
        return wrapped
    return decorator

@app.before_request
def reject_cross_site_writes():
    # Browsers always send Origin on cross-site POSTs; API clients authenticate with X-API-Key instead
    if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and not request.headers.get('X-API-Key'):
        origin = request.headers.get('Origin')
        if origin and urlparse(origin).netloc != request.host:
            return error_response(403, 'Cross-site request blocked.')

# Razorpay Checkout loads scripts from several of its subdomains (checkout, cdn for fraud checks),
# opens its payment window in a frame, and reports back to its API
_last_housekeeping = float('-inf')
_housekeeping_lock = threading.Lock()

@app.before_request
def housekeeping():
    """Delete short-lived records on the schedule the privacy policy promises (at most every 10 minutes)."""
    global _last_housekeeping
    if time.monotonic() - _last_housekeeping < HOUSEKEEPING_INTERVAL or not _housekeeping_lock.acquire(blocking=False):
        return
    try:
        _last_housekeeping = time.monotonic()
        now = utcnow()
        conn = get_db()
        try:
            conn.execute('DELETE FROM login_failures WHERE failed_at <= ?', (now - LOGIN_LOCKOUT_WINDOW,))
            conn.execute('DELETE FROM password_resets WHERE expires_at <= ?', (now,))
            conn.execute("DELETE FROM payments WHERE status = 'created' AND created_at <= ?",
                         (now - ABANDONED_ORDER_RETENTION,))
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.error('Housekeeping failed: %s at %s', type(e).__name__, error_location(e))
    finally:
        _housekeeping_lock.release()

# Profile pictures from Google and GitHub sign-in are loaded straight from their image servers
AVATAR_HOSTS = ('*.googleusercontent.com', 'avatars.githubusercontent.com')

# Cloudflare Turnstile (sign-up bot check) loads its script and widget frame from challenges.cloudflare.com
CONTENT_SECURITY_POLICY = '; '.join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' https://*.razorpay.com https://challenges.cloudflare.com",
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
    "font-src 'self' https://fonts.gstatic.com",
    "img-src 'self' data: https://*.razorpay.com " + ' '.join(f'https://{host}' for host in AVATAR_HOSTS),
    "frame-src https://*.razorpay.com https://challenges.cloudflare.com",
    "connect-src 'self' https://*.razorpay.com",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "object-src 'none'",
])

@app.after_request
def add_security_headers(response):
    headers = response.headers
    headers.setdefault('Content-Security-Policy', CONTENT_SECURITY_POLICY)
    headers.setdefault('X-Content-Type-Options', 'nosniff')
    headers.setdefault('X-Frame-Options', 'DENY')
    headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    headers.setdefault('Permissions-Policy', 'camera=(), microphone=(), geolocation=(), '
                                             'payment=(self "https://api.razorpay.com" "https://checkout.razorpay.com")')
    # Netbanking and wallet payments open in a popup that must be able to report back
    headers.setdefault('Cross-Origin-Opener-Policy', 'same-origin-allow-popups')
    if request.is_secure:
        headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')

    # Pages are revalidated on every visit (they can show account details); API answers are never cached
    if request.path.startswith(JSON_PATHS) or request.path.startswith('/admin'):
        headers['Cache-Control'] = 'no-store'
    elif response.mimetype == 'text/html' and 'Cache-Control' not in headers:
        headers['Cache-Control'] = 'no-cache'
    return response

@app.errorhandler(404)
def not_found(e):
    return error_response(404, "We couldn't find that page.")

@app.errorhandler(405)
def method_not_allowed(e):
    return error_response(405, "That method isn't allowed here.")

@app.errorhandler(413)
def request_too_large(e):
    return error_response(413, 'Request too large (max 1 MB).')

@app.errorhandler(500)
def server_error(e):
    # Flask has already logged the traceback
    return error_response(500, 'Something went wrong on our side. Please try again.')

# ==========================================
#   SEO & TEMPLATE HELPERS
# ==========================================

_asset_versions = {}

def asset(filename):
    """URL of a static file with a content hash, so it can be cached for a year and still update."""
    version = _asset_versions.get(filename)
    if version is None:
        with open(os.path.join(app.static_folder, filename), 'rb') as f:
            version = hashlib.sha256(f.read()).hexdigest()[:10]
        _asset_versions[filename] = version
    return url_for('static', filename=filename, v=version)

def public_base_url():
    return SITE_URL or request.url_root.rstrip('/')

@app.context_processor
def template_globals():
    # Emails can be rendered outside a request (scripts, tests), where there's no URL to work from
    base = public_base_url() if has_request_context() else SITE_URL
    path = (request.path.rstrip('/') or '/') if has_request_context() else '/'
    return {
        'asset': asset,
        'site_url': base,
        'canonical_url': base + path,
        'turnstile_site_key': TURNSTILE_SITE_KEY if turnstile_enabled() else '',
        'is_admin': lambda: has_request_context() and admin.current_admin() is not None,
        'contact_email': CONTACT_EMAIL,
        'contact_phone': CONTACT_PHONE,
        'business_name': BUSINESS_NAME,
        'business_address': BUSINESS_ADDRESS,
        'grievance_officer': GRIEVANCE_OFFICER,
        'jurisdiction_city': JURISDICTION_CITY,
        'api_trial_requests': billing.trial_requests(),
        'api_plans': billing.PLANS,
        'current_year': utcnow().year,
    }

def visitor_country():
    """(ISO code, name) of the visitor's country from their IP, for regional pricing."""
    code, name = billing.lookup_country(client_ip())
    if not code and GEO_DEFAULT_COUNTRY:
        code = GEO_DEFAULT_COUNTRY
    return code, name

def visitor_pricing():
    return billing.pricing_for(*visitor_country())

def safe_next(target):
    """Only allow redirects to paths on this site."""
    return target if target and target.startswith('/') and not target.startswith('//') else None

@app.before_request
def remember_referral():
    # A referral link (?ref=CODE) is remembered until the visitor signs up
    code = billing.normalize_code(request.args.get('ref'))
    if request.method == 'GET' and code and 'user_id' not in session and re.fullmatch(r'[A-Z0-9]{4,16}', code):
        session['ref'] = code

# Pages listed in the sitemap: (path, template whose change date is used as lastmod)
SITEMAP_PAGES = [('/', 'landing.html'), ('/pricing', 'pricing.html'), ('/docs', 'docs.html'), ('/signup', 'signup.html'),
                 ('/privacy', 'privacy.html'), ('/terms', 'terms.html'), ('/refunds', 'refunds.html'),
                 ('/contact', 'contact.html')]
# Paths search engines shouldn't crawl (account pages, API endpoints, auth flows)
PRIVATE_PATHS = ['/editor', '/dashboard', '/api/', '/humanize', '/auth/', '/login/', '/logout', '/forgot-password',
                 '/verify-email', '/admin', '/internal/']

@app.route('/robots.txt')
def robots_txt():
    lines = ['User-agent: *', 'Allow: /'] + [f'Disallow: {path}' for path in PRIVATE_PATHS]
    lines += ['', f'Sitemap: {public_base_url()}/sitemap.xml']
    return Response('\n'.join(lines) + '\n', mimetype='text/plain')

@app.route('/sitemap.xml')
def sitemap_xml():
    entries = []
    for path, template in SITEMAP_PAGES:
        modified = datetime.fromtimestamp(os.path.getmtime(os.path.join(BASE_DIR, 'templates', template)), timezone.utc)
        entries.append(f'  <url>\n    <loc>{xml_escape(public_base_url() + path)}</loc>\n'
                       f'    <lastmod>{modified:%Y-%m-%d}</lastmod>\n  </url>')
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + '\n'.join(entries) + '\n</urlset>\n')
    return Response(xml, mimetype='application/xml')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.static_folder, 'img'), 'favicon.ico', max_age=7 * 86400)

@app.route('/site.webmanifest')
def web_manifest():
    manifest = {
        'name': 'AI Humanizer',
        'short_name': 'Humanizer',
        'description': 'Rewrite AI-generated drafts in a natural, human voice.',
        'start_url': '/editor',
        'display': 'standalone',
        'background_color': '#FAF9F6',
        'theme_color': '#FAF9F6',
        'icons': [
            {'src': asset('img/icon-192.png'), 'sizes': '192x192', 'type': 'image/png'},
            {'src': asset('img/icon-512.png'), 'sizes': '512x512', 'type': 'image/png'},
        ],
    }
    return Response(json.dumps(manifest), mimetype='application/manifest+json')

# ==========================================
#   PAGE ROUTES
# ==========================================

@app.route('/')
def landing():
    return render_template('landing.html', pricing=visitor_pricing())

@app.route('/pricing')
def pricing_page():
    return render_template('pricing.html', pricing=visitor_pricing())

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/refunds')
def refunds():
    return render_template('refunds.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/editor')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
@rate_limited('login')
def login():
    if request.method == 'POST':
        data = get_json_body()
        email = get_str(data, 'email')
        password = get_str(data, 'password', strip=False)

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400

        throttle_key = email.lower()
        now = utcnow()
        window_start = now - LOGIN_LOCKOUT_WINDOW

        conn = get_db()
        try:
            # Slow down password guessing: lock the account for a while after repeated failures
            recent_failures = conn.execute(
                'SELECT failed_at FROM login_failures WHERE email = ? AND failed_at > ? ORDER BY failed_at',
                (throttle_key, window_start)).fetchall()
            if len(recent_failures) >= MAX_LOGIN_FAILURES:
                unlock_at = db.to_datetime(recent_failures[-MAX_LOGIN_FAILURES]['failed_at']) + LOGIN_LOCKOUT_WINDOW
                minutes = max(1, math.ceil((unlock_at - now).total_seconds() / 60))
                return jsonify({"success": False, "message": f"Too many failed attempts. Try again in {minutes} minute(s)."}), 429

            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if not user or not verify_password(user['password'], password):
                conn.execute('INSERT INTO login_failures (email, failed_at) VALUES (?, ?)', (throttle_key, now))
                conn.execute('DELETE FROM login_failures WHERE failed_at <= ?', (window_start,))
                conn.commit()
                return jsonify({"success": False, "message": "Invalid credentials"}), 401

            conn.execute('DELETE FROM login_failures WHERE email = ?', (throttle_key,))

            # Upgrade old SHA-256 hashes now that we have the plaintext password
            if is_legacy_password_hash(user['password']):
                conn.execute('UPDATE users SET password = ? WHERE id = ?', (hash_password(password), user['id']))
            conn.commit()
        finally:
            conn.close()

        session.clear()
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['avatar'] = user['avatar_url']   # set if the account was ever used with Google/GitHub
        return jsonify({"success": True, "message": "Login successful"})

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
@rate_limited('signup')
def signup():
    if request.method == 'POST':
        data = get_json_body()
        username = get_str(data, 'username')
        email = get_str(data, 'email')
        raw_password = get_str(data, 'password', strip=False)

        if not username or not email or not raw_password:
            return jsonify({"success": False, "message": "All fields are required"}), 400
        if len(username) > 50:
            return jsonify({"success": False, "message": "Username must be 50 characters or fewer"}), 400
        if '@' not in email or len(email) > 254:
            return jsonify({"success": False, "message": "Please enter a valid email address"}), 400
        if len(raw_password) < MIN_PASSWORD_LENGTH:
            return jsonify({"success": False, "message": f"Password must be at least {MIN_PASSWORD_LENGTH} characters"}), 400
        if not turnstile_passed(get_str(data, 'turnstile_token')):
            return jsonify({"success": False, "message": "We couldn't confirm you're human. Please complete the check and try again.",
                            "turnstile_failed": True}), 400

        password = hash_password(raw_password)
        conn = get_db()
        try:
            user_id = conn.execute('INSERT INTO users (username, email, password, email_verified) VALUES (?, ?, ?, 0) RETURNING id',
                                   (username, email, password)).fetchone()['id']
            billing.record_referral(conn, user_id, session.get('ref'), utcnow())
            conn.commit()
        except db.IntegrityError:
            return jsonify({"success": False, "message": "Username or email already exists"}), 400
        finally:
            conn.close()

        session.pop('ref', None)
        verify_sent = mailer.is_configured() and send_verification_email({'id': user_id, 'email': email, 'username': username})
        message = (f"Account created. We sent a confirmation link to {email}." if verify_sent
                   else "Account created successfully")
        return jsonify({"success": True, "message": message, "verify_sent": bool(verify_sent)})

    return render_template('signup.html')

@app.route('/verify-email')
def verify_email():
    """Open the link from the confirmation email."""
    try:
        data = verify_serializer().loads(request.args.get('token', ''), max_age=int(VERIFY_LINK_LIFETIME.total_seconds()))
    except SignatureExpired:
        return render_template('verify_email.html', state='expired'), 400
    except BadSignature:
        return render_template('verify_email.html', state='invalid'), 400

    conn = get_db()
    try:
        # The link only counts for the address it was sent to
        updated = conn.execute('UPDATE users SET email_verified = 1 WHERE id = ? AND lower(email) = lower(?)',
                               (data.get('uid'), data.get('email', ''))).rowcount
        conn.commit()
    finally:
        conn.close()
    if not updated:
        return render_template('verify_email.html', state='invalid'), 400
    return render_template('verify_email.html', state='verified', signed_in='user_id' in session)

@app.route('/api/account', methods=['GET'])
@login_required
def account_info():
    user = get_user(session['user_id'])
    if user is None:
        session.clear()
        return jsonify({"error": "Unauthorized"}), 401
    conn = get_db()
    try:
        referral = billing.referral_summary(conn, user['id'])
    finally:
        conn.close()
    referral['url'] = f"{public_base_url()}/signup?ref={referral['code']}" if referral['code'] else None
    return jsonify({
        "username": user['username'], "email": user['email'],
        "email_verified": bool(user['email_verified']), "verification_required": verification_required(),
        "referral": referral,
    })

@app.route('/api/account/resend-verification', methods=['POST'])
@login_required
def resend_verification():
    retry_after = limiter.hit(f"verify_email:{session['user_id']}", *RATE_LIMITS['verify_email'])
    if retry_after:
        return too_many_requests(retry_after)
    user = get_user(session['user_id'])
    if user is None or user['email_verified']:
        return jsonify({"success": True, "message": "Your email address is already verified."})
    if not mailer.is_configured() or not send_verification_email(user):
        return jsonify({"success": False, "error": "We couldn't send the email. Please try again later."}), 503
    return jsonify({"success": True, "message": f"We sent a new link to {user['email']}."})

@app.route('/forgot-password')
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/auth/check')
def check_auth():
    if 'user_id' in session:
        return jsonify({"authenticated": True, "username": session.get('username')})
    else:
        return jsonify({"authenticated": False}), 401

@app.route('/healthz')
def healthz():
    return 'ok'

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# ==========================================
#   AUTH ROUTES: SOCIAL LOGIN (GOOGLE & GITHUB)
# ==========================================

class OAuthProfileError(Exception):
    """A provider answered, but the account can't be used (message is shown to the user)."""

def safe_avatar_url(url):
    """A profile picture address from Google or GitHub, if it's https on their image servers (the only
    hosts the CSP allows); anything else is ignored and the initial is shown instead."""
    if not isinstance(url, str) or len(url) > 500:
        return None
    parsed = urlparse(url)
    host = parsed.hostname or ''
    if parsed.scheme != 'https' or not any(host == h or (h.startswith('*.') and host.endswith(h[1:])) for h in AVATAR_HOSTS):
        return None
    return url

def get_oauth_profile(provider):
    """Finish the OAuth exchange and return (provider_user_id, verified_email, display_name, avatar_url)."""
    client = oauth.create_client(provider)
    token = client.authorize_access_token()

    if provider == 'google':
        info = token.get('userinfo') or client.userinfo(token=token)
        if not info.get('email') or not info.get('email_verified'):
            raise OAuthProfileError("Your Google account's email address isn't verified.")
        return (str(info['sub']), info['email'], info.get('name') or info['email'].split('@')[0],
                safe_avatar_url(info.get('picture')))

    user_resp = client.get('user', token=token)
    user_resp.raise_for_status()
    profile = user_resp.json()
    emails_resp = client.get('user/emails', token=token)
    emails_resp.raise_for_status()
    email = next((e['email'] for e in emails_resp.json() if e.get('primary') and e.get('verified')), None)
    if not email:
        raise OAuthProfileError('Add a verified primary email to your GitHub account, then try again.')
    return str(profile['id']), email, profile.get('login') or email.split('@')[0], safe_avatar_url(profile.get('avatar_url'))

def unique_username(conn, name):
    base = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('_.-')[:40] or 'user'
    candidate, n = base, 1
    while conn.execute('SELECT 1 FROM users WHERE username = ?', (candidate,)).fetchone():
        n += 1
        candidate = f'{base}{n}'
    return candidate

def find_or_create_oauth_user(provider, provider_user_id, email, display_name, referral_code=None, avatar_url=None):
    conn = get_db()

    def with_avatar(user):
        # Keep the picture from the latest Google/GitHub sign-in (people change them)
        if avatar_url and user['avatar_url'] != avatar_url:
            conn.execute('UPDATE users SET avatar_url = ? WHERE id = ?', (avatar_url, user['id']))
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE id = ?', (user['id'],)).fetchone()
        return user

    try:
        user = conn.execute('''SELECT u.* FROM oauth_accounts o JOIN users u ON u.id = o.user_id
                               WHERE o.provider = ? AND o.provider_user_id = ?''',
                            (provider, provider_user_id)).fetchone()
        if user:
            return with_avatar(user)

        # Providers only give us verified emails, so it's safe to link to a matching account
        user = conn.execute('SELECT * FROM users WHERE lower(email) = lower(?)', (email,)).fetchone()
        if user and not user['email_verified']:
            # Someone may have signed up with this address without owning it; their password stops working
            conn.execute('UPDATE users SET email_verified = 1, password = ? WHERE id = ?', (OAUTH_ONLY_PASSWORD, user['id']))
            user = conn.execute('SELECT * FROM users WHERE id = ?', (user['id'],)).fetchone()
        if not user:
            new_id = conn.execute('INSERT INTO users (username, email, password, email_verified) VALUES (?, ?, ?, 1) RETURNING id',
                                  (unique_username(conn, display_name), email, OAUTH_ONLY_PASSWORD)).fetchone()['id']
            billing.record_referral(conn, new_id, referral_code, utcnow())
            user = conn.execute('SELECT * FROM users WHERE id = ?', (new_id,)).fetchone()

        conn.execute('INSERT INTO oauth_accounts (user_id, provider, provider_user_id, email) VALUES (?, ?, ?, ?)',
                     (user['id'], provider, provider_user_id, email))
        conn.commit()
        return with_avatar(user)
    finally:
        conn.close()

@app.route('/login/<any(google, github):provider>')
def oauth_login(provider):
    if provider not in OAUTH_PROVIDERS:
        logger.warning('%s sign-in is not configured: set %s_CLIENT_ID and %s_CLIENT_SECRET',
                       OAUTH_NAMES[provider], provider.upper(), provider.upper())
        flash(f"Sign in with {OAUTH_NAMES[provider]} isn't set up yet. Use your email and password for now.", 'error')
        return redirect('/login')

    session['login_next'] = safe_next(request.args.get('next'))
    redirect_uri = url_for('oauth_callback', provider=provider, _external=True)
    return oauth.create_client(provider).authorize_redirect(redirect_uri)

@app.route('/login/<any(google, github):provider>/callback')
def oauth_callback(provider):
    name = OAUTH_NAMES[provider]
    if provider not in OAUTH_PROVIDERS:
        return redirect('/login')
    if request.args.get('error'):
        flash(f'{name} sign-in was cancelled.', 'error')
        return redirect('/login')

    try:
        provider_user_id, email, display_name, avatar_url = get_oauth_profile(provider)
    except OAuthProfileError as e:
        flash(str(e), 'error')
        return redirect('/login')
    except Exception as e:
        logger.error('%s sign-in failed: %s at %s', name, type(e).__name__, error_location(e))
        flash(f"{name} sign-in didn't complete. Please try again.", 'error')
        return redirect('/login')

    user = find_or_create_oauth_user(provider, provider_user_id, email, display_name, session.get('ref'), avatar_url)
    next_page = safe_next(session.get('login_next'))
    session.clear()
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['avatar'] = user['avatar_url']
    return redirect(next_page or '/editor')

# ==========================================
#   API ROUTES: FORGOT PASSWORD (OTP)
# ==========================================

@app.route('/api/send-otp', methods=['POST'])
@rate_limited('send_otp')
def send_otp():
    email = get_str(get_json_body(), 'email')
    if not email:
        return jsonify({"success": False, "message": "Email is required."}), 400

    # Same reply whether or not the account exists, so emails can't be probed
    generic_reply = {"success": True, "message": "If this email exists, an OTP has been sent."}

    conn = get_db()
    try:
        # 1. Check if user exists
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

        if not user:
            return jsonify(generic_reply)

        # 2. Don't resend within the cooldown (stops email flooding and attempt resets)
        now = utcnow()
        record = conn.execute('SELECT expires_at FROM password_resets WHERE email = ?', (email,)).fetchone()
        if record and now - (db.to_datetime(record['expires_at']) - OTP_LIFETIME) < OTP_RESEND_COOLDOWN:
            return jsonify(generic_reply)

        # 3. Generate OTP with a cryptographically secure generator
        otp = f"{secrets.randbelow(1000000):06d}"

        # 4. Save OTP (replacing any earlier code for this email)
        conn.execute('''INSERT INTO password_resets (email, otp, expires_at, attempts) VALUES (?, ?, ?, 0)
                        ON CONFLICT (email) DO UPDATE
                        SET otp = excluded.otp, expires_at = excluded.expires_at, attempts = 0''',
                     (email, otp, now + OTP_LIFETIME))
        conn.commit()

        # 5. Send Email
        if send_otp_email(email, otp):
            return jsonify(generic_reply)

        conn.execute('DELETE FROM password_resets WHERE email = ?', (email,))
        conn.commit()
        return jsonify({"success": False, "message": "Failed to send email. Please try again later."}), 500
    finally:
        conn.close()

@app.route('/api/reset-password', methods=['POST'])
@rate_limited('reset_password')
def reset_password_logic():
    data = get_json_body()
    email = get_str(data, 'email')
    otp = get_str(data, 'otp')
    new_password = get_str(data, 'new_password', strip=False)

    if not email or not otp or not new_password:
        return jsonify({"success": False, "message": "All fields are required."}), 400
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return jsonify({"success": False, "message": f"Password must be at least {MIN_PASSWORD_LENGTH} characters."}), 400

    conn = get_db()

    try:
        # 1. Verify OTP
        record = conn.execute('SELECT * FROM password_resets WHERE email = ?', (email,)).fetchone()

        if not record:
            return jsonify({"success": False, "message": "Invalid or expired code. Request a new one."}), 400

        if utcnow() > db.to_datetime(record['expires_at']):
            conn.execute('DELETE FROM password_resets WHERE email = ?', (email,))
            conn.commit()
            return jsonify({"success": False, "message": "OTP has expired."}), 400

        if not hmac.compare_digest(otp, record['otp']):
            # Limit guesses so the 6-digit code can't be brute-forced
            attempts = record['attempts'] + 1
            if attempts >= MAX_OTP_ATTEMPTS:
                conn.execute('DELETE FROM password_resets WHERE email = ?', (email,))
                conn.commit()
                return jsonify({"success": False, "message": "Too many wrong attempts. Request a new code."}), 400
            conn.execute('UPDATE password_resets SET attempts = ? WHERE email = ?', (attempts, email))
            conn.commit()
            return jsonify({"success": False, "message": "Invalid OTP."}), 400

        # 2. Update Password (the code reached this inbox, so the address is verified too)
        conn.execute('UPDATE users SET password = ?, email_verified = 1 WHERE email = ?', (hash_password(new_password), email))

        # 3. Delete OTP (Prevents reuse)
        conn.execute('DELETE FROM password_resets WHERE email = ?', (email,))

        # 4. Lift any login lockout so the owner can sign in right away
        conn.execute('DELETE FROM login_failures WHERE email = ?', (email.lower(),))

        # Commit everything
        conn.commit()

        return jsonify({"success": True, "message": "Password reset successfully!"})

    except Exception as e:
        logger.error('Password reset failed: %s at %s', type(e).__name__, error_location(e))
        return jsonify({"success": False, "message": "Server error."}), 500

    finally:
        conn.close()

# ==========================================
#   API ROUTES: DASHBOARD & USAGE
# ==========================================

@app.route('/api/keys', methods=['GET'])
@login_required
def get_api_keys():
    user_id = session['user_id']
    conn = get_db()
    try:
        keys = conn.execute('SELECT * FROM api_keys WHERE user_id = ? AND is_active = 1 ORDER BY created_at DESC',
                            (user_id,)).fetchall()
    finally:
        conn.close()

    keys_list = []
    for key in keys:
        keys_list.append({
            "id": key['id'],
            "name": key['key_name'],
            "key_preview": (key['key_prefix'] or 'hmnz_') + '…',
            "created_at": db.timestamp_str(key['created_at'])
        })
    return jsonify({"keys": keys_list})

@app.route('/api/keys', methods=['POST'])
@login_required
def create_api_key_endpoint():
    key_name = get_str(get_json_body(), 'name')[:50] or 'My API Key'
    user_id = session['user_id']
    if needs_verification(get_user(user_id)):
        return unverified_response()
    api_key = generate_api_key()

    conn = get_db()
    try:
        conn.execute('INSERT INTO api_keys (user_id, key_name, api_key, key_prefix) VALUES (?, ?, ?, ?)',
                     (user_id, key_name, hash_api_key(api_key), api_key[:12]))
        conn.commit()
    finally:
        conn.close()

    # The raw key is returned only here; the database keeps just its hash
    return jsonify({
        "success": True,
        "key": {
            "name": key_name,
            "api_key": api_key,
            "created_at": utcnow().strftime("%Y-%m-%d")
        }
    })

@app.route('/api/keys/<int:key_id>', methods=['DELETE'])
@login_required
def delete_api_key_endpoint(key_id):
    user_id = session['user_id']
    conn = get_db()
    try:
        # Deactivate rather than delete, so past usage still counts in the dashboard stats
        cursor = conn.execute('UPDATE api_keys SET is_active = 0 WHERE id = ? AND user_id = ? AND is_active = 1',
                              (key_id, user_id))
        conn.commit()
    finally:
        conn.close()
    if cursor.rowcount == 0:
        return jsonify({"success": False, "error": "Key not found"}), 404
    return jsonify({"success": True})

@app.route('/api/usage', methods=['GET'])
@login_required
def get_dashboard_stats():
    user_id = session['user_id']
    conn = get_db()
    try:
        stats = conn.execute('''
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN u.status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                AVG(u.response_time) as avg_latency
            FROM api_usage u
            JOIN api_keys k ON u.api_key_id = k.id
            WHERE k.user_id = ?
        ''', (user_id,)).fetchone()

        logs_data = conn.execute('''
            SELECT u.timestamp, u.endpoint, u.status, u.response_time
            FROM api_usage u
            JOIN api_keys k ON u.api_key_id = k.id
            WHERE k.user_id = ?
            ORDER BY u.timestamp DESC LIMIT 10
        ''', (user_id,)).fetchall()
    finally:
        conn.close()

    total_req = stats['total_requests'] or 0
    avg_lat = round(float(stats['avg_latency']), 2) if stats['avg_latency'] else 0
    success_rate = f"{round(stats['successful_requests'] / total_req * 100, 1):g}%" if total_req else "N/A"

    return jsonify({
        "stats": {
            "total_requests": total_req,
            "success_rate": success_rate,
            "avg_latency_ms": avg_lat
        },
        "logs": [
            {
                "timestamp": db.timestamp_str(log['timestamp']),
                "endpoint": log['endpoint'],
                "status_code": 200 if log['status'] == 'success' else 500,
                "latency_ms": round(log['response_time'] or 0, 2)
            } for log in logs_data
        ]
    })

@app.route('/api/usage/daily', methods=['GET'])
@login_required
def daily_usage():
    """API requests per day (UTC) and per key, for the dashboard chart."""
    try:
        days = min(90, max(7, int(request.args.get('days', 30))))
    except ValueError:
        days = 30
    today = utcnow().date()
    first_day = today - timedelta(days=days - 1)
    conn = get_db()
    try:
        rows = conn.execute('''
            SELECT DATE(u.timestamp) AS day, k.id AS key_id, k.key_name, k.is_active, COUNT(*) AS requests,
                   SUM(CASE WHEN u.status = 'success' THEN 0 ELSE 1 END) AS errors
            FROM api_usage u JOIN api_keys k ON u.api_key_id = k.id
            WHERE k.user_id = ? AND u.timestamp >= ?
            GROUP BY DATE(u.timestamp), k.id, k.key_name, k.is_active
        ''', (session['user_id'], datetime.combine(first_day, datetime.min.time()))).fetchall()
    finally:
        conn.close()

    labels = [(first_day + timedelta(days=i)).isoformat() for i in range(days)]
    position = {day: i for i, day in enumerate(labels)}
    totals, errors, keys = [0] * days, [0] * days, {}
    for row in rows:
        i = position.get(str(row['day'])[:10])
        if i is None:
            continue
        key = keys.setdefault(row['key_id'], {"id": row['key_id'], "name": row['key_name'],
                                               "active": bool(row['is_active']), "counts": [0] * days, "total": 0})
        key['counts'][i] += row['requests']
        key['total'] += row['requests']
        totals[i] += row['requests']
        errors[i] += row['errors'] or 0
    return jsonify({"days": labels, "totals": totals, "errors": errors,
                    "keys": sorted(keys.values(), key=lambda k: -k['total'])})

# ==========================================
#   API ROUTES: BILLING (RAZORPAY)
# ==========================================

@app.route('/api/billing/status', methods=['GET'])
@login_required
def billing_status():
    return jsonify(billing.account_status(session['user_id'], utcnow()))

def requested_product(data):
    # 'product' since yearly plans and top-ups; older pages sent 'plan'
    return get_str(data, 'product') or get_str(data, 'plan')

@app.route('/api/billing/quote', methods=['POST'])
@login_required
@rate_limited('coupon')
def billing_quote():
    """Price after a coupon, so the pricing page can show the discount before checkout."""
    data = get_json_body()
    country, _ = visitor_country()
    try:
        quote = billing.quote(requested_product(data), country, get_str(data, 'coupon'), session['user_id'], utcnow())
    except billing.PaymentError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(quote)

@app.route('/api/billing/coupon', methods=['POST'])
@login_required
@rate_limited('coupon')
def billing_coupon():
    """Check a coupon code; the pricing page shows the discount on the products it applies to."""
    try:
        return jsonify(billing.coupon_info(get_str(get_json_body(), 'coupon'), session['user_id'], utcnow()))
    except billing.CouponError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/billing/order', methods=['POST'])
@login_required
@rate_limited('billing')
def create_billing_order():
    if not billing.is_enabled():
        return jsonify({"error": "Payments aren't available yet. Please check back soon."}), 503
    data = get_json_body()
    product_id = requested_product(data)
    if product_id not in billing.PRODUCTS:
        return jsonify({"error": "Unknown plan."}), 400
    user = get_user(session['user_id'])
    if needs_verification(user):
        return unverified_response()

    country, _ = visitor_country()
    try:
        order = billing.create_order(session['user_id'], product_id, country, utcnow(), get_str(data, 'coupon'))
    except billing.CouponError as e:
        return jsonify({"error": str(e)}), 400
    except billing.PaymentError as e:
        return jsonify({"error": str(e)}), 502

    if order.get('free'):
        return jsonify({**order, "success": True, **billing.account_status(session['user_id'], utcnow())})
    order['description'] = billing.PRODUCTS[product_id]['description']
    order['prefill'] = {'name': user['username'], 'email': user['email']} if user else {}
    return jsonify(order)

@app.route('/api/billing/subscribe', methods=['POST'])
@login_required
@rate_limited('billing')
def create_billing_subscription():
    """Start auto-renewal for a plan (Razorpay Subscriptions: card or UPI AutoPay)."""
    if not billing.autopay_enabled():
        return jsonify({"error": "Auto-renewal isn't available yet. You can still buy a plan for 30 days or a year."}), 503
    data = get_json_body()
    if get_str(data, 'coupon'):
        return jsonify({"error": "Coupons can't be combined with auto-renewal. Turn off auto-renew to use one."}), 400
    user = get_user(session['user_id'])
    if needs_verification(user):
        return unverified_response()

    country, _ = visitor_country()
    try:
        subscription = billing.create_autopay(session['user_id'], requested_product(data), country, utcnow())
    except billing.PaymentError as e:
        return jsonify({"error": str(e)}), 400
    subscription['description'] = billing.PRODUCTS[subscription['product']]['description'] + ', renews automatically'
    subscription['prefill'] = {'name': user['username'], 'email': user['email']}
    return jsonify(subscription)

@app.route('/api/billing/verify-subscription', methods=['POST'])
@login_required
def verify_billing_subscription():
    data = get_json_body()
    payment_id = get_str(data, 'razorpay_payment_id')
    subscription_id = get_str(data, 'razorpay_subscription_id')
    signature = get_str(data, 'razorpay_signature')
    if not (payment_id and subscription_id and signature) or \
            not billing.verify_subscription_signature(payment_id, subscription_id, signature):
        return jsonify({"error": "We couldn't verify this payment."}), 400

    conn = get_db()
    try:
        sub = conn.execute('SELECT user_id FROM autopay_subscriptions WHERE razorpay_subscription_id = ?',
                           (subscription_id,)).fetchone()
    finally:
        conn.close()
    if not sub or sub['user_id'] != session['user_id']:
        return jsonify({"error": "Subscription not found."}), 404

    try:
        payment = billing.razorpay_request('GET', f'/payments/{payment_id}')
    except billing.PaymentError as e:
        return jsonify({"error": str(e)}), 502
    if payment.get('status') != 'captured':
        # The subscription.charged webhook grants the plan once the first charge settles
        return jsonify({"error": "Auto-renewal is set up. Your plan will activate as soon as the first payment completes."}), 409
    billing.record_subscription_charge(subscription_id, payment, utcnow())
    return jsonify({"success": True, **billing.account_status(session['user_id'], utcnow())})

@app.route('/api/billing/autopay/cancel', methods=['POST'])
@login_required
def cancel_billing_autopay():
    try:
        billing.cancel_autopay(session['user_id'], utcnow())
    except billing.PaymentError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"success": True, **billing.account_status(session['user_id'], utcnow())})

@app.route('/api/billing/profile', methods=['GET', 'POST'])
@login_required
def billing_profile():
    """Business name, GSTIN and state printed on invoices."""
    conn = get_db()
    try:
        if request.method == 'POST':
            data = get_json_body()
            error = invoices.save_profile(conn, session['user_id'], get_str(data, 'business_name'), get_str(data, 'gstin'),
                                          get_str(data, 'state_code'), get_str(data, 'address'), utcnow())
            if error:
                return jsonify({"error": error}), 400
            conn.commit()
        profile = invoices.get_profile(conn, session['user_id'])
    finally:
        conn.close()
    return jsonify({"profile": profile, "states": invoices.STATES, "gst_invoices": bool(invoices.seller_gstin())})

@app.route('/api/billing/invoices/<int:invoice_id>.pdf', methods=['GET'])
@login_required
def download_invoice(invoice_id):
    owner = None if admin.current_admin() else session['user_id']
    result = invoices.pdf_for(invoice_id, owner)
    if result is None:
        return jsonify({"error": "Invoice not found."}), 404
    filename, pdf = result
    return Response(pdf, mimetype='application/pdf',
                    headers={'Content-Disposition': f'inline; filename="{filename}"'})

def after_payment(payment_id, extras):
    """Runs once per completed purchase: issue the invoice, email the receipt, thank the referrer."""
    if not has_app_context():
        # Payments fulfilled from a script or worker still render their emails
        with app.app_context():
            return after_payment(payment_id, extras)
    conn = get_db()
    try:
        payment = conn.execute('''SELECT p.*, u.username, u.email FROM payments p JOIN users u ON u.id = p.user_id
                                  WHERE p.id = ?''', (payment_id,)).fetchone()
    finally:
        conn.close()
    product = billing.PRODUCTS.get(payment['plan_id'], {'name': payment['plan_id'], 'description': payment['plan_id']})
    invoice = invoices.create_invoice(payment_id, product['description'])
    attachments = []
    if invoice is not None:
        try:
            attachments.append(invoices.pdf_for(invoice['id']) + ('application/pdf',))
        except Exception as e:
            logger.error('Invoice PDF for payment %s failed: %s at %s', payment_id, type(e).__name__, error_location(e))
    mailer.send(payment['email'], f'Your {BUSINESS_NAME} receipt', 'receipt', attachments=attachments,
                username=payment['username'], product_name=product['name'],
                amount=billing.format_price(payment['amount'], payment['currency']),
                paid_date=db.to_datetime(payment['paid_at']).strftime('%d %b %Y'),
                invoice_number=invoice['number'] if invoice else None, dashboard_url=f'{public_base_url()}/dashboard')

    reward = extras.get('referral_reward')
    if reward:
        referrer = get_user(reward['referrer_id'])
        if referrer:
            mailer.send(referrer['email'], f"You earned {reward['credits']:,} free API requests", 'referral_reward',
                        username=referrer['username'], credits=f"{reward['credits']:,}", dashboard_url=f'{public_base_url()}/dashboard')

billing.on_paid.append(after_payment)

@app.route('/api/billing/verify', methods=['POST'])
@login_required
def verify_billing_payment():
    data = get_json_body()
    order_id = get_str(data, 'razorpay_order_id')
    payment_id = get_str(data, 'razorpay_payment_id')
    signature = get_str(data, 'razorpay_signature')
    if not (order_id and payment_id and signature) or not billing.verify_payment_signature(order_id, payment_id, signature):
        return jsonify({"error": "We couldn't verify this payment."}), 400

    conn = get_db()
    try:
        order = conn.execute('SELECT user_id, amount, currency FROM payments WHERE razorpay_order_id = ?',
                             (order_id,)).fetchone()
    finally:
        conn.close()
    if not order or order['user_id'] != session['user_id']:
        return jsonify({"error": "Order not found."}), 404

    try:
        payment = billing.capture_if_needed(payment_id, order['amount'], order['currency'])
    except billing.PaymentError as e:
        return jsonify({"error": str(e)}), 502
    if payment.get('status') != 'captured' or payment.get('order_id') != order_id:
        return jsonify({"error": "The payment hasn't completed yet. If you were charged, your plan will activate shortly."}), 409

    billing.fulfil_payment(order_id, payment_id, utcnow(), payment.get('amount'), payment.get('currency'))
    return jsonify({"success": True, **billing.account_status(session['user_id'], utcnow())})

@app.route('/api/billing/webhook', methods=['POST'])
def razorpay_webhook():
    # Razorpay confirms paid orders here too, in case the browser closed before /verify ran
    body = request.get_data()
    if not billing.verify_webhook_signature(body, request.headers.get('X-Razorpay-Signature')):
        return jsonify({"error": "Invalid signature"}), 400
    try:
        event = json.loads(body or b'{}')
    except ValueError:
        return jsonify({"error": "Invalid JSON"}), 400
    name = event.get('event', '')
    payload = event.get('payload', {})
    payment = payload.get('payment', {}).get('entity', {})
    subscription = payload.get('subscription', {}).get('entity', {})

    if name in ('order.paid', 'payment.captured'):
        # Auto-renewal charges also fire these, for orders Razorpay created itself; subscription.charged handles them
        if payment.get('order_id') and payment.get('id') and payment.get('status') == 'captured' and not payment.get('invoice_id'):
            billing.fulfil_payment(payment['order_id'], payment['id'], utcnow(), payment.get('amount'), payment.get('currency'))
    elif name == 'subscription.charged':
        if subscription.get('id') and payment.get('id') and payment.get('status') == 'captured':
            billing.record_subscription_charge(subscription['id'], payment, utcnow())
    elif name.startswith('subscription.') and subscription.get('id') and subscription.get('status'):
        # activated, authenticated, pending (charge failing, retrying), halted (retries exhausted), cancelled, completed
        billing.update_autopay_status(subscription['id'], subscription['status'], utcnow())
    return jsonify({"status": "ok"})

def plan_required_response(subscription):
    """402 telling an API client why the request wasn't allowed and where to fix it."""
    pricing_url = f'{public_base_url()}/pricing'
    if subscription is None or subscription['plan_id'] in (billing.TRIAL_PLAN, billing.NO_PLAN):
        message = f'The REST API needs a paid plan or a top-up. Choose one at {pricing_url}'
    elif db.to_datetime(subscription['period_end']) <= utcnow():
        message = f'Your API plan has expired. Renew it or buy a top-up at {pricing_url}'
    else:
        message = f'Your API plan has used all its requests. Buy a top-up or upgrade at {pricing_url}'
    return jsonify({"error": message}), 402

def send_quota_warning(user_id, row):
    user = get_user(user_id)
    if user is None:
        return
    mailer.send(user['email'], f"You've used 80% of your {BUSINESS_NAME} API requests", 'quota_warning',
                username=user['username'], plan_name=billing.plan_label(row['plan_id']),
                used=f"{row['requests_used']:,}", limit=f"{row['requests_limit']:,}", pricing_url=f'{public_base_url()}/pricing')

def quota_headers(response, charge):
    row = charge.row
    plan_left = max(0, row['requests_limit'] - row['requests_used']) if db.to_datetime(row['period_end']) > utcnow() else 0
    response.headers['X-Quota-Limit'] = str(row['requests_limit'])
    response.headers['X-Quota-Remaining'] = str(plan_left)
    response.headers['X-Credits-Remaining'] = str(row['credits'])
    return response

# ==========================================
#   SCHEDULED TASKS
# ==========================================

@app.route('/internal/tasks', methods=['POST'])
def scheduled_tasks():
    """Daily jobs, called by .github/workflows/daily-tasks.yml with the CRON_SECRET bearer token."""
    supplied = request.headers.get('Authorization', '').removeprefix('Bearer ').strip()
    if not CRON_SECRET or not hmac.compare_digest(supplied, CRON_SECRET):
        return error_response(404, "We couldn't find that page.")

    sent = 0
    for row in billing.claim_expiry_reminders(utcnow()):
        sent += bool(mailer.send(row['email'], f'Your {BUSINESS_NAME} API plan ends soon', 'plan_expiring',
                                 username=row['username'], plan_name=billing.plan_label(row['plan_id']),
                                 end_date=db.to_datetime(row['period_end']).strftime('%d %b %Y'),
                                 remaining=f"{max(0, row['requests_limit'] - row['requests_used']):,}",
                                 pricing_url=f'{public_base_url()}/pricing'))
    logger.info('Scheduled tasks: %s expiry reminder(s) sent', sent)
    return jsonify({"success": True, "reminders_sent": sent})

# ==========================================
#   CORE HUMANIZER ENDPOINT
# ==========================================

@app.route('/humanize', methods=['POST'])
def humanize():
    api_key = request.headers.get('X-API-Key')
    start_time = time.perf_counter()
    api_key_id = None
    api_user_id = None

    if api_key:
        key_data = verify_api_key(api_key)
        if not key_data:
            return jsonify({"error": "Invalid or inactive API key"}), 401
        if needs_verification(key_data):
            return unverified_response()
        api_key_id = key_data['id']
        api_user_id = key_data['user_id']
    else:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401

    # Rewriting is CPU-heavy, so each key / signed-in user gets a fair share
    identity = f'key:{api_key_id}' if api_key_id else f'user:{session["user_id"]}'
    retry_after = limiter.hit(f'humanize:{identity}', *RATE_LIMITS['humanize'])
    if retry_after:
        return too_many_requests(retry_after)

    data = get_json_body()
    text = data.get('text', '')
    tone = data.get('tone', 'standard')
    deep_mode = data.get('deep_mode', False)

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "No text provided"}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text too long (max {MAX_TEXT_LENGTH} characters)"}), 400
    if tone not in TONES:
        return jsonify({"error": f"Invalid tone. Use one of: {', '.join(TONES)}"}), 400
    if not isinstance(deep_mode, bool):
        return jsonify({"error": "deep_mode must be true or false"}), 400

    # The REST API is paid: each valid API request uses one request from the account's plan
    # (then its top-up credits). The web editor (signed-in session, no API key) stays free.
    charge = None
    if api_user_id:
        charge = billing.consume_request(api_user_id, utcnow())
        if not charge.allowed:
            return plan_required_response(charge.row)
        if charge.warn:
            send_quota_warning(api_user_id, charge.row)

    try:
        humanized_text = perform_nuclear_chaos(text, tone, deep_mode)
        metrics = calculate_diff_metrics(text, humanized_text)

        response_time = (time.perf_counter() - start_time) * 1000

        if api_key_id:
            log_api_usage(api_key_id, '/humanize', 'success', response_time)

        response = jsonify({
            "original": text,
            "humanized": humanized_text,
            "diff_html": metrics['diff_html'],
            "longest_unchanged": metrics['longest_unchanged'],
            "structural_changes": f"{metrics['structural_score']}%",
            "changed_words": metrics['changed_words_count'],
            "new_words": metrics['new_words_count'],
            "moved_words": metrics['moved_words_count'],
            "restructured_sentences": metrics['restructured_sentences'],
        })
        return quota_headers(response, charge) if charge else response

    except Exception as e:
        # Log only the error type and location: messages can quote the user's text, which is never stored
        logger.error('Humanize failed: %s at %s', type(e).__name__, error_location(e))
        if charge:
            billing.refund_request(api_user_id, charge)
        if api_key_id:
            response_time = (time.perf_counter() - start_time) * 1000
            log_api_usage(api_key_id, '/humanize', 'error', response_time, type(e).__name__)
        return jsonify({"error": "Processing failed."}), 500

@app.route('/humanize/batch', methods=['POST'])
def humanize_batch():
    """Rewrite up to MAX_BATCH_ITEMS texts in one call (API keys only). Each item costs one request;
    items that fail on our side are given back."""
    start_time = time.perf_counter()
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({"error": "The batch endpoint needs an API key in the X-API-Key header"}), 401
    key_data = verify_api_key(api_key)
    if not key_data:
        return jsonify({"error": "Invalid or inactive API key"}), 401
    if needs_verification(key_data):
        return unverified_response()

    data = get_json_body()
    default_tone = data.get('tone', 'standard')
    default_deep = data.get('deep_mode', False)
    raw_items = data.get('items')
    if not isinstance(raw_items, list) or not raw_items:
        return jsonify({"error": 'Send "items": a list of texts, or of objects with "text" (and optional "tone", "deep_mode", "id")'}), 400
    if len(raw_items) > MAX_BATCH_ITEMS:
        return jsonify({"error": f"Too many items (max {MAX_BATCH_ITEMS} per request)"}), 400

    items = []
    for i, item in enumerate(raw_items):
        if isinstance(item, str):
            item = {'text': item}
        if not isinstance(item, dict):
            return jsonify({"error": f"Item {i}: must be a string or an object with \"text\""}), 400
        text, tone, deep_mode = item.get('text', ''), item.get('tone', default_tone), item.get('deep_mode', default_deep)
        item_id = item.get('id')
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": f"Item {i}: no text provided"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"Item {i}: text too long (max {MAX_TEXT_LENGTH} characters)"}), 400
        if tone not in TONES:
            return jsonify({"error": f"Item {i}: invalid tone. Use one of: {', '.join(TONES)}"}), 400
        if not isinstance(deep_mode, bool):
            return jsonify({"error": f"Item {i}: deep_mode must be true or false"}), 400
        if item_id is not None and not (isinstance(item_id, (str, int)) and not isinstance(item_id, bool) and len(str(item_id)) <= 100):
            return jsonify({"error": f"Item {i}: id must be a string or number (max 100 characters)"}), 400
        items.append({'text': text, 'tone': tone, 'deep_mode': deep_mode, 'id': item_id})
    if sum(len(item['text']) for item in items) > MAX_BATCH_CHARS:
        return jsonify({"error": f"Batch too large (max {MAX_BATCH_CHARS} characters in total)"}), 400

    # Each item counts against the key's rate limit, like a separate request
    retry_after = limiter.hit(f"humanize:key:{key_data['id']}", *RATE_LIMITS['humanize'], cost=len(items))
    if retry_after:
        return too_many_requests(retry_after)

    charge = billing.consume_request(key_data['user_id'], utcnow(), len(items))
    if not charge.allowed:
        return plan_required_response(charge.row)
    if charge.warn:
        send_quota_warning(key_data['user_id'], charge.row)

    results, usage_rows, failed = [], [], 0
    for i, item in enumerate(items):
        item_start = time.perf_counter()
        result = {"index": i} if item['id'] is None else {"index": i, "id": item['id']}
        try:
            humanized_text = perform_nuclear_chaos(item['text'], item['tone'], item['deep_mode'])
            metrics = calculate_diff_metrics(item['text'], humanized_text)
            result.update({"humanized": humanized_text, "longest_unchanged": metrics['longest_unchanged'],
                           "structural_changes": f"{metrics['structural_score']}%", "changed_words": metrics['changed_words_count'],
                           "new_words": metrics['new_words_count'], "moved_words": metrics['moved_words_count'],
                           "restructured_sentences": metrics['restructured_sentences']})
            usage_rows.append((key_data['id'], '/humanize/batch', 'success', (time.perf_counter() - item_start) * 1000, None))
        except Exception as e:
            logger.error('Batch item failed: %s at %s', type(e).__name__, error_location(e))
            result["error"] = "Processing failed."
            failed += 1
            usage_rows.append((key_data['id'], '/humanize/batch', 'error', (time.perf_counter() - item_start) * 1000, type(e).__name__))
        results.append(result)

    log_api_usage_many(usage_rows)
    if failed:
        billing.refund_request(key_data['user_id'], charge, failed)
    response = jsonify({"results": results, "count": len(items), "succeeded": len(items) - failed,
                        "processing_ms": round((time.perf_counter() - start_time) * 1000, 2)})
    return quota_headers(response, charge)

# Admin panel at /admin (only for verified accounts listed in ADMIN_EMAILS)
app.register_blueprint(admin.bp)

if __name__ == '__main__':
    # Running on 5001 to avoid conflicts. Debug mode only when FLASK_DEBUG=1 (never in production)
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5001)
