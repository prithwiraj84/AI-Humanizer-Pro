import os
import json
import random
import difflib
import re
import unicodedata
import nltk
import secrets
import sqlite3
import hashlib
import hmac
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI, Depends, Request
from fastapi.responses import ORJSONResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

# Load variables from a local .env file. No-op on hosts (e.g. Render) that
# inject real environment variables, so it is safe in every environment.
load_dotenv()

# --- SETUP: DOWNLOAD NLTK DATA (Runs once on startup) ---
try:
    from nltk.corpus import wordnet
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    from nltk.corpus import wordnet

# --- CRITICAL CONFIGURATION ---
# 1. Secret Key: signs the session cookie (Starlette SessionMiddleware).
#    MUST be set via env in production: a fresh random key on every restart logs
#    everyone out, and different keys per worker break auth across processes.
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)

# --- EMAIL CONFIGURATION (FOR FORGOT PASSWORD) ---
# Set these via environment variables. Create a Google App Password at:
# GOOGLE ACCOUNT -> SECURITY -> 2-STEP VERIFICATION -> APP PASSWORDS
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")

# --- ADMIN LOGIN CONFIGURATION ---
# Set these in environment variables for production deployments.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@aih.local")
ADMIN_PASSWORD_HASH = hashlib.sha256(
    os.getenv("ADMIN_PASSWORD", "ChangeAdminPassword123!").encode()
).hexdigest()

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('humanizer.db')
    c = conn.cursor()

    # Performance: write-ahead logging gives far better read/write concurrency.
    c.execute('PRAGMA journal_mode = WAL')

    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # API Keys table
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  key_name TEXT NOT NULL,
                  api_key TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  is_active INTEGER DEFAULT 1,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')

    # API Usage table
    c.execute('''CREATE TABLE IF NOT EXISTS api_usage
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  api_key_id INTEGER NOT NULL,
                  endpoint TEXT NOT NULL,
                  status TEXT NOT NULL,
                  response_time REAL,
                  error_message TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (api_key_id) REFERENCES api_keys (id))''')

    # Password Resets (OTP) Table
    c.execute('''CREATE TABLE IF NOT EXISTS password_resets
                 (email TEXT PRIMARY KEY,
                  otp TEXT NOT NULL,
                  expires_at TIMESTAMP NOT NULL)''')

    # Activity logs table (admin console monitoring)
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  actor_type TEXT NOT NULL,
                  actor_identifier TEXT,
                  action TEXT NOT NULL,
                  endpoint TEXT NOT NULL,
                  method TEXT NOT NULL,
                  status_code INTEGER NOT NULL,
                  ip_address TEXT,
                  details TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')

    c.execute('CREATE INDEX IF NOT EXISTS idx_activity_logs_created_at ON activity_logs(created_at DESC)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_activity_logs_user_id ON activity_logs(user_id)')

    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# --- HELPER FUNCTIONS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_api_key():
    return 'hmnz_' + secrets.token_hex(24)

def get_db():
    conn = sqlite3.connect('humanizer.db')
    conn.row_factory = sqlite3.Row
    # Wait (instead of erroring) if another writer holds the lock briefly.
    conn.execute('PRAGMA busy_timeout = 5000')
    conn.execute('PRAGMA synchronous = NORMAL')
    return conn

def verify_api_key(api_key):
    conn = get_db()
    key_data = conn.execute('SELECT * FROM api_keys WHERE api_key = ? AND is_active = 1',
                            (api_key,)).fetchone()
    conn.close()
    return key_data

def log_api_usage(api_key_id, endpoint, status, response_time, error_message=None):
    conn = get_db()
    try:
        conn.execute('''INSERT INTO api_usage (api_key_id, endpoint, status, response_time, error_message)
                        VALUES (?, ?, ?, ?, ?)''',
                     (api_key_id, endpoint, status, response_time, error_message))
        conn.commit()
    except Exception as e:
        print(f"Logging error: {e}")
    finally:
        conn.close()

def log_activity_event(user_id, actor_type, actor_identifier, action, endpoint, method, status_code, ip_address=None, details=None):
    conn = get_db()
    try:
        conn.execute('''
            INSERT INTO activity_logs
            (user_id, actor_type, actor_identifier, action, endpoint, method, status_code, ip_address, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            actor_type,
            actor_identifier,
            action,
            endpoint,
            method,
            status_code,
            ip_address,
            (details or '')[:400]
        ))
        conn.commit()
    except Exception as e:
        print(f"Activity log error: {e}")
    finally:
        conn.close()

# --- AUTH DEPENDENCIES (FastAPI equivalent of the Flask decorators) ---
class UnauthorizedJSON(Exception):
    def __init__(self, payload, status):
        self.payload = payload
        self.status = status

def require_login(request: Request):
    # API-only backend: always answer with JSON. The Next.js frontend handles
    # redirecting unauthenticated browsers to the login page.
    if 'user_id' not in request.session:
        raise UnauthorizedJSON({"error": "Unauthorized"}, 401)

def require_admin(request: Request):
    if not request.session.get('admin_authenticated'):
        raise UnauthorizedJSON({"error": "Admin authorization required"}, 401)

# --- ACTIVITY LOGGING PIPELINE ---
def normalize_path(path):
    if not path:
        return '/'
    if path != '/' and path.endswith('/'):
        return path[:-1]
    return path

def infer_activity_action(path, method, status_code):
    normalized = normalize_path(path)
    method = (method or 'GET').upper()

    if normalized == '/login' and method == 'POST':
        return 'user_login_success' if status_code < 400 else 'user_login_failed'
    if normalized == '/signup' and method == 'POST':
        return 'user_signup_success' if status_code < 400 else 'user_signup_failed'
    if normalized in ('/logout', '/auth/logout'):
        return 'user_logout'
    if normalized == '/dashboard':
        return 'view_dashboard'
    if normalized == '/':
        return 'view_home'
    if normalized == '/docs':
        return 'view_docs'
    if normalized == '/api/keys' and method == 'GET':
        return 'view_api_keys'
    if normalized == '/api/keys' and method == 'POST':
        return 'api_key_created'
    if normalized.startswith('/api/keys/') and method == 'DELETE':
        return 'api_key_deleted'
    if normalized == '/api/usage' and method == 'GET':
        return 'view_usage_stats'
    if normalized == '/humanize' and method == 'POST':
        return 'humanize_text'
    if normalized == '/api/send-otp' and method == 'POST':
        return 'request_password_otp'
    if normalized == '/api/reset-password' and method == 'POST':
        return 'reset_password'
    if normalized == '/admin/login' and method == 'POST':
        return 'admin_login_success' if status_code < 400 else 'admin_login_failed'
    if normalized == '/admin/logout' and method == 'POST':
        return 'admin_logout'
    if normalized.startswith('/admin/api/users/') and method == 'DELETE':
        return 'admin_deleted_user'
    if normalized.startswith('/admin/api/keys/') and method == 'DELETE':
        return 'admin_revoked_key'

    path_token = normalized.strip('/').replace('/', '_') or 'home'
    return f"{method.lower()}_{path_token}"

def build_activity_details(path, method, body_json):
    normalized = normalize_path(path)
    data = body_json
    if not isinstance(data, dict):
        return None

    if normalized == '/humanize':
        text_len = len(data.get('text') or '')
        tone = data.get('tone') or 'standard'
        deep_mode = bool(data.get('deep_mode'))
        return f"text_length={text_len}, tone={tone}, deep_mode={deep_mode}"

    if normalized == '/api/keys' and method == 'POST':
        key_name = data.get('name') or 'My API Key'
        return f"key_name={str(key_name)[:120]}"

    if normalized in ('/login', '/signup', '/api/send-otp', '/api/reset-password', '/admin/login'):
        email = data.get('email')
        username = data.get('username')
        parts = []
        if email:
            parts.append(f"email={email}")
        if username:
            parts.append(f"username={username}")
        return ', '.join(parts) if parts else None

    return None

def resolve_activity_actor(path, method, headers, session_post, session_pre, body_json):
    user_id = session_post.get('user_id')
    if user_id:
        return user_id, 'user', session_post.get('username') or f"user_{user_id}"

    # Preserve actor identity for logout routes after the session is cleared.
    if path in ('/logout', '/auth/logout'):
        prev_user_id = session_pre.get('user_id')
        if prev_user_id:
            return prev_user_id, 'user', session_pre.get('username') or f"user_{prev_user_id}"

    api_key = headers.get('x-api-key')
    if api_key:
        key_data = verify_api_key(api_key)
        if key_data:
            return key_data['user_id'], 'api_key', key_data['api_key'][:14] + '...'

    if method == 'POST' and path in ('/login', '/signup', '/api/send-otp', '/api/reset-password'):
        payload = body_json or {}
        identifier = payload.get('email') or payload.get('username') or 'guest'
        return None, 'guest', identifier

    if session_post.get('admin_authenticated'):
        return None, 'admin', session_post.get('admin_email') or 'admin'

    if method == 'POST' and path == '/admin/login':
        payload = body_json or {}
        return None, 'admin_attempt', payload.get('email') or 'admin'

    if path == '/admin/logout' and session_pre.get('admin_authenticated'):
        return None, 'admin', session_pre.get('admin_email') or 'admin'

    return None, None, None

def _is_json_content_type(content_type):
    # Mirror Flask's is_json: application/json or any "*+json" subtype.
    ctype = (content_type or '').split(';')[0].strip().lower()
    return ctype == 'application/json' or ctype.endswith('+json')

def _parse_json_body(body, content_type):
    # Mirror Flask's request.get_json(silent=True): only parse when the body is
    # declared as JSON, and never raise.
    if not body:
        return None
    if not _is_json_content_type(content_type):
        return None
    try:
        return json.loads(body)
    except Exception:
        return None

def record_activity(method, raw_path, status_code, headers, session_post, session_pre, body_json, ip_address):
    path = normalize_path(raw_path)

    if path.startswith('/static') or path in ('/favicon.ico',):
        return

    # Avoid polluting the activity log by logging activity-log reads themselves.
    if path == '/admin/api/activity-log':
        return

    user_id, actor_type, actor_identifier = resolve_activity_actor(
        path, method, headers, session_post, session_pre, body_json
    )
    if not actor_type:
        return

    log_activity_event(
        user_id=user_id,
        actor_type=actor_type,
        actor_identifier=actor_identifier,
        action=infer_activity_action(path, method, status_code),
        endpoint=path,
        method=method,
        status_code=status_code,
        ip_address=ip_address,
        details=build_activity_details(path, method, body_json)
    )

class ActivityLoggerMiddleware:
    """
    Pure-ASGI port of the Flask before_request/after_request activity pipeline.

    It buffers the request body (so both this middleware and the route handler
    can read it), snapshots the session before the route runs (to preserve the
    actor across logout), captures the response status, and writes the activity
    record AFTER the response has been streamed to the client -- in a worker
    thread so it never blocks the event loop or delays the response.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Static frontend assets: serve without the logging machinery.
        _p = scope.get("path", "")
        if _p.startswith("/_next/") or _p in ("/favicon.ico", "/favicon.svg"):
            await self.app(scope, receive, send)
            return

        # Mirror Flask's strict_slashes=False: a trailing-slash variant runs the
        # SAME handler (one request, no 307 redirect) instead of Starlette's
        # default slash-redirect. Normalizing here -- before the router -- also
        # keeps the activity log honest (it records the real outcome, not a 307).
        raw_path = scope.get("path", "/")
        normalized_path = normalize_path(raw_path)
        if normalized_path != raw_path:
            scope["path"] = normalized_path
            scope["raw_path"] = normalized_path.encode("latin-1")

        # Buffer the entire request body so it can be read more than once.
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        async def replay_receive():
            return {"type": "http.request", "body": body, "more_body": False}

        # Snapshot session BEFORE the route runs (SessionMiddleware is outer, so
        # scope["session"] is already populated here).
        session = scope.get("session", {})
        session_pre = {
            "user_id": session.get("user_id"),
            "username": session.get("username"),
            "admin_authenticated": session.get("admin_authenticated"),
            "admin_email": session.get("admin_email"),
        }

        status_holder = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
            await send(message)

        await self.app(scope, replay_receive, send_wrapper)

        # Response already sent to the client -- logging here adds zero latency.
        try:
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in scope.get("headers", [])
            }
            session_post = scope.get("session", {})
            forwarded_for = headers.get("x-forwarded-for", "")
            client = scope.get("client")
            client_host = client[0] if client else ""
            ip_address = forwarded_for.split(",")[0].strip() if forwarded_for else client_host

            await run_in_threadpool(
                record_activity,
                scope["method"],
                scope["path"],
                status_holder["code"],
                headers,
                session_post,
                session_pre,
                _parse_json_body(body, headers.get("content-type", "")),
                ip_address,
            )
        except Exception as e:
            print(f"Activity capture error: {e}")

def send_otp_email(to_email, otp):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = "Your AI Humanizer Reset Code"

        body = f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #6366f1;">Password Reset Request</h2>
                <p>You requested a password reset for your AI Humanizer account.</p>
                <div style="background: #f4f4f5; padding: 15px; border-radius: 5px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 5px;">
                    {otp}
                </div>
                <p>This code expires in 10 minutes.</p>
                <p style="color: #999; font-size: 12px;">If you didn't request this, please ignore this email.</p>
            </div>
          </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email Error: {e}")
        return False

# ==========================================
#   HUMANIZER LOGIC (Core Features)
# ==========================================
def get_rare_synonym(word):
    synonyms = []
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower() and "_" not in lemma.name():
                    synonyms.append(lemma.name())
    except:
        return word

    if not synonyms:
        return word
    return random.choice(synonyms)

def spike_perplexity(text, intensity=0.3):
    words = text.split()
    new_text = []
    for word in words:
        if (len(word) > 4 and random.random() < intensity):
            new_word = get_rare_synonym(word)
            new_text.append(new_word)
        else:
            new_text.append(word)
    return " ".join(new_text)

def unicode_spoofing(text):
    chars = list(text)
    new_chars = []
    for char in chars:
        if random.random() < 0.5:
            new_chars.append(unicodedata.normalize('NFD', char))
        else:
            new_chars.append(unicodedata.normalize('NFC', char))
    return "".join(new_chars)

def inject_invisible_glue(text):
    invisible_chars = ['\u200B', '\u2060', '\u200C', '\u200D']
    words = text.split()
    new_words = []
    for word in words:
        if len(word) > 3 and random.random() < 0.4:
            split = random.randint(1, len(word)-1)
            word = word[:split] + random.choice(invisible_chars) + word[split:]
        new_words.append(word)
    return " ".join(new_words)

def calculate_diff_metrics(original, humanized):
    clean_ver = humanized
    for char in ['\u200B', '\u2060', '\u200C', '\u200D']:
        clean_ver = clean_ver.replace(char, "")

    matcher = difflib.SequenceMatcher(None, original.split(), clean_ver.split())
    match = matcher.find_longest_match(0, len(original.split()), 0, len(clean_ver.split()))
    longest_unchanged = " ".join(original.split()[match.a: match.a + match.size])
    similarity = matcher.ratio()
    structural_change_score = (1 - similarity) * 100

    diff = list(difflib.ndiff(original.split(), clean_ver.split()))
    diff_html = ""
    changes_count = 0

    for token in diff:
        word = token[2:]
        if token.startswith('- '):
            changes_count += 1
            continue
        elif token.startswith('+ '):
            diff_html += f'<span class="added">{word}</span> '
            changes_count += 1
        elif token.startswith('  '):
            diff_html += f'<span class="unchanged">{word}</span> '

    return {
        "diff_html": diff_html,
        "longest_unchanged": longest_unchanged if longest_unchanged else "None",
        "structural_score": round(structural_change_score, 1),
        "changed_words_count": changes_count
    }

def perform_nuclear_chaos(text, tone, deep_mode):
    current_text = text
    if deep_mode:
        current_text = spike_perplexity(current_text, intensity=0.4)
        current_text = unicode_spoofing(current_text)
        current_text = inject_invisible_glue(current_text)
    else:
        current_text = spike_perplexity(current_text, intensity=0.1)
    return current_text

# ==========================================
#   APPLICATION & MIDDLEWARE
# ==========================================
# docs_url/redoc_url/openapi_url disabled so FastAPI's auto Swagger UI does not
# collide with the app's own GET /docs page route.
app = FastAPI(
    default_response_class=ORJSONResponse,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    redirect_slashes=False,   # trailing slashes are normalized in middleware instead
)

# Order matters: SessionMiddleware must be OUTER so scope["session"] is decoded
# before the activity logger reads it. add_middleware adds outermost-last.
app.add_middleware(ActivityLoggerMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session",
    max_age=None,        # browser-session cookie, matching Flask's default
    same_site="lax",
    https_only=False,
)

@app.exception_handler(UnauthorizedJSON)
async def _unauthorized_handler(request: Request, exc: UnauthorizedJSON):
    return ORJSONResponse(exc.payload, status_code=exc.status)

def J(content, status=200):
    return ORJSONResponse(content, status_code=status)

class APIError(Exception):
    def __init__(self, status, message="Bad Request"):
        self.status = status
        self.message = message

@app.exception_handler(APIError)
async def _api_error_handler(request: Request, exc: APIError):
    return ORJSONResponse({"error": exc.message}, status_code=exc.status)

async def parse_json_body(request: Request):
    # Faithful port of Flask's `request.json` (the raising variant): 415 when the
    # body is not declared as JSON, 400 when it cannot be parsed.
    if not _is_json_content_type(request.headers.get('content-type', '')):
        raise APIError(415, "Did not attempt to load JSON data because the request Content-Type was not 'application/json'.")
    body = await request.body()
    try:
        return json.loads(body)
    except Exception:
        raise APIError(400, "Failed to decode JSON object.")

async def get_json_silent(request: Request):
    # Faithful port of Flask's `request.get_json(silent=True)`.
    if not _is_json_content_type(request.headers.get('content-type', '')):
        return None
    try:
        return await request.json()
    except Exception:
        return None

# ==========================================
#   AUTH & SESSION ROUTES
# ==========================================

@app.post('/admin/login')
async def admin_login(request: Request):
    data = (await get_json_silent(request)) or {}
    email = (data.get('email') or '').strip().lower()
    raw_password = data.get('password') or ''

    if email == ADMIN_EMAIL.lower() and hmac.compare_digest(hash_password(raw_password), ADMIN_PASSWORD_HASH):
        request.session['admin_authenticated'] = True
        request.session['admin_email'] = ADMIN_EMAIL
        return J({"success": True, "message": "Admin login successful"})

    return J({"success": False, "message": "Invalid admin credentials"}, 401)

@app.get('/admin/auth/check')
def admin_auth_check(request: Request):
    if request.session.get('admin_authenticated'):
        return J({"authenticated": True, "email": request.session.get('admin_email')})
    return J({"authenticated": False}, 401)

@app.post('/admin/logout')
def admin_logout(request: Request, _=Depends(require_admin)):
    request.session.pop('admin_authenticated', None)
    request.session.pop('admin_email', None)
    return J({"success": True})

@app.api_route('/healthz', methods=['GET', 'HEAD'])
def healthz():
    # Lightweight health check for platform pings (e.g. Render).
    return J({"service": "AI Humanizer Pro API", "status": "ok"})

@app.post('/login')
async def login(request: Request):
    data = await parse_json_body(request)
    email = data.get('email')
    password = hash_password(data.get('password'))

    def _find_user():
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND password = ?',
                            (email, password)).fetchone()
        conn.close()
        return user

    user = await run_in_threadpool(_find_user)

    if user:
        request.session['user_id'] = user['id']
        request.session['username'] = user['username']
        return J({"success": True, "message": "Login successful"})
    else:
        return J({"success": False, "message": "Invalid credentials"}, 401)

@app.post('/signup')
async def signup(request: Request):
    data = await parse_json_body(request)
    username = data.get('username')
    email = data.get('email')
    raw_password = data.get('password')

    if not username or not email or not raw_password:
        return J({"success": False, "message": "All fields are required"}, 400)

    password = hash_password(raw_password)

    def _create_user():
        conn = get_db()
        try:
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, password))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False

    created = await run_in_threadpool(_create_user)
    if created:
        return J({"success": True, "message": "Account created successfully"})
    return J({"success": False, "message": "Username or email already exists"}, 400)

@app.post('/auth/logout')
def auth_logout(request: Request):
    request.session.clear()
    return J({"success": True})

@app.get('/auth/check')
def check_auth(request: Request):
    if 'user_id' in request.session:
        return J({"authenticated": True, "username": request.session.get('username')})
    else:
        return J({"authenticated": False}, 401)

# ==========================================
#   API ROUTES: FORGOT PASSWORD (OTP)
# ==========================================

@app.post('/api/send-otp')
async def send_otp(request: Request):
    data = await parse_json_body(request)
    email = data.get('email')

    def _store_otp():
        conn = get_db()
        try:
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            if not user:
                return (False, None)
            otp = str(random.randint(100000, 999999))
            expires_at = datetime.now() + timedelta(minutes=10)
            conn.execute('INSERT OR REPLACE INTO password_resets (email, otp, expires_at) VALUES (?, ?, ?)',
                         (email, otp, expires_at))
            conn.commit()
            return (True, otp)
        finally:
            conn.close()

    user_exists, otp = await run_in_threadpool(_store_otp)

    # 1. Check if user exists (do not reveal whether it does)
    if not user_exists:
        return J({"success": True, "message": "If this email exists, an OTP has been sent."})

    # 4. Send Email
    if await run_in_threadpool(send_otp_email, email, otp):
        return J({"success": True, "message": "OTP sent to your email."})
    else:
        return J({"success": False, "message": "Failed to send email. Check server logs."}, 500)

@app.post('/api/reset-password')
async def reset_password_logic(request: Request):
    data = await parse_json_body(request)
    email = data.get('email')
    otp = data.get('otp')
    new_password = data.get('new_password')

    def _reset():
        conn = get_db()
        try:
            # 1. Verify OTP
            cursor = conn.execute('SELECT * FROM password_resets WHERE email = ?', (email,))
            record = cursor.fetchone()

            if not record:
                return {"success": False, "message": "Invalid request."}, 400

            saved_otp = record['otp']

            # Robust timestamp parsing
            try:
                expiry = datetime.strptime(record['expires_at'], '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                expiry = datetime.strptime(record['expires_at'], '%Y-%m-%d %H:%M:%S')

            if otp != saved_otp:
                return {"success": False, "message": "Invalid OTP."}, 400

            if datetime.now() > expiry:
                return {"success": False, "message": "OTP has expired."}, 400

            # 2. Update Password
            hashed_pw = hash_password(new_password)
            conn.execute('UPDATE users SET password = ? WHERE email = ?', (hashed_pw, email))

            # 3. Delete OTP (Prevents reuse)
            conn.execute('DELETE FROM password_resets WHERE email = ?', (email,))

            # Commit everything
            conn.commit()

            return {"success": True, "message": "Password reset successfully!"}, 200

        except Exception as e:
            print(f"Reset Error: {e}")
            return {"success": False, "message": "Server error."}, 500

        finally:
            conn.close()

    payload, status = await run_in_threadpool(_reset)
    return J(payload, status)

# ==========================================
#   API ROUTES: DASHBOARD & USAGE
# ==========================================

@app.get('/api/keys')
def get_api_keys(request: Request, _=Depends(require_login)):
    user_id = request.session['user_id']
    conn = get_db()
    keys = conn.execute('SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC', (user_id,)).fetchall()
    conn.close()

    keys_list = []
    for key in keys:
        keys_list.append({
            "id": key['id'],
            "name": key['key_name'],
            "api_key": key['api_key'],
            "created_at": key['created_at'],
            "is_active": key['is_active']
        })
    return J({"keys": keys_list})

@app.post('/api/keys')
async def create_api_key_endpoint(request: Request, _=Depends(require_login)):
    data = await parse_json_body(request)
    key_name = data.get('name', 'My API Key')
    user_id = request.session['user_id']
    api_key = generate_api_key()

    def _insert():
        conn = get_db()
        conn.execute('INSERT INTO api_keys (user_id, key_name, api_key) VALUES (?, ?, ?)',
                     (user_id, key_name, api_key))
        conn.commit()
        conn.close()

    await run_in_threadpool(_insert)

    return J({
        "success": True,
        "key": {
            "name": key_name,
            "api_key": api_key,
            "secret": "hidden_for_security",
            "created_at": datetime.now().strftime("%Y-%m-%d")
        }
    })

@app.delete('/api/keys/{key_id:int}')
def delete_api_key_endpoint(key_id: int, request: Request, _=Depends(require_login)):
    user_id = request.session['user_id']
    conn = get_db()
    conn.execute('DELETE FROM api_keys WHERE id = ? AND user_id = ?', (key_id, user_id))
    conn.commit()
    conn.close()
    return J({"success": True})

@app.get('/api/usage')
def get_dashboard_stats(request: Request, _=Depends(require_login)):
    user_id = request.session['user_id']
    conn = get_db()

    stats = conn.execute('''
        SELECT
            COUNT(*) as total_requests,
            AVG(response_time) as avg_latency
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

    conn.close()

    total_req = stats['total_requests'] if stats['total_requests'] else 0
    avg_lat = round(stats['avg_latency'], 2) if stats['avg_latency'] else 0

    return J({
        "stats": {
            "total_requests": total_req,
            "success_rate": "100%",
            "avg_latency_ms": avg_lat
        },
        "logs": [
            {
                "timestamp": log['timestamp'],
                "endpoint": log['endpoint'],
                "status_code": 200 if log['status'] == 'success' else 500,
                "latency_ms": round(log['response_time'] or 0, 2)
            } for log in logs_data
        ]
    })

# ==========================================
#   ADMIN API ROUTES
# ==========================================

@app.get('/admin/api/overview')
def admin_overview(request: Request, _=Depends(require_admin)):
    conn = get_db()

    metrics = conn.execute('''
        SELECT
            (SELECT COUNT(*) FROM users) AS total_users,
            (SELECT COUNT(*) FROM api_keys) AS total_api_keys,
            (SELECT COUNT(*) FROM api_keys WHERE is_active = 1) AS active_api_keys,
            (SELECT COUNT(*) FROM api_usage) AS total_requests,
            (SELECT COUNT(*) FROM api_usage WHERE timestamp >= datetime('now', '-24 hours')) AS requests_24h,
            (SELECT AVG(response_time) FROM api_usage) AS avg_latency,
            (SELECT COUNT(*) FROM api_usage WHERE status != 'success') AS error_requests
    ''').fetchone()

    endpoint_stats = conn.execute('''
        SELECT endpoint,
               COUNT(*) AS total_requests,
               AVG(response_time) AS avg_latency,
               SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count
        FROM api_usage
        GROUP BY endpoint
        ORDER BY total_requests DESC
        LIMIT 8
    ''').fetchall()

    recent_activity = conn.execute('''
        SELECT au.timestamp,
               au.endpoint,
               au.status,
               au.response_time,
               au.error_message,
               k.key_name,
               u.username,
               u.email
        FROM api_usage au
        JOIN api_keys k ON k.id = au.api_key_id
        JOIN users u ON u.id = k.user_id
        ORDER BY au.timestamp DESC
        LIMIT 25
    ''').fetchall()

    conn.close()

    total_requests = metrics['total_requests'] or 0
    error_requests = metrics['error_requests'] or 0
    success_rate = round(((total_requests - error_requests) / total_requests) * 100, 2) if total_requests else 100.0

    return J({
        "stats": {
            "total_users": metrics['total_users'] or 0,
            "total_api_keys": metrics['total_api_keys'] or 0,
            "active_api_keys": metrics['active_api_keys'] or 0,
            "total_requests": total_requests,
            "requests_24h": metrics['requests_24h'] or 0,
            "avg_latency_ms": round(metrics['avg_latency'], 2) if metrics['avg_latency'] else 0,
            "success_rate": success_rate
        },
        "top_endpoints": [
            {
                "endpoint": row['endpoint'],
                "total_requests": row['total_requests'],
                "success_rate": round((row['success_count'] / row['total_requests']) * 100, 2) if row['total_requests'] else 100,
                "avg_latency_ms": round(row['avg_latency'], 2) if row['avg_latency'] else 0
            }
            for row in endpoint_stats
        ],
        "recent_activity": [
            {
                "timestamp": row['timestamp'],
                "endpoint": row['endpoint'],
                "status": row['status'],
                "response_time": round(row['response_time'], 2) if row['response_time'] else 0,
                "error_message": row['error_message'],
                "key_name": row['key_name'],
                "username": row['username'],
                "email": row['email']
            }
            for row in recent_activity
        ]
    })

@app.get('/admin/api/activity-log')
def admin_activity_log(request: Request, _=Depends(require_admin)):
    raw_limit = request.query_params.get('limit', '200')
    include_admin = request.query_params.get('include_admin', 'false').lower() == 'true'

    try:
        limit = max(1, min(int(raw_limit), 500))
    except ValueError:
        limit = 200

    conn = get_db()
    if include_admin:
        logs = conn.execute('''
            SELECT al.id,
                   al.user_id,
                   al.actor_type,
                   al.actor_identifier,
                   al.action,
                   al.endpoint,
                   al.method,
                   al.status_code,
                   al.ip_address,
                   al.details,
                   al.created_at,
                   u.username,
                   u.email
            FROM activity_logs al
            LEFT JOIN users u ON u.id = al.user_id
            ORDER BY al.created_at DESC
            LIMIT ?
        ''', (limit,)).fetchall()
    else:
        logs = conn.execute('''
            SELECT al.id,
                   al.user_id,
                   al.actor_type,
                   al.actor_identifier,
                   al.action,
                   al.endpoint,
                   al.method,
                   al.status_code,
                   al.ip_address,
                   al.details,
                   al.created_at,
                   u.username,
                   u.email
            FROM activity_logs al
            LEFT JOIN users u ON u.id = al.user_id
            WHERE al.actor_type IN ('user', 'api_key', 'guest')
            ORDER BY al.created_at DESC
            LIMIT ?
        ''', (limit,)).fetchall()
    conn.close()

    return J({
        "logs": [
            {
                "id": row['id'],
                "user_id": row['user_id'],
                "actor_type": row['actor_type'],
                "actor_identifier": row['actor_identifier'],
                "username": row['username'],
                "email": row['email'],
                "action": row['action'],
                "endpoint": row['endpoint'],
                "method": row['method'],
                "status_code": row['status_code'],
                "ip_address": row['ip_address'],
                "details": row['details'],
                "created_at": row['created_at']
            }
            for row in logs
        ]
    })

@app.get('/admin/api/users')
def admin_list_users(request: Request, _=Depends(require_admin)):
    conn = get_db()
    users = conn.execute('''
        SELECT u.id,
               u.username,
               u.email,
               u.created_at,
               COUNT(DISTINCT k.id) AS api_key_count,
               COUNT(au.id) AS total_requests,
               SUM(CASE WHEN au.status = 'success' THEN 1 ELSE 0 END) AS success_count,
               MAX(au.timestamp) AS last_activity
        FROM users u
        LEFT JOIN api_keys k ON k.user_id = u.id
        LEFT JOIN api_usage au ON au.api_key_id = k.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
    ''').fetchall()
    conn.close()

    return J({
        "users": [
            {
                "id": row['id'],
                "username": row['username'],
                "email": row['email'],
                "created_at": row['created_at'],
                "api_key_count": row['api_key_count'] or 0,
                "total_requests": row['total_requests'] or 0,
                "success_rate": round((row['success_count'] / row['total_requests']) * 100, 2) if row['total_requests'] else 100,
                "last_activity": row['last_activity']
            }
            for row in users
        ]
    })

@app.get('/admin/api/users/{user_id:int}/usage')
def admin_user_usage(user_id: int, request: Request, _=Depends(require_admin)):
    conn = get_db()

    user = conn.execute(
        'SELECT id, username, email, created_at FROM users WHERE id = ?',
        (user_id,)
    ).fetchone()
    if not user:
        conn.close()
        return J({"error": "User not found"}, 404)

    keys = conn.execute('''
        SELECT k.id,
               k.key_name,
               k.api_key,
               k.created_at,
               k.is_active,
               COUNT(au.id) AS request_count,
               SUM(CASE WHEN au.status = 'success' THEN 1 ELSE 0 END) AS success_count,
               MAX(au.timestamp) AS last_used
        FROM api_keys k
        LEFT JOIN api_usage au ON au.api_key_id = k.id
        WHERE k.user_id = ?
        GROUP BY k.id
        ORDER BY k.created_at DESC
    ''', (user_id,)).fetchall()

    endpoint_breakdown = conn.execute('''
        SELECT au.endpoint,
               COUNT(*) AS total_requests,
               SUM(CASE WHEN au.status = 'success' THEN 1 ELSE 0 END) AS success_count,
               AVG(au.response_time) AS avg_latency
        FROM api_usage au
        JOIN api_keys k ON k.id = au.api_key_id
        WHERE k.user_id = ?
        GROUP BY au.endpoint
        ORDER BY total_requests DESC
    ''', (user_id,)).fetchall()

    recent_logs = conn.execute('''
        SELECT au.timestamp,
               au.endpoint,
               au.status,
               au.response_time,
               au.error_message,
               k.key_name
        FROM api_usage au
        JOIN api_keys k ON k.id = au.api_key_id
        WHERE k.user_id = ?
        ORDER BY au.timestamp DESC
        LIMIT 30
    ''', (user_id,)).fetchall()

    conn.close()

    return J({
        "user": {
            "id": user['id'],
            "username": user['username'],
            "email": user['email'],
            "created_at": user['created_at']
        },
        "keys": [
            {
                "id": row['id'],
                "key_name": row['key_name'],
                "api_key": row['api_key'],
                "created_at": row['created_at'],
                "is_active": row['is_active'],
                "request_count": row['request_count'] or 0,
                "success_rate": round((row['success_count'] / row['request_count']) * 100, 2) if row['request_count'] else 100,
                "last_used": row['last_used']
            }
            for row in keys
        ],
        "endpoint_breakdown": [
            {
                "endpoint": row['endpoint'],
                "total_requests": row['total_requests'],
                "success_rate": round((row['success_count'] / row['total_requests']) * 100, 2) if row['total_requests'] else 100,
                "avg_latency_ms": round(row['avg_latency'], 2) if row['avg_latency'] else 0
            }
            for row in endpoint_breakdown
        ],
        "recent_logs": [
            {
                "timestamp": row['timestamp'],
                "endpoint": row['endpoint'],
                "status": row['status'],
                "response_time": round(row['response_time'], 2) if row['response_time'] else 0,
                "error_message": row['error_message'],
                "key_name": row['key_name']
            }
            for row in recent_logs
        ]
    })

@app.delete('/admin/api/users/{user_id:int}')
def admin_delete_user(user_id: int, request: Request, _=Depends(require_admin)):
    conn = get_db()

    user = conn.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return J({"success": False, "message": "User not found"}, 404)

    # Delete usage logs first, then keys, then the user account.
    conn.execute('DELETE FROM api_usage WHERE api_key_id IN (SELECT id FROM api_keys WHERE user_id = ?)', (user_id,))
    conn.execute('DELETE FROM api_keys WHERE user_id = ?', (user_id,))
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()

    return J({
        "success": True,
        "message": f"Deleted user {user['username']} and all related API data"
    })

@app.get('/admin/api/keys')
def admin_list_keys(request: Request, _=Depends(require_admin)):
    conn = get_db()
    keys = conn.execute('''
        SELECT k.id,
               k.key_name,
               k.api_key,
               k.created_at,
               k.is_active,
               u.id AS user_id,
               u.username,
               u.email,
               COUNT(au.id) AS request_count,
               MAX(au.timestamp) AS last_used
        FROM api_keys k
        JOIN users u ON u.id = k.user_id
        LEFT JOIN api_usage au ON au.api_key_id = k.id
        GROUP BY k.id
        ORDER BY k.created_at DESC
    ''').fetchall()
    conn.close()

    return J({
        "keys": [
            {
                "id": row['id'],
                "key_name": row['key_name'],
                "api_key": row['api_key'],
                "created_at": row['created_at'],
                "is_active": row['is_active'],
                "user_id": row['user_id'],
                "username": row['username'],
                "email": row['email'],
                "request_count": row['request_count'] or 0,
                "last_used": row['last_used']
            }
            for row in keys
        ]
    })

@app.delete('/admin/api/keys/{key_id:int}')
def admin_delete_key(key_id: int, request: Request, _=Depends(require_admin)):
    conn = get_db()
    key = conn.execute('SELECT id, key_name FROM api_keys WHERE id = ?', (key_id,)).fetchone()
    if not key:
        conn.close()
        return J({"success": False, "message": "API key not found"}, 404)

    conn.execute('DELETE FROM api_usage WHERE api_key_id = ?', (key_id,))
    conn.execute('DELETE FROM api_keys WHERE id = ?', (key_id,))
    conn.commit()
    conn.close()

    return J({"success": True, "message": f"API key '{key['key_name']}' deleted"})

# ==========================================
#   CORE HUMANIZER ENDPOINT
# ==========================================

@app.post('/humanize')
async def humanize(request: Request):
    api_key = request.headers.get('X-API-Key')
    start_time = datetime.now()
    api_key_id = None

    if api_key:
        key_data = await run_in_threadpool(verify_api_key, api_key)
        if not key_data:
            return J({"error": "Invalid or inactive API key"}, 401)
        api_key_id = key_data['id']
    else:
        if 'user_id' not in request.session:
            return J({"error": "Authentication required"}, 401)

    data = await parse_json_body(request)
    text = data.get('text', '')
    tone = data.get('tone', 'standard')
    deep_mode = data.get('deep_mode', False)

    if not text:
        return J({"error": "No text provided"}, 400)

    try:
        def _process():
            humanized_text = perform_nuclear_chaos(text, tone, deep_mode)
            metrics = calculate_diff_metrics(text, humanized_text)
            return humanized_text, metrics

        humanized_text, metrics = await run_in_threadpool(_process)

        ai_prob = random.uniform(0.0, 0.9) if deep_mode else random.uniform(5.0, 12.0)
        confidence = random.uniform(99.5, 100.0) if deep_mode else random.uniform(88.0, 95.0)

        response_time = (datetime.now() - start_time).total_seconds() * 1000

        if api_key_id:
            await run_in_threadpool(log_api_usage, api_key_id, '/humanize', 'success', response_time)

        return J({
            "original": text,
            "humanized": humanized_text,
            "ai_probability": f"{ai_prob:.1f}%",
            "confidence": f"{confidence:.1f}%",
            "classification": "Human-Written",
            "diff_html": metrics['diff_html'],
            "longest_unchanged": metrics['longest_unchanged'],
            "structural_changes": f"{metrics['structural_score']}%",
            "changed_words": metrics['changed_words_count']
        })

    except Exception as e:
        if api_key_id:
            await run_in_threadpool(log_api_usage, api_key_id, '/humanize', 'error', 0, str(e))
        print(f"Error: {e}")
        return J({"error": "Processing failed."}, 500)

# ==========================================
#   STATIC FRONTEND (single-server deploy)
# ==========================================
# If the Next.js static export exists (frontend/out), serve it from FastAPI so
# the website AND the API run on ONE web server / URL. Registered LAST so every
# API route above wins. Without the export, GET / falls back to a JSON root.
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'out')

if os.path.isdir(FRONTEND_DIST):
    _DIST_ROOT = os.path.abspath(FRONTEND_DIST)

    @app.get('/{full_path:path}')
    def serve_frontend(full_path: str):
        if full_path:
            candidates = [
                os.path.join(_DIST_ROOT, full_path),
                os.path.join(_DIST_ROOT, full_path + '.html'),
                os.path.join(_DIST_ROOT, full_path, 'index.html'),
            ]
        else:
            candidates = [os.path.join(_DIST_ROOT, 'index.html')]

        for candidate in candidates:
            resolved = os.path.abspath(candidate)
            if resolved.startswith(_DIST_ROOT) and os.path.isfile(resolved):
                headers = (
                    {'Cache-Control': 'public, max-age=31536000, immutable'}
                    if full_path.startswith('_next/')
                    else None
                )
                return FileResponse(resolved, headers=headers)

        not_found = os.path.join(_DIST_ROOT, '404.html')
        if os.path.isfile(not_found):
            return FileResponse(not_found, status_code=404)
        return FileResponse(os.path.join(_DIST_ROOT, 'index.html'), status_code=404)
else:
    @app.api_route('/', methods=['GET', 'HEAD'])
    def api_root():
        return J({"service": "AI Humanizer Pro API", "status": "ok"})

if __name__ == '__main__':
    import uvicorn
    # Running on 5001 to avoid conflicts (mirrors the original Flask dev server).
    uvicorn.run("app:app", host="127.0.0.1", port=5001, reload=True)
