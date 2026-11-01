# Gunicorn settings, loaded automatically by `gunicorn app:app`.
# Tuned for small instances such as Render's free tier (512 MB RAM, shared CPU):
# one worker holds the spaCy model (~200 MB in total), and threads keep slow I/O
# (email, OAuth) from blocking other requests. Raise WEB_CONCURRENCY only with more RAM.
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = int(os.environ.get('WEB_CONCURRENCY', 1))
worker_class = 'gthread'
threads = int(os.environ.get('GUNICORN_THREADS', 4))
timeout = 60
graceful_timeout = 20
keepalive = 5

# Render's proxy terminates HTTPS and forwards the original scheme in X-Forwarded-Proto
forwarded_allow_ips = '*'

accesslog = '-'
errorlog = '-'
loglevel = 'info'
