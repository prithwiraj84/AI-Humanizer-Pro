# Gunicorn settings, loaded automatically by `gunicorn app:app` (and by start.py).
# Tuned for small instances: one worker holds the spaCy model (~170 MB) and, when Deep Rewrite is
# set up, the paraphrasing model (~250 MB more); threads keep slow I/O (email, OAuth, payments) from
# blocking other requests. Raise WEB_CONCURRENCY only if the server has RAM for another full copy.
import os

# Render sets PORT; Botkeep sets SERVER_PORT
bind = f"0.0.0.0:{os.environ.get('PORT') or os.environ.get('SERVER_PORT') or '10000'}"
workers = int(os.environ.get('WEB_CONCURRENCY', 1))
worker_class = 'gthread'
threads = int(os.environ.get('GUNICORN_THREADS', 4))
# How long a worker may go silent before gunicorn restarts it. With gthread workers this is not a limit on
# a request: a long Deep Rewrite keeps running, and only the proxy's or the client's own timeout cuts it off.
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 120))
graceful_timeout = 20
keepalive = 5

# The host's proxy terminates HTTPS and forwards the original scheme in X-Forwarded-Proto
forwarded_allow_ips = '*'

accesslog = '-'
errorlog = '-'
loglevel = 'info'
