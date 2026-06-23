# Gunicorn configuration — auto-loaded by `gunicorn app:app` from the project root.
#
# app.py is a FastAPI (ASGI) application. Gunicorn's default "sync" worker is
# WSGI and calls the app as wsgi(environ, start_response), which fails with:
#   TypeError: FastAPI.__call__() missing 1 required positional argument: 'send'
# Serving it with the Uvicorn worker class makes gunicorn speak ASGI.
import os

worker_class = "uvicorn.workers.UvicornWorker"

# Bind to the port Render (or any host) provides; falls back to 10000 locally.
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Render sets WEB_CONCURRENCY based on the instance size.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))

timeout = 120
