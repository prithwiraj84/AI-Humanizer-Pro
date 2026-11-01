# Container image for the Flask app, served by gunicorn (settings in gunicorn.conf.py).
# Used by render.yaml; works on any Docker host. Configuration comes from environment variables.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them (all have prebuilt Linux wheels)
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Country database for regional pricing; never fails the build if DB-IP is unreachable
RUN python scripts/fetch_geoip.py

# Run as an unprivileged user that can still write the local SQLite file (used only without DATABASE_URL)
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

# gunicorn listens on $PORT, which Render sets (10000 by default)
EXPOSE 10000
CMD ["gunicorn", "app:app"]
