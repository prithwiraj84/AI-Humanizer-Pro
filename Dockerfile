# syntax=docker/dockerfile:1
# ============================================================
#  Single image: builds the Next.js static site (Node) and
#  serves it together with the FastAPI API (Python) — one
#  web server, one URL.
# ============================================================

# ---------- Stage 1: build the Next.js static export ----------
FROM node:22-slim AS frontend
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
# Static export + same-origin API calls (no /backend proxy in single-server mode).
ENV STATIC_EXPORT=true
ENV NEXT_PUBLIC_API_BASE=
# Public URL used for SEO metadata / sitemap (override at build with --build-arg).
ARG NEXT_PUBLIC_SITE_URL=https://noai.devprithwiraj.in
ENV NEXT_PUBLIC_SITE_URL=${NEXT_PUBLIC_SITE_URL}
RUN npm run build          # → /app/frontend/out

# ---------- Stage 2: FastAPI serving the export + API ----------
FROM python:3.12-slim AS app
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/share/nltk_data

COPY requirements.txt ./
RUN pip install -r requirements.txt
# Pre-download NLTK data so the first request is fast and works offline.
RUN python -c "import nltk; nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')"

COPY app.py gunicorn.conf.py ./
COPY --from=frontend /app/frontend/out ./frontend/out

EXPOSE 10000
CMD ["gunicorn", "app:app"]
