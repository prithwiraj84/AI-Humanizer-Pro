# AI Humanizer Pro — Web (Next.js + Three.js)

A modern, SEO-optimized frontend for **AI Humanizer Pro**, built with the
Next.js App Router, a scroll-driven Three.js landing experience
(react-three-fiber), Tailwind CSS, and Framer Motion. It talks to the existing
**FastAPI** backend.

## Architecture

```
Browser ──► Next.js (localhost:3000)
              │  /backend/*  ──(rewrite proxy)──►  FastAPI (127.0.0.1:5001)
              └─ pages: / /humanizer /dashboard /docs /login /signup ...
```

The Next.js dev server proxies every `/backend/*` request to FastAPI
(`next.config.js`). This keeps the FastAPI **session cookie same-origin**, so
auth works with no CORS configuration. The admin console stays on FastAPI.

## Prerequisites

- Node.js 18+ (tested on v22) and npm
- The FastAPI backend running on port **5001** (`python app.py` in the repo root)

## Getting started

```bash
cd frontend
npm install

# 1) Start the backend (in the project root, separate terminal)
#    python app.py            # serves FastAPI on http://127.0.0.1:5001

# 2) Start the frontend
npm run dev                   # http://localhost:3000
```

Open http://localhost:3000.

## Production build

```bash
npm run build
npm run start
```

## Configuration

| Env var                 | Default                  | Purpose                                   |
| ----------------------- | ------------------------ | ----------------------------------------- |
| `BACKEND_URL`           | `http://127.0.0.1:5001`  | Where `/backend/*` is proxied (build/run) |
| `NEXT_PUBLIC_SITE_URL`  | `http://localhost:3000`  | Canonical site URL for SEO/metadata       |

## Pages

| Route              | Description                                   | SEO        |
| ------------------ | --------------------------------------------- | ---------- |
| `/`                | 3D scroll landing page                        | Indexed    |
| `/humanizer`       | The humanizer tool (auth required)            | noindex    |
| `/dashboard`       | Usage stats + API key management (auth)       | noindex    |
| `/docs`            | API documentation                             | Indexed    |
| `/login`,`/signup` | Auth                                          | noindex    |
| `/forgot-password` | OTP password reset                            | noindex    |

## SEO

- Per-page `metadata` (title template, description, keywords, OpenGraph, Twitter)
- `SoftwareApplication` JSON-LD in the root layout
- `app/sitemap.js` and `app/robots.js`
- Real server-rendered HTML content; the WebGL canvas loads client-side only
