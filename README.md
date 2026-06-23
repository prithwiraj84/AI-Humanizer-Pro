<!-- ====================== HEADER ====================== -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6366f1,a855f7,22d3ee&height=220&section=header&text=AI%20Humanizer%20Pro&fontSize=52&fontColor=ffffff&fontAlignY=38&animation=fadeIn&desc=Make%20AI%20text%20sound%20authentically%20human&descSize=18&descAlignY=60" width="100%" alt="AI Humanizer Pro" />

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=600&size=26&duration=3000&pause=800&color=A855F7&center=true&vCenter=true&width=720&lines=Turn+AI+drafts+into+natural%2C+human+writing;FastAPI+%2B+Next.js+%2B+Three.js;Detector-resistant+%C2%B7+Tone+control+%C2%B7+Deep+mode;Built+for+students+%26+working+professionals" alt="Typing SVG" />
</a>

<br/>

<!-- ====================== BADGES ====================== -->
<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/Three.js-r169-000000?style=for-the-badge&logo=threedotjs&logoColor=white" alt="Three.js" />
  <img src="https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Tailwind" />
</p>

<p>
  <img src="https://img.shields.io/badge/status-active-success?style=flat-square" alt="status" />
  <img src="https://img.shields.io/badge/PRs-welcome-6366f1?style=flat-square" alt="prs" />
  <img src="https://img.shields.io/badge/license-MIT-a855f7?style=flat-square" alt="license" />
  <img src="https://img.shields.io/badge/made%20with-%E2%9D%A4-ec4899?style=flat-square" alt="made with love" />
</p>

<!-- ====================== PREVIEW ====================== -->
<br/>
<img src="frontend/public/og.png" width="760" alt="AI Humanizer Pro preview" />

</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## ✨ Overview

**AI Humanizer Pro** rewrites robotic, AI-generated text into natural writing that reads like a real person — and slips past AI detectors. It ships as a **decoupled full-stack app**:

- ⚡ **FastAPI** ASGI backend (API-only) — auth, API keys, usage analytics, OTP reset, admin console API, and the NLTK-powered humanizer.
- 🎨 **Next.js 14 + Three.js** frontend — an SEO-optimized, scroll-driven 3D landing page plus the humanizer tool, dashboard, docs, and a full admin console.

<div align="center">
<img src="https://skillicons.dev/icons?i=python,fastapi,nextjs,react,threejs,tailwind,sqlite,vercel&theme=dark" alt="tech stack" />
</div>

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 🚀 Features

| | Feature | Description |
|:--:|:--|:--|
| 🪄 | **Deep humanization** | Restructures phrasing, perplexity & rhythm so output reads human, not machine |
| 🛡️ | **Detector-resistant** | Lowers AI-probability scores while preserving your meaning |
| 🎚️ | **Tone control** | Standard · Professional · Casual · Academic voices |
| ⚡ | **Instant results** | Async, non-blocking backend with a live before/after diff |
| 🔑 | **Developer API** | Generate keys and humanize via a single REST endpoint |
| 📊 | **Usage analytics** | Per-user request, latency & success-rate dashboard |
| 🧑‍💼 | **Admin console** | Users, API keys & a full activity log |
| 🌌 | **3D landing page** | Scroll-driven Three.js scene + Framer Motion reveals |
| 🔍 | **SEO-first** | Metadata, OpenGraph, JSON-LD, sitemap & robots |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 🏗️ Architecture

```mermaid
flowchart LR
    U([🧑 Browser]) -->|pages| N["▲ Next.js 14<br/>App Router"]
    N -->|backend proxy| F["⚡ FastAPI<br/>ASGI · Uvicorn"]
    N -.->|WebGL| T["🌌 Three.js<br/>scroll scene"]
    F --> DB[("🗄️ SQLite<br/>WAL")]
    F --> M["📧 SMTP<br/>OTP email"]
    F --> NL["📚 NLTK<br/>WordNet"]

    classDef front fill:#6366f1,stroke:#fff,color:#fff;
    classDef back fill:#009688,stroke:#fff,color:#fff;
    classDef store fill:#a855f7,stroke:#fff,color:#fff;
    class N,T front;
    class F,M,NL back;
    class DB store;
```

> The Next.js dev server proxies every `/backend/*` request to FastAPI, keeping the session cookie **same-origin** (no CORS, no `SameSite` gymnastics).

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 📂 Project structure

```text
ai-humanizer-pro/
├── app.py                 # FastAPI API (auth, keys, usage, admin, humanizer)
├── gunicorn.conf.py       # Uvicorn worker config for production
├── requirements.txt       # Python deps
├── humanizer.db           # SQLite (gitignored)
└── frontend/              # Next.js 14 + Three.js
    ├── next.config.js     # /backend/* → FastAPI proxy
    ├── tailwind.config.js
    └── src/
        ├── app/           # routes: /, /humanizer, /dashboard, /docs, /admin …
        ├── components/    # three/, landing/, admin/, auth/, ui/, layout/
        ├── context/       # AuthContext
        └── lib/           # api client, site config
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## ⚙️ Quick start

> **Prerequisites:** Python 3.11+ and Node.js 18+ (tested on 22).

### 1️⃣ Backend (FastAPI)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                       # → http://127.0.0.1:5001
```

### 2️⃣ Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev                         # → http://localhost:3000
```

Open **http://localhost:3000** 🎉

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 🔐 Environment variables

**Backend**

| Variable | Default | Purpose |
|:--|:--|:--|
| `SECRET_KEY` | random per start | Signs the session cookie — **set in prod** |
| `ADMIN_EMAIL` | `admin@aih.local` | Admin console login |
| `ADMIN_PASSWORD` | `ChangeAdminPassword123!` | **Change in prod** |
| `SENDER_EMAIL` | — | SMTP sender for OTP email |
| `SENDER_PASSWORD` | — | SMTP app password |
| `SMTP_SERVER` / `SMTP_PORT` | `smtp.gmail.com` / `587` | Mail server |
| `PORT` / `WEB_CONCURRENCY` | `10000` / `1` | Set by host (Render) |

**Frontend**

| Variable | Default | Purpose |
|:--|:--|:--|
| `BACKEND_URL` | `http://127.0.0.1:5001` | Proxy target for `/backend/*` |
| `NEXT_PUBLIC_SITE_URL` | `http://localhost:3000` | Canonical URL for SEO |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 🧩 API reference

**`POST /humanize`** — authenticate with a session cookie or an `X-API-Key` header.

<details>
<summary>📦 Request</summary>

```json
{
  "text": "Your AI generated text here",
  "tone": "professional",
  "deep_mode": true
}
```
</details>

<details open>
<summary>🧪 Examples</summary>

```bash
curl -X POST https://your-domain.com/humanize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: hmnz_your_api_key_here" \
  -d '{"text":"...","tone":"professional","deep_mode":true}'
```

```js
const res = await fetch("https://your-domain.com/humanize", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": "hmnz_..." },
  body: JSON.stringify({ text: "...", tone: "professional", deep_mode: true }),
});
const data = await res.json();
console.log(data.humanized);
```
</details>

```mermaid
sequenceDiagram
    participant C as Client
    participant F as FastAPI
    participant N as NLTK
    C->>F: POST /humanize (X-API-Key)
    F->>F: verify key / session
    F->>N: rephrase + perplexity spike
    N-->>F: humanized text
    F->>F: diff metrics + log usage
    F-->>C: { humanized, ai_probability, diff_html, ... }
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 🗺️ Pages

| Route | Description | SEO |
|:--|:--|:--:|
| `/` | 3D scroll landing page | ✅ |
| `/humanizer` | The humanizer tool | 🔒 |
| `/dashboard` | Usage + API keys | 🔒 |
| `/docs` | API documentation | ✅ |
| `/login` · `/signup` · `/forgot-password` | Auth + OTP reset | 🔒 |
| `/admin` · `/admin/login` | Admin console | 🔒 |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## ☁️ Deployment

- **Backend → Render**: `gunicorn app:app` (uses `gunicorn.conf.py` → Uvicorn worker). Set the env vars above.
- **Frontend → Vercel**: set `BACKEND_URL` to your live API and `NEXT_PUBLIC_SITE_URL` to your domain.

> 🔴 **Note:** SQLite on ephemeral hosts resets on redeploy — use a managed Postgres or persistent disk for durable data.

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 🧭 Roadmap

- [x] FastAPI API-only backend
- [x] Next.js + Three.js frontend
- [x] Admin console in Next.js
- [x] SEO (metadata, JSON-LD, sitemap)
- [ ] Migrate SQLite → Postgres
- [ ] Rate limiting & quotas
- [ ] Stripe billing tiers

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider" />

## 👤 Author

**Prithwiraj** — Toshi Consulting
📧 `digital.toshiconsulting@gmail.com`
🌐 [noai.devprithwiraj.in](https://noai.devprithwiraj.in)

<div align="center">

⭐ **If this project helped you, give it a star!** ⭐

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=22d3ee,a855f7,6366f1&height=120&section=footer" width="100%" alt="footer" />

</div>
