# AI Humanizer Pro

Rewrite AI-generated drafts in a natural, human voice. Pick a tone, review every highlighted change,
and use the same engine from your own apps through a paid REST API.

The web editor is free with an account. The REST API is sold as prepaid plans and top-ups through
Razorpay, priced by region so it stays affordable everywhere.

## Features

**Editor**
- Four tones: standard, professional, casual and academic, plus Chaos Mode for a stronger rewrite.
- Deep Rewrite (beta, optional): small models on the server reword whole sentences before the
  rule-based layers run. Offered only when the models are installed.
- A change map: new words in red, moved words (restructured sentences) underlined in yellow, and the
  longest unchanged run in blue, with counts of changed and moved words and restructured sentences.
- The six most recent results are kept in the browser's local storage (never on the server).

**Rewriting engine** (rule-based, no external AI service)
- Six layers: stock AI phrases, openers and transitions ("plays a crucial role in", "In today's
  fast-paced digital landscape", "cannot be overstated", "questions regarding X" → "questions about
  X"); contractions by tone; word choice; sentence structure (passive to active, "the integration of
  X" → "integrating X", "there are X that" → "X", "it is important to do X" → "doing X is
  important", "it is crucial for X to Y" → "X needs to Y", "this highlights the importance of X" →
  "this shows why X matters", "X is a technique that does Y" → "X does Y", "the key is to do X" →
  "doing X is the key", "changed the way we work" → "changed how we work", "is one of the most" →
  "is among the most", "The authors warn that the sample was small" → "The sample was small, the
  authors warn", "The museum will reopen next spring" → "Next spring, the museum will reopen", "Not
  only does X…, but…", moving adverbial phrases and clauses, reordering lists); rhythm (splitting
  long sentences, including trailing "…, making it …", "…, including …" and "…, although …"
  clauses, and joining related short ones so lengths vary like human writing, with varied
  openings); punctuation.
- Each rule checks that it fits before it fires: a phrase isn't moved where it would describe
  something else ("its highest level since 2019" keeps "since 2019" in place), lists that run in
  order ("small, medium and large") keep their order, a joined sentence keeps the capital of a name
  but not of an ordinary word, and "offered her services" never becomes "had her services".
- Word choice uses hand-checked swaps for overused AI words and formal words, each with rules for
  where it fits ("nearly" stays in "not nearly enough", "sufficient" never follows "an", "cultivate"
  only becomes "build" for things like trust). WordNet synonyms are only tried when the Deep Rewrite
  model is loaded to check each swap in its sentence: about a quarter of them take another sense of
  the word ("sunny" → "gay", "employs 4,500 people" → "applies"), which no word statistic catches.
- Chaos Mode rewrites more of everything: every hand-checked word swap that fits, every optional
  opener, most filler openers ("Moreover,", "In conclusion,") left out entirely, empty intensifiers
  always cut, phrases moved and lists reordered wherever they fit, more contractions in the standard
  tone, and long sentences also split before a closing relative clause ("proposed a new scheme that
  would offer…" → "proposed a new scheme. The new scheme would offer…"). The sentence templates are
  also applied to sentences Deep Rewrite wrote. On its own the gain is modest: a rule only fires
  where it fits, and on most sentences only a few fit, in any mode. Most of Chaos Mode's extra
  reach comes from Deep Rewrite. Combined with Deep Rewrite, it also asks the model for
  more rewrites, gives sentences that changed only a little a second round (kept only if it is at least
  as faithful), and tries more model-checked synonym swaps, nouns included. In the editor, turning
  Chaos Mode on also turns Deep Rewrite on when it is installed. The API keeps the two separate, so
  `deep_mode` alone stays fast: it uses the rules that need no model.
- Input is cleaned first: zero-width and other invisible characters, control characters, odd spaces
  and look-alike letters from other alphabets (often left by copy-pasting from chat tools) are removed,
  and old-style line breaks become ordinary ones, so none of them reach the output.
- spaCy (`en_core_web_sm`) for tags, entities and the dependency parse; lemminflect for word forms.
- Data built offline with NLTK: a WordNet synonym table that keeps only senses both words are
  mainly used in (SemCor counts, `scripts/build_synonyms.py`), and Brown corpus word pairs that
  check a synonym fits its neighbours (`scripts/build_collocations.py`), plus action nouns and
  their verbs (`scripts/build_nominalizations.py`), the main kind of each noun that some swaps
  depend on, so "reduce the cream" in a recipe never becomes "cut the cream" (also built by
  `scripts/build_synonyms.py`), and nouns that name a person (`scripts/build_person_nouns.py`).
  Together about 13 MB in memory, so the app fits Render's 512 MB free tier.
- Names, numbers, dates, acronyms and citations are never changed, and nothing inside quotation
  marks is rewritten (someone's exact words), in straight, curly, „low-high“ or «angle» quotes,
  even when a quotation is wrapped over two lines. Code stays as written too (inline code, ``` and
  ~~~ fences, indented code blocks), and so do HTML tags, attributes and comments. Headings and list
  items only get word-level edits, verse gets no contractions, and a paragraph that isn't in English
  is left as written.
- Deep Rewrite runs a T5 paraphraser in 8-bit with CTranslate2 (`paraphraser.py`) and, when
  installed, back-translation through French with OPUS-MT models trained on human translations,
  whose wording is plainer than the T5's (which learned from ChatGPT paraphrases). Every candidate
  must keep each name, number and date (and how many of each), the topic's nouns and the sentence's
  subject, the key describing words ("remote work", "on an empty stomach", "is efficient"), who does
  what to whom (including who gives and who receives), the negations and what they apply to
  ("prevents", "unable to" and "avoid" count too), what the sentence claims ("must" vs "should", "can"
  vs "may", "studies suggest" vs "studies show", "some studies", "tended to", "often", who said it,
  no new hedges such as "purportedly", the same quantities and bounds such as "most", "up to" or
  "longer than", the same tense) and its point of view. It may add no new details or definitions, no
  stronger claims ("completely", "renowned", "optimal", a superlative), no opposites ("began" →
  "ended", "frequent" → "rare"), no new causes, links or connectives ("resulting in", "As part of a
  deal", "However,"), and no correlation turned into a cause; it must not turn an expectation into an
  outcome ("analysts expected higher earnings" → "expectations were met"), must be a whole,
  grammatical sentence (a main verb, agreeing subject and verb, no dropped article), may not be a
  near-copy that only swaps one word for another sense, may use no made-up words or punctuation, and
  must pass three checks with the T5 (a round trip back to the original, the single least likely
  word, and a forward plausibility check).
  Of the rewrites that are about as faithful as the best one, the most changed is used, ranked down
  if it adds AI-style phrasing. Sentences it rewrites then get word-level edits and, in Chaos Mode,
  the sentence templates; the others go through all the layers. Together the models need about
  450 MB more memory and take roughly 10–20 ms per character on a server with 1.5 CPUs (a
  2,400-character Chaos + Deep Rewrite took 20–25 seconds on an 8-core PC).
- Text is processed in memory and never stored or logged.

**Developer API**
- `POST /humanize` for one text and `POST /humanize/batch` for up to 25 texts per call.
- API keys stored as hashes, rate limits per key, quota headers on every response.
- Dashboard with usage charts (requests per day and per key), recent activity and key management.

**Billing**
- Regional prices by World Bank income group: rupees in India, US dollars elsewhere.
- 30-day and yearly plans (12 months for the price of 10), and top-ups that never expire.
- Optional auto-renewal with Razorpay Subscriptions (UPI AutoPay or card, rupee prices).
- Coupon codes and a referral program (free requests for both sides on a friend's first purchase).
- GST invoices as PDF: CGST + SGST or IGST, zero-rated exports, gapless numbering per financial year.
- Emails for receipts, an 80% quota warning and a reminder before a plan ends.

**Accounts and security**
- Email and password sign-up with email verification and Cloudflare Turnstile bot protection,
  or Google and GitHub sign-in.
- Passwords hashed with scrypt, account lockout after repeated failures, password reset by emailed code.
- Content Security Policy and other security headers, cross-site request checks, rate limits.
- Admin panel for looking up users and payments, issuing refunds, granting plans and managing coupons,
  with an audit log.

**Site**
- Landing, pricing, docs, terms, privacy, refund and contact pages.
- SEO: canonical links, sitemap, robots.txt, structured data and social preview images.

## Tech stack

| Area | Used |
|---|---|
| Web | Python 3.11, Flask 3, Jinja templates, plain CSS and JavaScript, gunicorn |
| NLP | spaCy 3.8, lemminflect; NLTK only for building the synonym table; CTranslate2 and sentencepiece for Deep Rewrite |
| Data | SQLite locally, PostgreSQL in production (psycopg 3 with a connection pool) |
| Payments | Razorpay Orders, Subscriptions and Refunds over its REST API |
| Other | Authlib (OAuth), fpdf2 (invoice PDFs), DB-IP country database (regional pricing), Cloudflare Turnstile |

## Project structure

```
app.py               Routes, auth, API endpoints, security headers, database schema
humanizer.py         Rewriting engine and diff metrics
paraphraser.py       Deep Rewrite model runner: T5 paraphraser and back-translation (optional)
billing.py           Plans, prices, credits, coupons, referrals, Razorpay orders and subscriptions
invoices.py          GST calculation, invoice numbering and PDF rendering
admin.py             Admin panel (Flask blueprint)
mailer.py            Transactional email over SMTP
db.py                SQLite / PostgreSQL access layer
templates/           Pages, partials and email templates
static/              CSS, JavaScript and images
data/                Synonym, word-pair, action-noun, noun-kind and person-noun tables, WordNet
                     licence (GeoIP and the Deep Rewrite models are downloaded or uploaded separately)
scripts/             build_synonyms.py (also builds the noun kinds), build_collocations.py,
                     build_nominalizations.py, build_person_nouns.py, fetch_geoip.py,
                     build_paraphraser.py, fetch_paraphraser.py
.github/workflows/   Daily job that sends plan-expiry reminders
start.py             Production start command for hosts such as Botkeep (python start.py)
gunicorn.conf.py     Server settings tuned for small instances
render.yaml          Render Blueprint
```

## Run it locally

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python scripts/fetch_geoip.py  # country database for regional pricing
```

Create a `.env` file in the project folder (it is git-ignored). The minimum is:

```
SECRET_KEY=any-long-random-string
```

Then start the app and open http://localhost:5001:

```bash
python app.py
```

Without `DATABASE_URL` the app uses a local SQLite file (`humanizer.db`). To test regional pricing
locally, set `GEO_DEFAULT_COUNTRY=IN` (or another country code).

## Configuration

All settings are environment variables, read from `.env` locally or from the host's settings in
production. Leaving an optional value empty turns that feature off.

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | Signs sessions and email links. Required in production. |
| `DATABASE_URL` | PostgreSQL connection string (e.g. a free Neon database). Required in production. |
| `SITE_URL` | Public address, e.g. `https://your-domain.com`, for links, the sitemap and emails. |
| `SENDER_EMAIL`, `SENDER_PASSWORD` | SMTP account for emails (for Gmail, an app password). |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Google sign-in. |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | GitHub sign-in. |
| `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET` | Razorpay API keys. Payments stay off without them. |
| `RAZORPAY_WEBHOOK_SECRET` | Secret for the webhook at `/api/billing/webhook`. |
| `RAZORPAY_AUTOPAY` | `1` to offer auto-renewal (Subscriptions must be enabled on the Razorpay account). |
| `API_TRIAL_REQUESTS` | Free API requests per new account (default 50, `0` to disable). |
| `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY` | Cloudflare Turnstile on the sign-up form. |
| `REQUIRE_EMAIL_VERIFICATION` | `1` or `0`. Empty: on whenever email sending is configured. |
| `ADMIN_EMAILS` | Comma-separated emails of verified accounts that can open `/admin`. |
| `CRON_SECRET` | Shared secret for `POST /internal/tasks`, called by the daily GitHub Action. |
| `BUSINESS_NAME`, `BUSINESS_ADDRESS`, `CONTACT_EMAIL`, `CONTACT_PHONE` | Shown on legal pages, emails and invoices. |
| `GRIEVANCE_OFFICER`, `JURISDICTION_CITY` | Named in the privacy policy and terms. |
| `BUSINESS_GSTIN` | Your GSTIN for tax invoices. Empty: plain receipts. |
| `GST_SAC_CODE`, `INVOICE_PREFIX`, `GST_LUT_ARN` | Optional invoice details (SAC code, number prefix, LUT for exports). |
| `GEO_DEFAULT_COUNTRY` | Country used for pricing when the visitor's location is unknown. |
| `SECURE_COOKIES` | `1` on any HTTPS site (automatic on Render) so login cookies are only sent over HTTPS. |
| `TRUST_PROXY` | `1` when a proxy such as Cloudflare sits in front (automatic on Render), so the app sees the visitor's IP and HTTPS. |
| `PARAPHRASER_URL`, `BACKTRANSLATE_URL` | Download links for `paraphraser.zip` and `backtranslate.zip`; `start.py` installs a missing model from them. |
| `PARAPHRASER_DIR`, `BACKTRANSLATE_DIR` | Where the models live, if not in `data/` (the installer uses them too). |
| `DEEP_REWRITE_MAX_CHARS` | Longest text Deep Rewrite accepts (default 2,500 characters). |
| `DEEP_REWRITE_SECONDS` | Time a Deep Rewrite aims to stay within (default 45). It is not a hard cap: back-translation is skipped once 40% of it has gone, and Chaos Mode's second round when it wouldn't finish in time; with two Deep Rewrites running at once, both skip these extras. |
| `DEEP_REWRITE_QUEUE` | Deep Rewrite requests allowed at once, running or waiting (default 2); more get a 503 at once. |
| `PARAPHRASER_THREADS` | CPU threads for each model (default 2 on a server with at least 2 CPUs, otherwise 1). |
| `GUNICORN_TIMEOUT` | Seconds a worker may go silent before gunicorn restarts it (default 120). It doesn't cut off a long request; the proxy's or client's timeout does, so give Deep Rewrite clients a long one. |

Empty or malformed numbers in these settings fall back to the default, with a warning in the log.

## Deploy on Botkeep

One Botkeep server with 1.5–2 GB of RAM fits the app with Deep Rewrite (measured: about 650 MB in
use with both models loaded, about 850 MB at the peak of a 2,500-character Chaos + Deep Rewrite).
The packages and models take about 0.9 GB of disk, and about 1.2 GB while an uploaded model zip is
being unpacked.

1. **Create the server** from GitHub (this repo, branch `main`) with Python 3.12, and give it most
   of your plan's RAM, CPU and storage.
2. **Start command:** `python start.py`, not `python app.py` (that's the development server).
   `start.py` downloads the country database, installs the Deep Rewrite model if it was uploaded,
   then starts gunicorn on `0.0.0.0` and the port Botkeep assigns (`SERVER_PORT`).
3. **Environment:** the same values as on Render (keep `DATABASE_URL`, so nothing needs migrating),
   plus `SITE_URL=https://your-domain`, `SECURE_COOKIES=1` and `TRUST_PROXY=1`. Botkeep writes these
   to `.env`, which the app loads.
4. **Domains:** add your domain, create the DNS records the panel shows, and wait for it to verify
   and issue the HTTPS certificate. With Cloudflare DNS, keep the records "DNS only" (grey cloud).
5. **Deep Rewrite (optional):** upload `paraphraser.zip` (built below) into the `data` folder and
   restart: it's unpacked into `data/paraphraser/` and the zip is deleted. Alternatively, set
   `PARAPHRASER_URL` to a download link for the zip. The Console then shows `Deep Rewrite: available`.
   For the second, back-translation source, upload `backtranslate.zip` the same way (or set
   `BACKTRANSLATE_URL`); the Console then also shows `Back-translation loaded: fr`.
6. **Point outside services at the new address:** Google and GitHub sign-in callback URLs
   (`/login/google/callback`, `/login/github/callback`), the Razorpay webhook, the Turnstile
   hostname and the `SITE_URL` secret in GitHub Actions. Then test sign-in, the editor and a payment
   before switching the old host off.

### Building the Deep Rewrite models

The server only needs the converted models; converting them needs PyTorch, so do it once on your
own computer (a few GB of free disk):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers<5" ctranslate2 sentencepiece
python scripts/build_paraphraser.py --backtranslate --zip
# writes data/paraphraser/ (~225 MB) + paraphraser.zip and data/backtranslate/ (~160 MB) + backtranslate.zip
```

All are git-ignored. The T5 model is
[humarin/chatgpt_paraphraser_on_T5_base](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base),
licensed under OpenRAIL: its use restrictions apply to anyone who runs or shares it. The
back-translation models are OPUS-MT's `opus-mt-en-fr` and `opus-mt-fr-en`, licensed under CC BY 4.0,
which requires crediting the authors (see Credits). `MODEL_LICENSE.txt` in each folder has the terms.

## Deploy on Render

1. Push this repository to GitHub, then in Render choose **New → Blueprint** and select it.
   `render.yaml` creates a free web service built from the `Dockerfile` and asks for the secrets.
2. Set `DATABASE_URL` to a PostgreSQL database. The free tier's disk is wiped on every restart, so
   SQLite is only for local use.
3. In Razorpay, add a webhook to `https://your-domain/api/billing/webhook` with the events
   `order.paid` and, for auto-renewal, the `subscription.*` events listed in `render.yaml`.
4. For plan-expiry reminders, add `SITE_URL` and `CRON_SECRET` as GitHub Actions secrets
   (Settings → Secrets and variables → Actions). The workflow runs daily at 09:00 IST.

The image build installs the spaCy model and downloads the DB-IP country database. Memory use stays
around 200 MB, within the free tier's 512 MB; that leaves no room for Deep Rewrite, so it stays off
there. The same image runs on any Docker host:

```bash
docker build -t ai-humanizer .
docker run -p 10000:10000 --env-file .env ai-humanizer
```

## API quick start

Create a key on the dashboard, then:

```bash
curl -X POST https://your-domain.com/humanize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "Your text here", "tone": "professional"}'
```

Batch requests send `items`, a list of strings or objects with `text`, `tone`, `deep_mode` and `id`:

```bash
curl -X POST https://your-domain.com/humanize/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"items": ["First text", {"id": "faq-2", "text": "Second text", "tone": "casual"}]}'
```

Responses include `X-Quota-Limit`, `X-Quota-Remaining` and `X-Credits-Remaining`. The full reference,
with error codes and examples in several languages, is at `/docs` on the running site.

## Credits

- Synonym data derived from [WordNet](https://wordnet.princeton.edu/) (licence in `data/WORDNET_LICENSE.txt`).
- IP to country data by [DB-IP](https://db-ip.com), licensed under CC BY 4.0.
- Deep Rewrite model: [humarin/chatgpt_paraphraser_on_T5_base](https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base) (OpenRAIL).
- Back-translation models: [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT) `opus-mt-en-fr` and
  `opus-mt-fr-en` by the Language Technology Research Group, University of Helsinki (Tiedemann &
  Thottingal, EAMT 2020), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  converted to CTranslate2 8-bit, otherwise unchanged.

Developed by Prithwiraj Das.
