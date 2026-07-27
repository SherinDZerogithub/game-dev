# Lumen — AI Storyteller

An AI-driven interactive story engine. You type any action; an AI narrator
writes the next scene, tracks inventory, vitals, relationships and plot
threads, and generates a fresh cinematic background image for every scene.

Built to be elegant, professional and interview-safe — no candy or cartoon
themes. Designed for showcasing in a portfolio and on LinkedIn.

---

## What changed (v2 redesign)

- **New theme.** Replaced the candy/sugar/witch "Sweetheart Quest" with
  **Lumen**, a dark, glassmorphic, editorial aesthetic.
- **No emojis.** Every glyph is a crisp inline SVG icon or a geometric
  unicode mark. Nothing childish.
- **AI background images per scene.** Each narration generates a wide
  cinematic establishing shot that becomes the full-bleed page background and
  crossfades smoothly between scenes.
- **Dark overlay layers.** A radial + linear dark gradient (plus a subtle film
  grain) sits over every background image so white text always stays readable.
- **Flexible AI providers.** Groq is the default text provider, with optional
  Gemini fallback and rotation across multiple keys when a key is rate-limited
  or exhausted.
- **More interactive UI.** Curated starting-world preset cards, animated
  choice cards, twist mode, animated vitals, achievement toasts, smooth
  crossfades and entrance animations.
- **Renamed stat.** `sanity` → `resolve` (old saves auto-migrate).
- **Dropped Streamlit app.** The HTML frontend is the deployable artifact.

---

## Tech stack

- **Backend:** FastAPI + Pydantic v2
- **AI:** Groq chat API for narration, with optional Google Gemini support
  - narration: `llama-3.1-8b-instant`
  - scene images: Pollinations by default, or optional Gemini
- **Image generation:** Pollinations free image API (no key), with optional Gemini
- **Persistence:** SQLite
- **Frontend:** single-file HTML + CSS + vanilla JS (glassmorphism, inline SVG icons)
- **Accounts:** server-side scrypt password hashes, HttpOnly sessions, one-time reset tokens, and per-user story preferences
- **Guest mode:** unsigned-in players can play normally, but guest story state stays in server memory only and is never written to SQLite

---

## Setup

```bash
cd lumen
pip install -r requirements.txt
```

Copy `prompts/.env.example` to `prompts/.env`, then add your keys. The app
accepts multiple keys separated by commas or new lines:

```
# Groq is the free text-generation provider.
LLM_PROVIDER=groq
GROQ_API_KEYS=your-groq-key-1,your-groq-key-2
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant

# Use Pollinations for images; it requires no API key.
IMAGE_PROVIDER=pollinations
```

`LLM_PROVIDER=auto` tries every Groq key, then every Gemini key. Set
`LLM_PROVIDER=gemini` to use Gemini only. For images, set
`IMAGE_PROVIDER=gemini` to use Gemini, `IMAGE_PROVIDER=pollinations` to use
the keyless Pollinations generator, or `IMAGE_PROVIDER=auto` to try Gemini
before Pollinations.

The single-key variables `GROQ_API_KEY` and `GEMINI_API_KEY` also work.
Never commit `prompts/.env` or paste real keys into source code.

Accounts work out of the box with the same SQLite database. Passwords are
never stored directly. For a real deployment, set `LUMEN_SECURE_COOKIES=1`
behind HTTPS and connect `/auth/forgot-password` to an email provider; the
current self-contained demo returns a one-time reset token so it can be tested
without an email service.

---

## Run

```bash
uvicorn prompts.main:app --reload --port 8000
```

Open http://localhost:8000 — the frontend is served from `/`.

---

## API

| Method | Path                   | Purpose                                  |
|--------|------------------------|------------------------------------------|
| GET    | `/presets`             | Curated opening worlds                   |
| POST   | `/auth/signup`         | Create an account                        |
| POST   | `/auth/login`          | Start a secure browser session           |
| POST   | `/auth/logout`         | End the browser session                  |
| POST   | `/auth/forgot-password`| Prepare a one-time reset token           |
| POST   | `/auth/reset-password` | Set a new password                       |
| GET    | `/auth/me`             | Return the current account and preferences |
| POST   | `/start`               | Start a new game                         |
| GET    | `/games`               | List the signed-in user's saved games   |
| POST   | `/turn`                | Submit a player action                   |
| GET    | `/game/{session_id}`   | Fetch a saved game                       |
| GET    | `/scene-image?path=`   | Serve a generated scene image            |
| GET    | `/health`              | Health check                             |

---

## Tests

```bash
python -m pytest prompts/test_game_engine.py -v
# or
python -m unittest prompts.test_game_engine -v
```

---

## Deploy to LinkedIn / interviews

The frontend is a single static `frontend/web/index.html`. For a permanent
public demo, deploy the FastAPI app to any host (Render, Railway, Fly.io) and
share the root URL. The image cache persists in `data/image_cache`.
