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
- **New image API key.** Wired to your Gemini key for both narration
  (`gemini-2.0-flash`) and scene images (`gemini-2.5-flash-image`), with an
  automatic free Pollinations fallback if the image quota runs out.
- **More interactive UI.** Curated starting-world preset cards, animated
  choice cards, twist mode, animated vitals, achievement toasts, smooth
  crossfades and entrance animations.
- **Renamed stat.** `sanity` → `resolve` (old saves auto-migrate).
- **Dropped Streamlit app.** The HTML frontend is the deployable artifact.

---

## Tech stack

- **Backend:** FastAPI + Pydantic v2
- **AI:** Google Gemini via `google-genai`
  - narration: `gemini-2.0-flash`
  - scene images: `gemini-2.5-flash-image`
- **Image fallback:** Pollinations free image API (no key)
- **Persistence:** SQLite
- **Frontend:** single-file HTML + CSS + vanilla JS (glassmorphism, inline SVG icons)

---

## Setup

```bash
cd lumen
pip install -r requirements.txt
```

Your Gemini key is already set in `prompts/.env`:

```
GEMINI_API_KEY=''
```

Get a free key or rotate it at https://aistudio.google.com/app/apikey.

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
| POST   | `/start`               | Start a new game                         |
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
