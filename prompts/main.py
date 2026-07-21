"""
main.py

FastAPI backend exposing the Lumen — AI Storyteller API.

Run with:
    uvicorn prompts.main:app --reload --port 8000

Endpoints:
    GET  /presets                 Curated opening lore presets.
    POST /start                   Start a new game; returns session id, state, first scene.
    POST /turn                    Submit a player action; returns updated state and next scene.
    GET  /game/{session_id}       Fetch a saved game's state and history.
    GET  /scene-image?path=...    Safely serve generated images from data/image_cache.
    GET  /                        Serve the web frontend.
    GET  /health                  Health check.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from . import game_engine


# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_DIR = os.path.join(BASE_DIR, "frontend", "web")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_CACHE_DIR = os.path.abspath(os.path.join(DATA_DIR, "image_cache"))


# ---------------------------------------------------------------------------
# APPLICATION LIFESPAN
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    yield


app = FastAPI(
    title="Lumen — AI Storyteller",
    description="Backend API for Lumen, an AI-driven interactive story engine.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    opening_prompt: str = Field(
        default="You arrive at a quiet coastal town at dusk with a sealed letter in hand.",
        min_length=1,
        max_length=3000,
    )
    chaos_mode: bool = False
    custom_state: dict[str, Any] | None = None
    character_class: str = ""
    difficulty: str = "Balanced"


class TurnRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    player_input: str = Field(min_length=1, max_length=3000)


# ---------------------------------------------------------------------------
# PRESETS
# ---------------------------------------------------------------------------
@app.get("/presets")
def presets():
    """Return the curated opening lore presets."""
    return {"presets": game_engine.list_presets()}


# ---------------------------------------------------------------------------
# GAME ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/start")
def start_game(req: StartRequest):
    """Start a new game."""
    try:
        return game_engine.new_game(
            opening_prompt=req.opening_prompt.strip(),
            chaos_mode=req.chaos_mode,
            custom_state=req.custom_state,
            character_class=req.character_class,
            difficulty=req.difficulty,
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/turn")
def take_turn(req: TurnRequest):
    """Submit one player action."""
    try:
        return game_engine.take_turn(
            session_id=req.session_id.strip(),
            player_input=req.player_input.strip(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/game/{session_id}")
def get_game(session_id: str):
    """Fetch a saved game."""
    try:
        return game_engine.get_game(session_id.strip())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# IMAGE ENDPOINT
# ---------------------------------------------------------------------------
@app.get("/scene-image")
def scene_image(path: str = Query(..., min_length=1)):
    """Serve a generated image. Only files inside data/image_cache are allowed."""
    requested_path = os.path.realpath(os.path.abspath(path))
    allowed_directory = os.path.realpath(IMAGE_CACHE_DIR)

    try:
        common_directory = os.path.commonpath([requested_path, allowed_directory])
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Image not found") from exc

    if common_directory != allowed_directory:
        raise HTTPException(status_code=404, detail="Image not found")

    if not os.path.isfile(requested_path):
        raise HTTPException(status_code=404, detail="Image not found")

    if not requested_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(status_code=400, detail="Unsupported image format")

    extension = os.path.splitext(requested_path)[1].lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    return FileResponse(
        path=requested_path,
        media_type=media_types[extension],
        filename=os.path.basename(requested_path),
    )


# ---------------------------------------------------------------------------
# FRONTEND AND HEALTH
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """Serve the browser frontend."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html was not found")
    return FileResponse(path=index_path, media_type="text/html")


@app.get("/health")
def health():
    """Simple health check."""
    return {"status": "ok", "message": "Lumen backend is running."}
