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
from datetime import datetime, timedelta
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import db
from . import game_engine
from . import auth


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


class SignupRequest(BaseModel):
    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=10, max_length=200)
    display_name: str = Field(min_length=1, max_length=60)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=5, max_length=200)


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=20, max_length=300)
    password: str = Field(min_length=10, max_length=200)


SESSION_COOKIE = "lumen_session"
SESSION_DAYS = 30


def _normalise_email(email: str) -> str:
    value = email.strip().lower()
    if "@" not in value or "." not in value.rsplit("@", 1)[-1]:
        raise HTTPException(status_code=422, detail="Enter a valid email address")
    return value


def _public_user(user: dict) -> dict:
    profile = db.get_story_profile(user["user_id"])
    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "story_profile": profile,
    }


def _set_session_cookie(response: Response, user_id: int):
    token = auth.new_token()
    expires_at = (datetime.utcnow() + timedelta(days=SESSION_DAYS)).isoformat()
    db.save_auth_session(auth.token_hash(token), user_id, expires_at)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
        secure=os.getenv("LUMEN_SECURE_COOKIES", "0") == "1",
    )


def _current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return db.get_user_by_session(auth.token_hash(token))


def _require_game_access(session_id: str, request: Request):
    saved = game_engine.load_session(session_id)
    if saved is None:
        raise HTTPException(status_code=404, detail=f"No game found for session_id={session_id}")
    user = _current_user(request)
    owner_id = saved.get("user_id")
    if owner_id is not None and (user is None or int(user["user_id"]) != int(owner_id)):
        raise HTTPException(status_code=403, detail="This story belongs to another account")
    return user


def _require_user(request: Request) -> dict:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to access saved stories")
    return user


# ---------------------------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------------------------
@app.get("/auth/me")
def auth_me(request: Request):
    user = _current_user(request)
    return {"user": _public_user(user) if user else None}


@app.post("/auth/signup")
def signup(req: SignupRequest, response: Response):
    email = _normalise_email(req.email)
    display_name = " ".join(req.display_name.strip().split())
    if not display_name:
        raise HTTPException(status_code=422, detail="Enter a display name")
    try:
        user = db.create_user(email, display_name, auth.hash_password(req.password))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _set_session_cookie(response, user["user_id"])
    return {"user": _public_user(user)}


@app.post("/auth/login")
def login(req: LoginRequest, response: Response):
    email = _normalise_email(req.email)
    user = db.get_user_by_email(email)
    if not user or not auth.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect")
    _set_session_cookie(response, user["user_id"])
    return {"user": _public_user(user)}


@app.post("/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        db.delete_auth_session(auth.token_hash(token))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.post("/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    email = _normalise_email(req.email)
    user = db.get_user_by_email(email)
    reset_token = None
    if user:
        reset_token = auth.new_token()
        expires_at = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
        db.create_password_reset(auth.token_hash(reset_token), user["user_id"], expires_at)
    result = {
        "message": "If an account exists for that email, a reset link has been prepared.",
    }
    # There is no email provider in this self-contained project. Returning a
    # one-time token keeps the local demo usable; production should email it.
    if reset_token and os.getenv("LUMEN_SHOW_RESET_TOKEN", "1") == "1":
        result["reset_token"] = reset_token
    return result


@app.post("/auth/reset-password")
def reset_password(req: ResetPasswordRequest, response: Response):
    user_id = db.consume_password_reset(auth.token_hash(req.token.strip()))
    if user_id is None:
        raise HTTPException(status_code=400, detail="That reset link is invalid or expired")
    db.update_password(user_id, auth.hash_password(req.password))
    user = db.get_user(user_id)
    _set_session_cookie(response, user_id)
    return {"user": _public_user(user)}


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
def start_game(req: StartRequest, request: Request):
    """Start a new game."""
    opening_prompt = req.opening_prompt.strip()
    if not opening_prompt:
        raise HTTPException(status_code=422, detail="opening_prompt must contain text")
    try:
        user = _current_user(request)
        return game_engine.new_game(
            opening_prompt=opening_prompt,
            chaos_mode=req.chaos_mode,
            custom_state=req.custom_state,
            character_class=req.character_class,
            difficulty=req.difficulty,
            user_id=user["user_id"] if user else None,
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/games")
def list_games(request: Request):
    """List the current user's saved stories, newest first."""
    user = _require_user(request)
    return {"games": db.list_user_games(user["user_id"])}


@app.post("/turn")
def take_turn(req: TurnRequest, request: Request):
    """Submit one player action."""
    player_input = req.player_input.strip()
    if not player_input:
        raise HTTPException(status_code=422, detail="player_input must contain text")
    try:
        user = _require_game_access(req.session_id.strip(), request)
        return game_engine.take_turn(
            session_id=req.session_id.strip(),
            player_input=player_input,
            user_id=user["user_id"] if user else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/game/{session_id}")
def get_game(session_id: str, request: Request):
    """Fetch a saved game."""
    try:
        _require_game_access(session_id.strip(), request)
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
