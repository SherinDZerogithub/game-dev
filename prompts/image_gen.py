"""
image_gen.py

Generates scene background illustrations for Lumen.

Primary provider : Google Gemini (gemini-2.5-flash-image) using the
                   google-genai SDK and rotating GEMINI image keys from .env.
Fallback provider : Pollinations free image API (no key required) — used
                    directly when selected or automatically when Gemini is
                    unavailable, misconfigured, blocked, or over quota.

Images are cached by a hash of the final prompt, preventing the same scene
from being generated and billed repeatedly.

The module fails softly: if every provider is unavailable it returns a
cached fallback image or None, and the frontend renders a graceful gradient
backdrop instead.
"""

import base64
import hashlib
import os
import random
import time
import urllib.parse
import urllib.request
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:  # Keep the app/test suite usable without the optional SDK.
    genai = None
    types = None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
IMAGE_MODEL_NAME = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
IMAGE_PROVIDER = os.environ.get("IMAGE_PROVIDER", "pollinations").strip().lower()
POLLINATIONS_URL = os.environ.get(
    "POLLINATIONS_IMAGE_URL", "https://image.pollinations.ai/prompt"
).rstrip("/")

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "image_cache")
)

MAX_PROMPT_LENGTH = 800

# Elegant, cinematic art directions that read well as dimmed backgrounds.
ART_STYLES = [
    "cinematic concept art, moody atmospheric lighting, soft volumetric fog, "
    "muted sophisticated palette, painterly textures, wide establishing shot",
    "editorial illustration, dramatic rim lighting, deep shadows, "
    "desaturated jewel-tone palette, refined composition, film still aesthetic",
    "atmospheric digital painting, golden-hour glow, long shadows, "
    "dreamy depth of field, art-directed color grading, cinematic widescreen",
    "noir-tinged concept art, low-key lighting, single practical light source, "
    "rich blacks, selective color, moody and immersive, filmic",
    "ethereal matte painting, soft diffused light, layered atmospheric depth, "
    "cool twilight palette with a warm accent, elegant and quiet, widescreen",
]

FALLBACK_PROMPTS = [
    "a quiet rain-slicked city street at dusk, neon reflections on wet pavement",
    "a vast cathedral library with shafts of dusty light between towering shelves",
    "a moonlit coastline with distant cliffs and a single lit window",
    "an overgrown conservatory greenhouse bathed in green filtered light",
    "a snow-covered mountain pass under a deep blue twilight sky",
]

_client = None
_gemini_clients = {}


def _split_api_keys(*names: str) -> list[str]:
    """Read one or more image-provider keys from environment variables."""
    keys = []
    for name in names:
        raw = os.environ.get(name, "")
        for value in raw.replace(";", ",").replace("\n", ",").split(","):
            value = value.strip()
            if value and value not in keys:
                keys.append(value)
    return keys


def _gemini_image_api_keys() -> list[str]:
    return _split_api_keys(
        "GEMINI_IMAGE_API_KEYS",
        "GEMINI_IMAGE_API_KEY",
        "GEMINI_API_KEYS",
        "GEMINI_API_KEY",
    )


# ---------------------------------------------------------------------------
# CLIENT
# ---------------------------------------------------------------------------
def _get_client(api_key: str | None = None):
    """Create and cache the Gemini client."""
    global _client
    if genai is None or types is None:
        raise RuntimeError("google-genai is not installed; image generation will use the fallback provider.")
    api_key = api_key or next(iter(_gemini_image_api_keys()), "")
    if not api_key:
        raise RuntimeError(
            "Gemini image generation is disabled because no Gemini image key is set."
        )
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(api_key=api_key)
    _client = _gemini_clients[api_key]
    return _client


# ---------------------------------------------------------------------------
# CACHE
# ---------------------------------------------------------------------------
def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(prompt: str, aspect_ratio: str, suffix: str = "") -> str:
    """Create a stable cache path from the prompt and aspect ratio."""
    _ensure_cache_dir()
    cache_key = f"{IMAGE_MODEL_NAME}|{aspect_ratio}|{suffix}|{prompt}"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{digest}.png")


def _get_art_style(turn_count: int = 0) -> str:
    """Rotate illustration styles based on the game turn."""
    return ART_STYLES[turn_count % len(ART_STYLES)]


# ---------------------------------------------------------------------------
# PROMPT PREPARATION
# ---------------------------------------------------------------------------
def _prepare_prompt(image_prompt: str, turn_count: int, wide: bool) -> str:
    """Clean and enrich a scene description for image generation."""
    if not isinstance(image_prompt, str) or len(image_prompt.strip()) < 3:
        image_prompt = random.choice(FALLBACK_PROMPTS)

    framing = (
        "Wide cinematic 16:9 establishing shot, atmospheric depth, "
        "uncluttered composition with negative space, suitable as a dimmed "
        "background behind light text. "
        if wide
        else "Single polished cinematic scene illustration. "
    )

    prompt = (
        f"{image_prompt.strip()}. "
        f"Visual style: {_get_art_style(turn_count)}. "
        f"{framing}"
        "No captions, labels, logos, watermarks, interface elements, faces "
        "in sharp focus, or readable text."
    )

    if len(prompt) > MAX_PROMPT_LENGTH:
        prompt = prompt[: MAX_PROMPT_LENGTH - 3].rstrip() + "..."
    return prompt


# ---------------------------------------------------------------------------
# GEMINI PROVIDER
# ---------------------------------------------------------------------------
def _response_parts(response: Any) -> list:
    """Return response parts across google-genai SDK response variations."""
    direct_parts = getattr(response, "parts", None)
    if direct_parts:
        return list(direct_parts)

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []

    content = getattr(candidates[0], "content", None)
    if content is None:
        return []

    return list(getattr(content, "parts", None) or [])


def _save_inline_image(part: Any, path: str) -> bool:
    """Extract and save image bytes from a Gemini response part."""
    inline_data = getattr(part, "inline_data", None)
    if inline_data is None:
        return False

    data = getattr(inline_data, "data", None)
    if not data:
        return False

    try:
        if isinstance(data, str):
            image_bytes = base64.b64decode(data)
        else:
            image_bytes = bytes(data)

        with open(path, "wb") as image_file:
            image_file.write(image_bytes)

        return os.path.exists(path) and os.path.getsize(path) > 0
    except (OSError, ValueError, TypeError, base64.binascii.Error):
        return False


def _request_gemini_image(prompt: str, aspect_ratio: str, api_key: str | None = None) -> Any:
    """Send one image-generation request to Gemini."""
    client = _get_client(api_key)
    try:
        # Current google-genai (>=1.x) exposes aspect_ratio via ImageConfig.
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        )
        return client.models.generate_content(
            model=IMAGE_MODEL_NAME, contents=[prompt], config=config
        )
    except (TypeError, ValueError):
        # Compatibility fallback for older google-genai packages without
        # image_config support — aspect ratio is then encoded in the prompt.
        ratio_prompt = f"{prompt} Aspect ratio {aspect_ratio}, widescreen composition."
        return client.models.generate_content(
            model=IMAGE_MODEL_NAME,
            contents=[ratio_prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )


def _generate_gemini_to_path(
    prompt: str, path: str, aspect_ratio: str, api_key: str | None = None
) -> str | None:
    """Generate one image via Gemini and save the first valid image response."""
    response = _request_gemini_image(prompt, aspect_ratio, api_key)

    for part in _response_parts(response):
        if _save_inline_image(part, path):
            return path

        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            try:
                image = as_image()
                if image is not None:
                    image.save(path)
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        return path
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# POLLINATIONS PROVIDER (free, no key, automatic fallback)
# ---------------------------------------------------------------------------
POLLINATIONS_STYLE_SUFFIX = (
    "cinematic concept art, moody atmospheric lighting, muted sophisticated palette, "
    "painterly textures, wide establishing shot, deep shadows, no text, no watermark, "
    "no logo, no UI"
)


def _generate_pollinations_to_path(prompt: str, path: str, width: int, height: int) -> str | None:
    """Generate one image via the free Pollinations endpoint."""
    full = f"{prompt}. {POLLINATIONS_STYLE_SUFFIX}"
    encoded = urllib.parse.quote(full, safe="")
    seed = abs(hash(prompt)) % 1_000_000
    url = (
        f"{POLLINATIONS_URL}/{encoded}"
        f"?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
    )

    request = urllib.request.Request(url, headers={"User-Agent": "Lumen/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
            data = response.read()
        if not data or len(data) < 1024:
            return None
        with open(path, "wb") as image_file:
            image_file.write(data)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    except Exception as exc:
        print(f"[image_gen] Pollinations request failed: {exc}")
    return None


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def _generate_to_path(prompt: str, path: str, aspect_ratio: str) -> str | None:
    """Generate through the selected provider, then use Pollinations."""
    if IMAGE_PROVIDER == "pollinations":
        w, h = (1280, 720) if aspect_ratio == "16:9" else (1024, 768)
        return _generate_pollinations_to_path(prompt, path, w, h)

    # Gemini first, rotating through all configured image keys.
    image_keys = _gemini_image_api_keys()
    for key_index, api_key in enumerate(image_keys, start=1):
        try:
            result = _generate_gemini_to_path(prompt, path, aspect_ratio, api_key)
            if result:
                print(f"[image_gen] image generated with Gemini key #{key_index}")
                return result
        except Exception as exc:
            print(
                f"[image_gen] Gemini key #{key_index} failed; trying the next key/provider: {exc}"
            )

    if not image_keys:
        print("[image_gen] No Gemini image keys configured; using Pollinations fallback.")

    # Pollinations fallback (separate cache slot so it does not collide).
    fallback_path = _cache_path(prompt, aspect_ratio, suffix="poll")
    if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 0:
        return fallback_path
    w, h = (1280, 720) if aspect_ratio == "16:9" else (1024, 768)
    return _generate_pollinations_to_path(prompt, fallback_path, w, h)


def generate_scene_image(
    image_prompt: str,
    turn_count: int = 0,
    force_regenerate: bool = False,
) -> str | None:
    """Generate a wide cinematic scene background."""
    return _generate(image_prompt, turn_count, force_regenerate, wide=True)


def generate_ambient_background(
    image_prompt: str,
    turn_count: int = 0,
    force_regenerate: bool = False,
) -> str | None:
    """Backward-compatible alias — ambient now reuses the wide scene image."""
    return _generate(image_prompt, turn_count, force_regenerate, wide=True)


def _generate(
    image_prompt: str,
    turn_count: int,
    force_regenerate: bool,
    wide: bool,
) -> str | None:
    aspect_ratio = "16:9" if wide else "4:3"
    final_prompt = _prepare_prompt(image_prompt, turn_count, wide=wide)
    path = _cache_path(final_prompt, aspect_ratio)

    if os.path.exists(path) and not force_regenerate:
        return path

    try:
        generated = _generate_to_path(final_prompt, path, aspect_ratio)
        if generated:
            return generated

        # Retry once with a simpler prompt.
        simpler = (
            f"{str(image_prompt).strip()[:180] or random.choice(FALLBACK_PROMPTS)}. "
            "Cinematic atmospheric background, muted palette, no text."
        )
        simpler_path = _cache_path(simpler, aspect_ratio)
        if os.path.exists(simpler_path) and not force_regenerate:
            return simpler_path
        return _generate_to_path(simpler, simpler_path, aspect_ratio) or _get_fallback_image()
    except Exception as exc:
        print(f"[image_gen] Image generation skipped (model={IMAGE_MODEL_NAME}): {exc}")
        return _get_fallback_image()


def _get_fallback_image() -> str | None:
    """Return fallback.png or the newest valid cached PNG."""
    _ensure_cache_dir()
    fallback_path = os.path.join(CACHE_DIR, "fallback.png")
    if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 0:
        return fallback_path
    try:
        cached_files = [
            os.path.join(CACHE_DIR, f)
            for f in os.listdir(CACHE_DIR)
            if f.lower().endswith(".png")
        ]
        valid = [p for p in cached_files if os.path.isfile(p) and os.path.getsize(p) > 0]
        if valid:
            return max(valid, key=os.path.getmtime)
    except OSError:
        pass
    return None


def generate_image(image_prompt: str, turn_count: int = 0) -> str | None:
    """Backward-compatible alias for older call sites."""
    return generate_scene_image(image_prompt=image_prompt, turn_count=turn_count)
