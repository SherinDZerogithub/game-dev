"""
game_engine.py

The orchestrator. This is "the game" — it wires together:
  state (db.py) -> prompt assembly -> llm_client.py -> chaos.py
  -> state updates -> rolling summary -> image_gen.py
"""

import os
import uuid
import re
from copy import deepcopy
from difflib import SequenceMatcher

from . import chaos
from . import db
from . import image_gen
from . import llm_client

# Guest stories live only for the lifetime of this server process. They are
# intentionally not written to SQLite and disappear when the guest leaves or
# the server restarts.
_GUEST_GAMES: dict[str, dict] = {}

# -----------------------------
# LOAD SYSTEM PROMPT
# -----------------------------
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
with open(
    os.path.join(PROMPTS_DIR, "system_prompt.txt"),
    "r",
    encoding="utf-8",
) as prompt_file:
    SYSTEM_PROMPT = prompt_file.read()

# -----------------------------
# CONFIG
# -----------------------------
SUMMARIZE_EVERY_N_TURNS = 6
CHAOS_MODE_SUMMARIZE_EVERY_N_TURNS = 10
# How many raw history entries to pass to the LLM each turn. A player action and
# its narration are two entries, so six entries is exactly three recent turns.
# Older turns are represented by the rolling summary, keeping token cost flat.
RECENT_TURNS_LIMIT = 6
# How many raw history entries to KEEP after a summary is written.
# Higher than 2 so the model still sees the last few beats of the scene.
HISTORY_KEEP_AFTER_SUMMARY = 4
MIN_STAT_VALUE = 0
MAX_STAT_VALUE = 200
REPETITION_LOOKBACK = 6
REPETITION_NGRAM_SIZE = 4
MAX_REPAIR_ATTEMPTS = 2
FORBIDDEN_TEMPLATE_FRAGMENTS = (
    "the discovery changes what the next step can be",
    "it points toward",
    "a secondary terminal boots itself and begins displaying coordinates",
    # Catches the model echoing the internal [TWIST EVENT] directive
    # verbatim into player-facing narration instead of dramatizing its
    # effect in original prose.
    "twist event",
    "internal gm directive",
)

# -----------------------------
# CURATED OPENING PRESETS (no candy/witch themes — elegant, interview-safe)
# -----------------------------
LORE_PRESETS = [
    {
        "id": "noir",
        "label": "The Language Exchange",
        "icon": "city",
        "opening": (
            "On your first week at university, you volunteer at a language-exchange "
            "café where students from around the world practise together. A notebook "
            "filled with phrases in several languages is left behind with a meeting "
            "time no one recognises. Listen carefully, ask when you do not understand, "
            "and find the student who needs your help before the café closes."
        ),
    },
    {
        "id": "romance",
        "label": "Letters from Vienna",
        "icon": "letter",
        "opening": (
            "A sealed letter arrives at your bookshop in a quiet coastal town, "
            "postmarked Vienna and written in a hand you have not seen in ten years. "
            "It asks you to come. The next train leaves at dawn."
        ),
    },
    {
        "id": "mystery",
        "label": "The Flexible Semester",
        "icon": "compass",
        "opening": (
            "Your first semester has barely begun when a timetable mix-up sends you "
            "to a student project team you never planned to join. Their deadline "
            "moves, a teammate drops out, and the brief changes again. Keep an open "
            "mind, listen to the group, and find a way forward when the plan will not "
            "stay still."
        ),
    },
    {
        "id": "scifi",
        "label": "Signal from Andromeda",
        "icon": "signal",
        "opening": (
            "You are the lone comms officer on a deep-space relay station when an "
            "impossible signal arrives from a galaxy that should be silent. Mission "
            "control is twelve hours away by lightspeed. Decide what to do before the "
            "next transmission window closes."
        ),
    },
    {
        "id": "fantasy",
        "label": "The Cartographer's Heir",
        "icon": "compass",
        "opening": (
            "Your grandmother, the kingdom's last royal cartographer, has left you a "
            "map of a road that appears on no other chart — and a key cut from a "
            "stone that hums when you hold it. The road begins at the edge of the old "
            "forest. Walk it."
        ),
    },
    {
        "id": "slice",
        "label": "The Midnight Study Café",
        "icon": "moon",
        "opening": (
            "At a quiet campus study café that opens after dark, you agree to help a "
            "small group prepare for their first major presentation. Everyone brings "
            "a different strength, one important source is missing, and the clock is "
            "moving fast. Share the work, protect your energy, and help the team find "
            "its own way to a confident finish."
        ),
    },
]

# -----------------------------
# DEFAULT GAME STATE
# -----------------------------
DEFAULT_STATE = {
    "inventory": [],
    "stats": {
        "hp": 100,
        "resolve": 100,
    },
    "flags": {},
    "story_summary": "",
    "turn_count": 0,
    "chaos_mode": False,
    "game_over": False,
    "world_prompt": "",
    "action_history": [],
    "narration_history": [],
    "last_choices": [],
    # Extended story-state features
    "relationships": {},
    "quests": [],
    "discovered_locations": [],
    "last_scene_image": None,
    "ambient_background": None,
    "character_class": "",
    "difficulty": "Balanced",
    "personalization": {
        "sessions_started": 0,
        "turns_played": 0,
        "genres": {},
        "archetypes": {},
        "pacing": {},
    },
    "narrative_context": {
        "location": "",
        "active_characters": [],
        "current_goal": "",
        "open_threads": [],
    },
    # Ring-buffer of recent fallback consequences to prevent the same sentence
    # from being reused across turns when the LLM is over quota.
    "last_consequences": [],
}

# -----------------------------
# INTERNAL HELPERS
# -----------------------------
def _ensure_state_shape(state: dict) -> None:
    """Ensure old save files remain compatible when new fields are added."""
    if not isinstance(state, dict):
        raise TypeError("game state must be a mapping")
    for key, value in DEFAULT_STATE.items():
        if key not in state:
            state[key] = deepcopy(value)

    if not isinstance(state.get("stats"), dict):
        state["stats"] = deepcopy(DEFAULT_STATE["stats"])
    else:
        # Backward-compat: older saves used "sanity"; map to "resolve".
        if "sanity" in state["stats"] and "resolve" not in state["stats"]:
            state["stats"]["resolve"] = state["stats"].pop("sanity")
        for key, value in DEFAULT_STATE["stats"].items():
            state["stats"].setdefault(key, value)
        for key, default in DEFAULT_STATE["stats"].items():
            value = state["stats"].get(key, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                value = default
            state["stats"][key] = max(MIN_STAT_VALUE, min(MAX_STAT_VALUE, value))

    if not isinstance(state.get("narrative_context"), dict):
        state["narrative_context"] = deepcopy(DEFAULT_STATE["narrative_context"])
    else:
        for key, value in DEFAULT_STATE["narrative_context"].items():
            state["narrative_context"].setdefault(key, deepcopy(value))

    # Ensure ring-buffer for fallback consequence deduplication
    if not isinstance(state.get("last_consequences"), list):
        state["last_consequences"] = []
    else:
        state["last_consequences"] = [
            str(item).strip()
            for item in state["last_consequences"]
            if str(item).strip()
        ][-6:]

    for key in ("inventory", "flags", "relationships", "quests", "discovered_locations"):
        expected = dict if key in {"flags", "relationships"} else list
        if not isinstance(state.get(key), expected):
            state[key] = deepcopy(DEFAULT_STATE[key])
    for key in ("action_history", "narration_history", "last_choices"):
        if not isinstance(state.get(key), list):
            state[key] = deepcopy(DEFAULT_STATE[key])
    if not isinstance(state.get("turn_count"), int) or isinstance(state.get("turn_count"), bool):
        state["turn_count"] = 0
    state["turn_count"] = max(0, state["turn_count"])


def _format_bullets(values: list) -> str:
    if not values:
        return "  - None"
    return "\n".join(f"  - {value}" for value in values)


def _join_text_blocks(*blocks: object) -> str:
    """Join independent prose/state blocks without letting paragraphs bleed together."""
    return "\n\n".join(
        str(block).strip()
        for block in blocks
        if block is not None and str(block).strip()
    )


def _clip_recent_turns(turns: list, limit: int = RECENT_TURNS_LIMIT) -> list:
    """Return only the bounded raw history window used for the next LLM call."""
    if not isinstance(turns, list):
        return []
    return turns[-max(0, limit):]


def _format_choice_block(choices: list[str]) -> str:
    """Render the API-facing choice block as predictable Markdown."""
    option_lines = "\n".join(f"* {str(choice).strip()}" for choice in choices if str(choice).strip())
    return _join_text_blocks("What do you want to do?", option_lines)


def _normalize_narration_paragraphs(value: object) -> str:
    """Preserve prose blocks while guaranteeing a blank line between them."""
    text = re.sub(r"\r\n?", "\n", str(value or "")).strip()
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n{2,}", text)
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs)


def _assemble_prompt(
    state: dict,
    recent_turns: list,
    player_input: str,
    chaos_fragment: str | None,
) -> str:
    """Build the complete prompt sent to the LLM."""
    _ensure_state_shape(state)

    recent_turns = _clip_recent_turns(recent_turns)
    history_text = _join_text_blocks(*(
        f"{turn.get('role', 'unknown')}: {turn.get('text', '')}"
        for turn in recent_turns
    ))
    if not history_text:
        history_text = "No previous turns."

    relationships = state.get("relationships", {})
    relationships_text = (
        _join_text_blocks(*(f"  - {name}: {value}" for name, value in relationships.items()))
        if relationships
        else "  - None"
    )

    narrative_context = state.get("narrative_context", {})
    recent_actions = state.get("action_history", [])[-6:]
    recent_narrations = state.get("narration_history", [])[-8:]
    active_characters = narrative_context.get("active_characters", [])
    open_threads = narrative_context.get("open_threads", [])

    chaos_text = (
        "\n[TWIST EVENT — INTERNAL GM DIRECTIVE, NOT PLAYER-FACING TEXT]\n"
        f"{chaos_fragment}\n"
        "Do not copy, quote, or paraphrase the directive above. Do not include the "
        "words \"twist event\" or any part of this instruction in the narration. "
        "Instead, express its effect entirely through original in-world prose — "
        "action, dialogue, and sensory detail — exactly as you would for any other "
        "turn.\n"
        if chaos_fragment
        else ""
    )

    character_line = ""
    if state.get("character_class"):
        character_line = f"\nCharacter archetype: {state['character_class']}\n"
    difficulty_line = ""
    if state.get("difficulty"):
        difficulty_line = f"\nDifficulty pacing: {state['difficulty']}\n"

    profile = state.get("personalization", {})
    profile_line = ""
    if profile and profile.get("sessions_started", 0):
        genres = profile.get("genres", {})
        archetypes = profile.get("archetypes", {})
        pacing = profile.get("pacing", {})
        favorite_genre = max(genres, key=genres.get) if genres else ""
        favorite_archetype = max(archetypes, key=archetypes.get) if archetypes else ""
        favorite_pacing = max(pacing, key=pacing.get) if pacing else ""
        profile_line = (
            "\n[PLAYER PREFERENCE MEMORY]\n"
            f"This returning player has completed {profile.get('turns_played', 0)} prior turns. "
            f"They often enjoy {favorite_genre or 'varied worlds'}"
            f"{f', especially when playing as {favorite_archetype}' if favorite_archetype else ''}. "
            f"Their usual pacing is {favorite_pacing or 'balanced'}. "
            "Personalize the tone and optional choices subtly; do not mention this memory or force a genre.\n"
        )

    location_text = narrative_context.get("location", "") or "Unknown"

    banned_phrases_instruction = ""
    if recent_narrations:
        banned_phrases_instruction = (
            "\n[CRITICAL ANTI-REPETITION CONSTRAINT]\n"
            "Do not end this turn with a variation of 'The discovery changes...' "
            "or 'It points toward...'. Do not treat the current goal, open thread, "
            "or clue as a fill-in-the-blank sentence template. Avoid any four-word "
            "sequence already used in the recent narrations. Show an organic, "
            "physical consequence of the player's action and immediately change "
            "the scene, character situation, available information, or danger.\n"
        )

    flags = state.get("flags", {})
    discovered_flags_lines = []
    for flag_key, flag_val in flags.items():
        if flag_val is True or flag_val == "discovered" or flag_val == "found":
            label = flag_key.replace("_", " ")
            discovered_flags_lines.append(label)
    discovered_flags_text = (
        _join_text_blocks(*(f"  - {line}" for line in discovered_flags_lines))
        if discovered_flags_lines
        else "  - None"
    )

    # ------------------------------------------------------------------
    # FSM SNAPSHOT — Strict structured state block the LLM must respect.
    # Injected as a labelled section so the model cannot miss it.
    # ------------------------------------------------------------------
    fsm_block = f"""[FINITE STATE MACHINE — STRICT LOCATION & GOAL LOCK]
CURRENT LOCATION: {location_text}
  You MUST place the player in this exact location. Do not invent a new
  room, street, station, or building unless the action explicitly moves
  them elsewhere.

CURRENT GOAL: {narrative_context.get("current_goal", "") or "None"}
  This is the active narrative thread. Advance or complicate it — never
  reset to a generic scene.

ACTIVE CHARACTERS PRESENT:
{_format_bullets(active_characters)}
  These characters exist in the current scene. Do not introduce unnamed
  strangers unless plausible AND you add them to active_characters.

OPEN STORY THREADS:
{_format_bullets(open_threads)}
  Preserve or explicitly resolve one of these. Do not silently drop them.

ALREADY DISCOVERED (FLAG GUARD):
{discovered_flags_text}
  The player already knows these facts. NEVER narrate discovering them
  again. Build forward from what is already known."""

    return f"""
{SYSTEM_PROMPT}
{character_line}{difficulty_line}{profile_line}
[WORLD PREMISE / PLAYER-GIVEN PROMPT]
{state.get("world_prompt", "") or "Use the opening scenario and recent story as the world premise."}

[STORY SUMMARY]
{state.get("story_summary", "") or "No summary yet."}

{fsm_block}

Continuity instruction: Treat the player input as an open-ended action, not as a menu command.
Connect it to the established story, show a concrete consequence, and preserve important
characters, goals, clues, locations, and open threads. Never reset to a generic scene.
The current LOCATION is authoritative: do not narrate a new room, platform, street, or
building unless the player's action clearly moves there. Keep time continuous as well;
an action takes moments unless the player explicitly waits, travels, or advances time.
If the player repeats an action, acknowledge that it was already attempted and produce a
different consequence, obstacle, discovery, or character reaction. Return exactly three
short, distinct suggestions, while still allowing any custom typed action. The suggestions
should be diverse, creative, and directly tied to the current narrative context.

[PLAYER STATS]
Vitality: {state["stats"].get("hp", 100)}
Resolve: {state["stats"].get("resolve", 100)}

[INVENTORY]
{_format_bullets(state.get("inventory", []))}

[FLAGS]
{state.get("flags", {})}

[RELATIONSHIPS]
{relationships_text}

[ACTIVE QUESTS]
{_format_bullets(state.get("quests", []))}

[DISCOVERED LOCATIONS]
{_format_bullets(state.get("discovered_locations", []))}

[LAST SUGGESTIONS]
{_format_bullets(state.get("last_choices", []))}

[RECENT PLAYER ACTIONS]
{_format_bullets(recent_actions)}

[RECENT NARRATIONS — DO NOT REPEAT OR PARAPHRASE THESE]
{_format_bullets(recent_narrations)}
{banned_phrases_instruction}
[RECENT TURNS]
{history_text}

{chaos_text}

[PLAYER INPUT]
{player_input}
""".strip()

def _apply_mapping_updates(target: dict, updates: dict | None) -> None:
    if isinstance(updates, dict):
        target.update(updates)


def _split_choice_text(value: str) -> list[str]:
    """Turn a display-style choice string into individual suggestions."""
    text = re.sub(r"\r\n?", "\n", value).strip()
    if not text:
        return []

    # Accept formats commonly returned by models or older clients:
    # newline/bullet/numbered lists, pipe/semicolon/middle-dot joins, and
    # simple comma-separated suggestions.
    text = re.sub(r"(?m)^\s*(?:[-*•▪◦]|\d+[.)])\s*", "", text)
    text = re.sub(r"\s+\d+[.)]\s+", "\n", text)
    text = re.sub(r"\s*(?:\||;|·|•)\s*", "\n", text)
    parts = [part.strip() for part in re.split(r"\n+", text) if part.strip()]

    if len(parts) == 1:
        comma_parts = [part.strip() for part in re.split(r",\s+(?=[A-Z0-9])", text)]
        if 1 < len(comma_parts) <= 3:
            parts = [part for part in comma_parts if part]
    return parts


def _normalize_choices(raw_choices: object) -> list[str]:
    """Normalize choice arrays, delimited strings, and choice objects."""
    if isinstance(raw_choices, str):
        return _split_choice_text(raw_choices)
    if isinstance(raw_choices, dict):
        for key in ("text", "label", "choice", "description", "value"):
            if key in raw_choices:
                return _normalize_choices(raw_choices[key])
        return []
    if not isinstance(raw_choices, (list, tuple)):
        return []

    normalized = []
    for item in raw_choices:
        normalized.extend(_normalize_choices(item))
    return normalized


def _apply_turn_result(state: dict, result: dict) -> dict:
    """Apply an LLM turn result to state and build the scene response."""
    _ensure_state_shape(state)

    narration = _normalize_narration_paragraphs(result.get("narration", ""))
    if not narration:
        narration = (
            "A faint movement catches your eye nearby, while the surrounding silence "
            "waits for your response."
        )

    state_updates = result.get("state_updates")
    if isinstance(state_updates, dict):
        for key, value in state_updates.items():
            if key in {"stats", "flags", "relationships", "narrative_context"}:
                if isinstance(value, dict):
                    _apply_mapping_updates(state[key], value)
            elif key in state:
                state[key] = deepcopy(value)

    # Model output is untrusted JSON.  Re-normalize immediately so malformed
    # state_updates cannot turn the next turn into an exception or an endless
    # fallback loop (for example, by replacing inventory with a string).
    _ensure_state_shape(state)

    stats_delta = result.get("stats_delta", {})
    if isinstance(stats_delta, dict):
        for key, delta in stats_delta.items():
            # Backward-compat: map legacy "sanity" deltas to "resolve".
            if key == "sanity":
                key = "resolve"
            if not isinstance(delta, (int, float)):
                continue
            current = state["stats"].get(key, 0)
            if not isinstance(current, (int, float)):
                current = 0
            state["stats"][key] = max(
                MIN_STAT_VALUE, min(MAX_STAT_VALUE, current + delta),
            )

    add_items = result.get("add_items", [])
    if isinstance(add_items, list):
        for item in add_items:
            if item not in state["inventory"]:
                state["inventory"].append(item)

    remove_items = result.get("remove_items", [])
    if isinstance(remove_items, list):
        for item in remove_items:
            if item in state["inventory"]:
                state["inventory"].remove(item)

    _apply_mapping_updates(state["flags"], result.get("flags"))
    _apply_mapping_updates(state["relationships"], result.get("relationships"))

    narrative_updates = result.get("narrative_context_updates")
    if isinstance(narrative_updates, dict):
        for key, value in narrative_updates.items():
            if key in state["narrative_context"]:
                state["narrative_context"][key] = deepcopy(value)
            elif key == "location":
                state["narrative_context"]["location"] = deepcopy(value)

    add_quests = result.get("add_quests", [])
    if isinstance(add_quests, list):
        for quest in add_quests:
            if quest not in state["quests"]:
                state["quests"].append(quest)

    complete_quests = result.get("complete_quests", [])
    if isinstance(complete_quests, list):
        for quest in complete_quests:
            if quest in state["quests"]:
                state["quests"].remove(quest)

    discovered_locations = result.get("discovered_locations", [])
    if isinstance(discovered_locations, list):
        for location in discovered_locations:
            if location not in state["discovered_locations"]:
                state["discovered_locations"].append(location)

    if result.get("game_over"):
        state["game_over"] = True

    turn_count = state.get("turn_count", 0)
    image_prompt = result.get("image_prompt") or narration
    scene_image = image_gen.generate_scene_image(image_prompt, turn_count)
    state["last_scene_image"] = scene_image
    state["ambient_background"] = scene_image

    # Get choices from result, ensure they're properly formatted
    choices = _normalize_choices(result.get("choices_hint", result.get("choices", [])))
    
    # Clean and deduplicate choices
    clean_choices = []
    seen_choices = set()
    for choice in choices:
        if not isinstance(choice, str):
            continue
        cleaned = choice.strip()
        key = _normalized_text(cleaned)
        if cleaned and key not in seen_choices:
            clean_choices.append(cleaned)
            seen_choices.add(key)

    # Keep the first three unique model suggestions only after all suggestions
    # have been inspected. This avoids losing valid options after duplicates.
    clean_choices = clean_choices[:3]

    # If we don't have 3 choices, fill the remaining slots from the current
    # action instead of guessing from keywords in the world premise.
    if len(clean_choices) < 3:
        recent_actions = state.get("action_history", [])
        last_action = recent_actions[-1] if recent_actions else "investigate"
        dynamic_fallbacks = [
            f"Confront the immediate situation regarding {last_action}",
            "Search the local environment for an alternative path",
            "Force an answer from the environment directly",
        ]

        for choice in dynamic_fallbacks:
            if len(clean_choices) < 3:
                key = _normalized_text(choice)
                if key not in seen_choices:
                    clean_choices.append(choice)
                    seen_choices.add(key)

    # If still less than 3, use the most basic fallbacks
    while len(clean_choices) < 3:
        fallbacks = [
            "explore your surroundings",
            "talk to someone nearby",
            "look for a useful object"
        ]
        for choice in fallbacks:
            if len(clean_choices) < 3:
                key = _normalized_text(choice)
                if key not in seen_choices:
                    clean_choices.append(choice)
                    seen_choices.add(key)
    
    # Store choices in state for reference
    state["last_choices"] = clean_choices[:3]

    return {
        "narration": narration,
        "choices_hint": clean_choices[:3],
        "choices_markdown": _format_choice_block(clean_choices[:3]),
        "image_prompt": image_prompt,
        "image_path": scene_image,
        "ambient_path": scene_image,
        "game_over": state["game_over"],
    }


def _apply_custom_state(state: dict, custom_state: dict | None) -> None:
    if not isinstance(custom_state, dict):
        return
    if isinstance(custom_state.get("inventory"), list):
        state["inventory"] = deepcopy(custom_state["inventory"])
    for mapping_key in ("stats", "flags", "relationships", "narrative_context"):
        value = custom_state.get(mapping_key)
        if isinstance(value, dict):
            state[mapping_key].update(deepcopy(value))
    for list_key in ("quests", "discovered_locations"):
        value = custom_state.get(list_key)
        if isinstance(value, list):
            state[list_key] = deepcopy(value)


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))


def _similarity(a: str, b: str) -> float:
    a_norm = _normalized_text(a)
    b_norm = _normalized_text(b)
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _word_tokens(value: str) -> list[str]:
    """Return lowercase alphanumeric tokens for structural comparisons."""
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def _contains_repeated_ngram(new_text: str, old_text: str, size: int) -> bool:
    """Return True when both texts contain an identical consecutive word block."""
    new_words = _word_tokens(new_text)
    old_words = _word_tokens(old_text)
    if len(new_words) < size or len(old_words) < size:
        return False

    old_blocks = {
        tuple(old_words[index:index + size])
        for index in range(len(old_words) - size + 1)
    }
    return any(
        tuple(new_words[index:index + size]) in old_blocks
        for index in range(len(new_words) - size + 1)
    )


def _result_repeats_recent_story(result: dict, state: dict) -> bool:
    """Detect exact templates, copied blocks, and highly similar recent turns."""
    narration = str(result.get("narration", "")).strip()
    if not narration:
        return False

    narration_lower = narration.lower()
    if any(fragment in narration_lower for fragment in FORBIDDEN_TEMPLATE_FRAGMENTS):
        return True

    recent_narrations = [
        str(old).strip()
        for old in state.get("narration_history", [])[-REPETITION_LOOKBACK:]
        if str(old).strip()
    ]

    for old_narration in recent_narrations:
        normalized_narration = _normalized_text(narration)
        normalized_old_narration = _normalized_text(old_narration)
        if _similarity(normalized_narration, normalized_old_narration) >= 0.50:
            return True
        if _contains_repeated_ngram(
            normalized_narration,
            normalized_old_narration,
            REPETITION_NGRAM_SIZE,
        ):
            return True

    return False



def _remove_forbidden_template_sentences(result: dict) -> dict:
    """Remove known stock bridge sentences without adding replacement prose."""
    cleaned = deepcopy(result) if isinstance(result, dict) else {}
    narration = str(cleaned.get("narration", "")).strip()
    if not narration:
        return cleaned

    sentences = re.split(r"(?<=[.!?])\s+", narration)
    kept = [
        sentence
        for sentence in sentences
        if not any(
            fragment in sentence.lower()
            for fragment in FORBIDDEN_TEMPLATE_FRAGMENTS
        )
    ]
    # If the whole draft is stock text, returning the original draft defeats
    # the purpose of this last-resort sanitizer and lets the loop leak through
    # to the player.  Keep a short neutral bridge that is safe to repair on the
    # next turn instead.
    cleaned["narration"] = (
        " ".join(kept).strip()
        or "The scene shifts around you, leaving one immediate detail impossible to ignore."
    )
    return cleaned


def _sanitize_player_input_echo(narration: str, player_input: str) -> str:
    """Keep the command backend-only if a provider echoes it into narration."""
    text = str(narration or "").strip()
    raw = str(player_input or "").strip()
    if not text or not raw:
        return text

    raw_normalized = _normalized_text(raw)
    if len(raw_normalized.split()) < 2:
        return text

    # Remove a sentence that contains the complete command, including common
    # model-added leading pronouns such as "You". Do not expose a mangled
    # command by deleting only the matching substring in the middle of prose.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for sentence in sentences:
        if re.search(
            rf"(?<![a-z0-9]){re.escape(raw_normalized)}(?![a-z0-9])",
            _normalized_text(sentence),
        ):
            continue
        kept.append(sentence)
    return " ".join(kept).strip() or (
        "The scene shifts around you, leaving one immediate detail impossible to ignore."
    )

def _narration_leaks_injected_instructions(result: dict) -> bool:
    """Detect when the narration substantially reproduces the raw
    [TWIST EVENT] directive text that was injected into the prompt, even if
    it doesn't contain the literal marker phrases in FORBIDDEN_TEMPLATE_FRAGMENTS
    (e.g. only the back half of a leaked block survived a sentence strip)."""
    narration = str(result.get("narration", "")).strip()
    if not narration:
        return False
    for fragment in chaos.CHAOS_FRAGMENTS:
        if _contains_repeated_ngram(narration, fragment, 6):
            return True
    return False


def _generate_non_repeating_turn(full_prompt: str, state: dict) -> dict:
    """Generate a turn and repair it when it repeats a recent structural pattern."""
    result = llm_client.generate_turn(
        full_prompt, temperature=0.85, last_consequences=state.get("last_consequences", [])
    )
    # Track fallback consequence to prevent repetition on next turn.
    _fb_consequence = result.pop("_fallback_consequence", "")
    if _fb_consequence:
        state.setdefault("last_consequences", []).append(_fb_consequence)
        state["last_consequences"] = state["last_consequences"][-6:]

    if not _result_repeats_recent_story(result, state) and not _narration_leaks_injected_instructions(result):
        return result

    repair_prompt = full_prompt
    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        repair_prompt += (
            "\n\n[REPAIR REQUIRED — ATTEMPT "
            f"{attempt}]\n"
            "The rejected draft repeated wording or a structural template from a "
            "recent turn. Rewrite the entire narration from scratch. Do not use "
            "the phrases 'The discovery changes', 'the next step can be', or "
            "'it points toward'. Do not reuse any consecutive four-word block from "
            "the recent narrations. Preserve continuity, but introduce a concrete "
            "new consequence: an object changes state, a character acts or speaks, "
            "new evidence becomes physically available, the environment changes, "
            "or an immediate obstacle appears. Return a complete valid JSON object."
        )
        # Re-read last_consequences fresh on every attempt (rather than the
        # value captured before this turn started) — otherwise a retry has no
        # way of knowing which consequence the *previous* retry in this same
        # turn just picked, and can re-select the exact phrase that just got
        # rejected for repeating.
        result = llm_client.generate_turn(
            repair_prompt,
            temperature=min(1.25, 1.0 + attempt * 0.1),
            last_consequences=state.get("last_consequences", []),
        )
        _fb_consequence = result.pop("_fallback_consequence", "")
        if _fb_consequence:
            state.setdefault("last_consequences", []).append(_fb_consequence)
            state["last_consequences"] = state["last_consequences"][-6:]
        if not _result_repeats_recent_story(result, state) and not _narration_leaks_injected_instructions(result):
            return result

    # Every repair attempt still failed. If the draft is merely reusing a
    # stock bridge sentence, strip just that sentence and keep the rest of
    # the model's organic prose. But if the draft is actually leaking the
    # raw injected [TWIST EVENT] directive, a sentence-level strip isn't
    # safe — only the first offending sentence contains the literal marker
    # phrase, and the remaining leaked instruction text would still reach
    # the player. In that case, discard the leaking draft entirely and use
    # the fully deterministic, chaos-free fallback narrator instead.
    if _narration_leaks_injected_instructions(result):
        result = llm_client._fallback_turn(full_prompt, state.get("last_consequences", []))
    cleaned = _remove_forbidden_template_sentences(result)
    if _result_repeats_recent_story(cleaned, state):
        # A provider can keep returning the same draft even after repair
        # prompts, and a deterministic fallback can collide with a recent
        # variant too.  Do not hand the repeated scene to the player as the
        # final fallback; preserve its choices/state while giving the story a
        # fresh, neutral beat to build from next turn.
        cleaned["narration"] = (
            "A new movement breaks the pattern: the scene shifts around you, "
            "and a fresh decision is now unavoidable."
        )
    return cleaned


def _remember_turn(state: dict, player_input: str, narration: str, choices: list) -> None:
    state.setdefault("action_history", []).append(player_input.strip())
    state["action_history"] = state["action_history"][-20:]
    state.setdefault("narration_history", []).append(narration.strip())
    state["narration_history"] = state["narration_history"][-12:]
    state["last_choices"] = list(choices[:3])


# -----------------------------
# PUBLIC API
# -----------------------------
def list_presets() -> list:
    """Return the curated opening lore presets for the frontend."""
    return deepcopy(LORE_PRESETS)


def load_session(session_id: str):
    """Load a persistent account game or a temporary in-memory guest game."""
    return db.load_game(session_id) or _GUEST_GAMES.get(session_id)


def new_game(
    opening_prompt: str,
    chaos_mode: bool = False,
    custom_state: dict | None = None,
    character_class: str = "",
    difficulty: str = "Balanced",
    user_id: int | None = None,
) -> dict:
    """Create, generate, persist, and return a new game session."""
    if not isinstance(opening_prompt, str) or not opening_prompt.strip():
        raise ValueError("opening_prompt must contain non-whitespace text")
    db.init_db()
    session_id = str(uuid.uuid4())
    state = deepcopy(DEFAULT_STATE)
    state["chaos_mode"] = chaos_mode
    state["world_prompt"] = opening_prompt.strip()
    state["character_class"] = character_class.strip() if isinstance(character_class, str) else ""
    state["difficulty"] = difficulty.strip() if isinstance(difficulty, str) else "Balanced"
    if user_id is not None:
        state["personalization"] = db.get_story_profile(user_id)

    _apply_custom_state(state, custom_state)
    _ensure_state_shape(state)

    player_start = f"(Game start) {opening_prompt}"
    full_prompt = _assemble_prompt(
        state=state,
        recent_turns=[],
        player_input=player_start,
        chaos_fragment=None,
    )

    result = _generate_non_repeating_turn(full_prompt, state)
    scene = _apply_turn_result(state, result)
    _remember_turn(state, player_start, scene["narration"], scene["choices_hint"])

    history = [
        {"role": "player", "text": player_start},
        {"role": "narrator", "text": scene["narration"]},
    ]

    if user_id is None:
        _GUEST_GAMES[session_id] = {"state": state, "history": history, "user_id": None}
    else:
        db.save_game(session_id, state, history, user_id=user_id)
        db.record_story_usage(
            user_id,
            opening_prompt=opening_prompt,
            character_class=state["character_class"],
            difficulty=state["difficulty"],
        )

    return {
        "session_id": session_id,
        "state": state,
        "scene": scene,
    }


def take_turn(session_id: str, player_input: str, user_id: int | None = None) -> dict:
    """Process one player action and persist the resulting game state."""
    if not isinstance(player_input, str) or not player_input.strip():
        raise ValueError("player_input must contain non-whitespace text")
    saved = load_session(session_id)
    if saved is None:
        raise ValueError(f"No game found for session_id={session_id}")
    if saved.get("user_id") is not None and saved.get("user_id") != user_id:
        raise ValueError("This story belongs to another account")

    state = saved["state"]
    history = saved["history"]
    _ensure_state_shape(state)

    if state.get("game_over"):
        return {
            "session_id": session_id,
            "state": state,
            "scene": {
                "narration": (
                    "The story has reached its ending. Begin a new tale to continue "
                    "the journey."
                ),
                "choices_hint": [],
                "image_path": state.get("last_scene_image"),
                "ambient_path": state.get("ambient_background"),
                "game_over": True,
            },
            "chaos_triggered": False,
        }

    chaos_fragment = chaos.maybe_trigger_chaos(
        chaos_mode=state.get("chaos_mode", False),
        turn_count=state.get("turn_count", 0),
    )
    chaos_triggered = chaos_fragment is not None

    full_prompt = _assemble_prompt(
        state=state,
        recent_turns=_clip_recent_turns(history),
        player_input=player_input,
        chaos_fragment=chaos_fragment,
    )

    result = _generate_non_repeating_turn(full_prompt, state)
    result["narration"] = _sanitize_player_input_echo(result.get("narration", ""), player_input)
    scene = _apply_turn_result(state, result)
    _remember_turn(state, player_input, scene["narration"], scene["choices_hint"])

    history.append({"role": "player", "text": player_input})
    history.append({"role": "narrator", "text": scene["narration"]})
    state["turn_count"] += 1

    summarize_every = (
        CHAOS_MODE_SUMMARIZE_EVERY_N_TURNS
        if state.get("chaos_mode")
        else SUMMARIZE_EVERY_N_TURNS
    )

    if state["turn_count"] % summarize_every == 0:
        history_text = _join_text_blocks(*(
            f"{entry.get('role', 'unknown')}: {entry.get('text', '')}"
            for entry in history
        ))
        state["story_summary"] = llm_client.summarize(history_text)
        # Keep the last few turns so the model still sees recent context,
        # while the summary covers everything older. Trimming to only 2
        # entries caused the LLM to lose scene continuity.
        history = history[-HISTORY_KEEP_AFTER_SUMMARY:]

    if saved.get("user_id") is None:
        _GUEST_GAMES[session_id] = {"state": state, "history": history, "user_id": None}
    else:
        db.save_game(session_id, state, history, user_id=saved.get("user_id"))
    if user_id is not None and saved.get("user_id") is not None:
        db.record_story_usage(user_id, turn=True)

    return {
        "session_id": session_id,
        "state": state,
        "scene": scene,
        "chaos_triggered": chaos_triggered,
    }


def get_game(session_id: str) -> dict:
    """Return a saved session's current state and retained history."""
    saved = load_session(session_id)
    if saved is None:
        raise ValueError(f"No game found for session_id={session_id}")

    state = saved["state"]
    _ensure_state_shape(state)

    return {
        "session_id": session_id,
        "state": state,
        "history": saved["history"],
    }
