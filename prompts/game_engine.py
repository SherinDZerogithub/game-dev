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
RECENT_TURNS_LIMIT = 8
MIN_STAT_VALUE = 0
MAX_STAT_VALUE = 200
REPETITION_LOOKBACK = 6
REPETITION_NGRAM_SIZE = 4
MAX_REPAIR_ATTEMPTS = 2
FORBIDDEN_TEMPLATE_FRAGMENTS = (
    "the discovery changes what the next step can be",
    "it points toward",
)

# -----------------------------
# CURATED OPENING PRESETS (no candy/witch themes — elegant, interview-safe)
# -----------------------------
LORE_PRESETS = [
    {
        "id": "noir",
        "label": "Neon Noir",
        "icon": "city",
        "opening": (
            "Rain streaks the neon of a city that never sleeps. You are a private "
            "investigator standing over an unsolved case file at 2 a.m. when the "
            "phone rings with a voice from a past you tried to bury. Find out who "
            "is calling — and why now."
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
        "label": "The Lighthouse Keeper",
        "icon": "lighthouse",
        "opening": (
            "You arrive to take up the post of lighthouse keeper on a remote island. "
            "The previous keeper left no note — only a locked journal and a warning "
            "scratched into the door: do not light the beam after midnight."
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
        "label": "The Midnight Bakery",
        "icon": "moon",
        "opening": (
            "You inherit a tiny bakery that only opens between midnight and dawn. On "
            "your first night, a regular customer arrives who has been coming for "
            "thirty years — and who is not, exactly, alive. Hear their story."
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
    "narrative_context": {
        "location": "",
        "active_characters": [],
        "current_goal": "",
        "open_threads": [],
    },
}

# -----------------------------
# INTERNAL HELPERS
# -----------------------------
def _ensure_state_shape(state: dict) -> None:
    """Ensure old save files remain compatible when new fields are added."""
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

    if not isinstance(state.get("narrative_context"), dict):
        state["narrative_context"] = deepcopy(DEFAULT_STATE["narrative_context"])
    else:
        for key, value in DEFAULT_STATE["narrative_context"].items():
            state["narrative_context"].setdefault(key, deepcopy(value))


def _format_bullets(values: list) -> str:
    if not values:
        return "  - None"
    return "\n".join(f"  - {value}" for value in values)


def _assemble_prompt(
    state: dict,
    recent_turns: list,
    player_input: str,
    chaos_fragment: str | None,
) -> str:
    """Build the complete prompt sent to the LLM."""
    _ensure_state_shape(state)

    history_text = "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('text', '')}"
        for turn in recent_turns
    )
    if not history_text:
        history_text = "No previous turns."

    relationships = state.get("relationships", {})
    relationships_text = (
        "\n".join(f"  - {name}: {value}" for name, value in relationships.items())
        if relationships
        else "  - None"
    )

    narrative_context = state.get("narrative_context", {})
    recent_actions = state.get("action_history", [])[-6:]
    recent_narrations = state.get("narration_history", [])[-8:] # Up from -4
    active_characters = narrative_context.get("active_characters", [])
    open_threads = narrative_context.get("open_threads", [])

    chaos_text = (
        f"\n[TWIST EVENT]\n{chaos_fragment}\n"
        if chaos_fragment
        else ""
    )

    character_line = ""
    if state.get("character_class"):
        character_line = f"\nCharacter archetype: {state['character_class']}\n"
    difficulty_line = ""
    if state.get("difficulty"):
        difficulty_line = f"\nDifficulty pacing: {state['difficulty']}\n"

    # Add location context more prominently.
    location_text = narrative_context.get("location", "") or "Unknown"

    # Dynamic anti-looping reinforcement. This must be interpolated into the
    # returned prompt; the previous version built it but never included it.
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

    # Build a human-readable summary of already-discovered facts so the LLM
    # never re-narrates finding something the player already found.
    flags = state.get("flags", {})
    discovered_flags_lines = []
    for flag_key, flag_val in flags.items():
        if flag_val is True or flag_val == "discovered" or flag_val == "found":
            # Convert snake_case to readable label
            label = flag_key.replace("_", " ")
            discovered_flags_lines.append(label)
    discovered_flags_text = (
        "\n".join(f"  - {line}" for line in discovered_flags_lines)
        if discovered_flags_lines
        else "  - None"
    )

    return f"""
{SYSTEM_PROMPT}
{character_line}{difficulty_line}
[WORLD PREMISE / PLAYER-GIVEN PROMPT]
{state.get("world_prompt", "") or "Use the opening scenario and recent story as the world premise."}

[STORY SUMMARY]
{state.get("story_summary", "") or "No summary yet."}

[LOCATION]
{location_text}

[NARRATIVE CONTEXT]
Active characters:
{_format_bullets(active_characters)}
Current goal: {narrative_context.get("current_goal", "") or "None"}
Open story threads:
{_format_bullets(open_threads)}

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

[ALREADY DISCOVERED — DO NOT RE-NARRATE THESE]
{discovered_flags_text}
If any item above is listed, the player already knows it. Do not describe finding
or revealing it again. Instead, treat it as established background and build forward.

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

[RECENT NARRATIONS \u2014 DO NOT REPEAT OR PARAPHRASE THESE]
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


def _apply_turn_result(state: dict, result: dict) -> dict:
    """Apply an LLM turn result to state and build the scene response."""
    _ensure_state_shape(state)

    narration = str(result.get("narration", "")).strip()
    if not narration:
        narration = (
            "The scene changes after your last move. A new detail in the image is "
            "waiting to be understood, and the next choice will decide which part "
            "of it matters."
        )

    state_updates = result.get("state_updates")
    if isinstance(state_updates, dict):
        for key, value in state_updates.items():
            if key in {"stats", "flags", "relationships", "narrative_context"}:
                if isinstance(value, dict):
                    _apply_mapping_updates(state[key], value)
            elif key in state:
                state[key] = deepcopy(value)

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
    choices = result.get("choices_hint", result.get("choices", []))
    if not isinstance(choices, list):
        choices = []
    
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
    cleaned["narration"] = " ".join(kept).strip() or narration
    return cleaned

def _generate_non_repeating_turn(full_prompt: str, state: dict) -> dict:
    """Generate a turn and repair it when it repeats a recent structural pattern."""
    result = llm_client.generate_turn(full_prompt, temperature=0.85)
    if not _result_repeats_recent_story(result, state):
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
        result = llm_client.generate_turn(
            repair_prompt,
            temperature=min(1.25, 1.0 + attempt * 0.1),
        )
        if not _result_repeats_recent_story(result, state):
            return result

    # Never concatenate a stock ending. If every provider attempt repeats the
    # known bridge, remove only that complete template sentence and retain the
    # model's organic narration and state updates.
    return _remove_forbidden_template_sentences(result)


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


def new_game(
    opening_prompt: str,
    chaos_mode: bool = False,
    custom_state: dict | None = None,
    character_class: str = "",
    difficulty: str = "Balanced",
) -> dict:
    """Create, generate, persist, and return a new game session."""
    db.init_db()
    session_id = str(uuid.uuid4())
    state = deepcopy(DEFAULT_STATE)
    state["chaos_mode"] = chaos_mode
    state["world_prompt"] = opening_prompt.strip()
    state["character_class"] = character_class.strip() if isinstance(character_class, str) else ""
    state["difficulty"] = difficulty.strip() if isinstance(difficulty, str) else "Balanced"

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

    db.save_game(session_id, state, history)

    return {
        "session_id": session_id,
        "state": state,
        "scene": scene,
    }


def take_turn(session_id: str, player_input: str) -> dict:
    """Process one player action and persist the resulting game state."""
    saved = db.load_game(session_id)
    if saved is None:
        raise ValueError(f"No game found for session_id={session_id}")

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
        recent_turns=history[-RECENT_TURNS_LIMIT:],
        player_input=player_input,
        chaos_fragment=chaos_fragment,
    )

    result = _generate_non_repeating_turn(full_prompt, state)
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
        history_text = "\n".join(
            f"{entry.get('role', 'unknown')}: {entry.get('text', '')}"
            for entry in history
        )
        state["story_summary"] = llm_client.summarize(history_text)
        history = history[-2:]

    db.save_game(session_id, state, history)

    return {
        "session_id": session_id,
        "state": state,
        "scene": scene,
        "chaos_triggered": chaos_triggered,
    }


def get_game(session_id: str) -> dict:
    """Return a saved session's current state and retained history."""
    saved = db.load_game(session_id)
    if saved is None:
        raise ValueError(f"No game found for session_id={session_id}")

    state = saved["state"]
    _ensure_state_shape(state)

    return {
        "session_id": session_id,
        "state": state,
        "history": saved["history"],
    }
