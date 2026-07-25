"""
llm_client.py

Wraps the Google Gemini API (modern unified `google-genai` SDK) for narrative
text generation. Handles sending the assembled prompt and safely parsing the
structured JSON response the game engine expects.

Set GEMINI_API_KEY in .env. Get one free at:
https://aistudio.google.com/app/apikey
"""

import os
import json
import re
import random
import traceback
try:
    from google import genai
    from google.genai import types
except ImportError:  # Allows offline tests and graceful fallback without the SDK.
    genai = None
    types = None

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_client = None


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "too many requests",
            "retry in",
            "429",
        )
    )


def _extract_player_action(full_prompt: str) -> str:
    match = re.search(
        r"^\[PLAYER INPUT\]\s*\n?(.*?)(?=\n\[[A-Z]|\Z)",
        full_prompt, re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _clean_image_prompt(text: str) -> str:
    words = text.split()
    if len(words) <= 24:
        return text
    return " ".join(words[:24])


def _get_client():
    global _client
    if genai is None:
        raise RuntimeError("google-genai is not installed. Install it with: pip install google-genai")
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create one at "
                "https://aistudio.google.com/app/apikey and set it in .env."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _extract_json(raw_text: str) -> dict:
    """Gemini sometimes wraps JSON in markdown fences or adds stray text."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from model output:\n{raw_text}")


def generate_turn(full_prompt: str, temperature: float = 0.9, last_consequences: list | None = None) -> dict:
    """Send the assembled prompt to Gemini and return the parsed structured dict."""
    try:
        client = _get_client()
    except RuntimeError as exc:
        print(f"[llm_client] Gemini unavailable, using fallback turn: {exc}")
        return _fallback_turn(full_prompt, last_consequences)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=(
                full_prompt
                + "\n\n[RESPONSE-LEVEL ANTI-TEMPLATE RULE]\n"
                + "Do not use fixed bridge sentences such as 'The discovery changes what "
                + "the next step can be' or 'it points toward'. End with a concrete "
                + "physical, social, or environmental consequence unique to this scene."
            ),
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        result = _extract_json(response.text)
        if not isinstance(result, dict):
            raise ValueError("Model response must be a JSON object")
        if not str(result.get("narration", "")).strip():
            fallback = _fallback_turn(full_prompt, last_consequences)
            result["narration"] = fallback["narration"]
            if not result.get("image_prompt"):
                result["image_prompt"] = fallback["image_prompt"]
            if not result.get("choices_hint") and not result.get("choices"):
                result["choices_hint"] = fallback["choices_hint"]
        choices = result.get("choices_hint", result.get("choices", []))
        # Ensure we have exactly 3 dynamic choices
        if not isinstance(choices, list) or len(choices) < 3:
            choices = _generate_dynamic_choices(full_prompt, result.get("narration", ""))
        result["choices_hint"] = choices[:3] if isinstance(choices, list) else []
        return result
    except Exception as exc:
        reason = "quota/provider error" if _is_quota_error(exc) else "Gemini request failed"
        print(f"[llm_client] {reason}, using fallback turn: {exc}")
        traceback.print_exc()
        return _fallback_turn(full_prompt, last_consequences)


def _section(full_prompt: str, name: str) -> str:
    pattern = rf"\[{re.escape(name)}\]\s*(.*?)(?=\n\[[A-Z][A-Z \u2014-]*\]|\Z)"
    match = re.search(pattern, full_prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


def _fsm_field(full_prompt: str, field: str) -> str:
    """Extract a labelled field from the FSM SNAPSHOT block, e.g. CURRENT LOCATION."""
    pattern = rf"^{re.escape(field)}:\s*(.+?)(?=\n[A-Z]|\Z)"
    match = re.search(pattern, full_prompt, re.MULTILINE | re.DOTALL)
    if match:
        # Return only the first line (the value), not the explanatory bullet beneath it
        value = match.group(1).strip()
        return value.splitlines()[0].strip()
    return ""


def _extract_context_from_prompt(full_prompt: str) -> dict:
    """Extract the engine's labelled prompt sections for API fallbacks."""
    recent_turns = _section(full_prompt, "RECENT TURNS")
    recent_narrations = _section(
        full_prompt, "RECENT NARRATIONS \u2014 DO NOT REPEAT OR PARAPHRASE THESE"
    )
    # The prompt now uses "RECENT NARRATIONS — DO NOT REPEAT" (en-dash variant too)
    if not recent_narrations:
        recent_narrations = _section(
            full_prompt, "RECENT NARRATIONS — DO NOT REPEAT OR PARAPHRASE THESE"
        )

    narrator_events = []
    for line in recent_turns.splitlines():
        if line.lower().startswith("narrator:"):
            narrator_events.append(line.split(":", 1)[1].strip())

    action_lines = []
    for line in _section(full_prompt, "RECENT PLAYER ACTIONS").splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            value = stripped[1:].strip()
            if value.lower() != "none":
                action_lines.append(value)

    # Location is now in the FSM block under "CURRENT LOCATION:" rather than
    # a standalone [LOCATION] section.
    location = (
        _section(full_prompt, "LOCATION")
        or _fsm_field(full_prompt, "CURRENT LOCATION")
    )

    # Already-discovered facts live inside the FSM block now.
    already_discovered = (
        _section(full_prompt, "ALREADY DISCOVERED \u2014 DO NOT RE-NARRATE THESE")
        or _section(full_prompt, "ALREADY DISCOVERED (FLAG GUARD)")
    )

    return {
        "world_prompt": _section(full_prompt, "WORLD PREMISE / PLAYER-GIVEN PROMPT"),
        "story_summary": _section(full_prompt, "STORY SUMMARY"),
        "recent_turns": recent_turns,
        "recent_narrations": recent_narrations,
        "recent_actions": action_lines,
        "last_narration": narrator_events[-1] if narrator_events else "",
        "inventory": _section(full_prompt, "INVENTORY"),
        "flags": _section(full_prompt, "FLAGS"),
        "location": location,
        "active_characters": _section(full_prompt, "ACTIVE CHARACTERS PRESENT"),
        "last_choices": _section(full_prompt, "LAST SUGGESTIONS"),
        "already_discovered": already_discovered,
    }


def _seed_for(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text or ""))


def _romance_consequence(context: dict, action_lower: str, variant: int) -> tuple[str, str, list[str]]:
    """Choose a romance event that belongs to the current location.

    Returns (event, place, pool) — the pool is the exact list the event was
    drawn from, so callers can correctly rotate away from an already-used
    event instead of silently falling back to an empty list.
    """
    location = context.get("location", "").lower()
    moves_to_station = any(word in action_lower for word in (
        "station", "train", "platform", "railway", "leave", "head to", "go to"
    ))
    at_station = any(word in location for word in ("station", "platform", "railway"))
    if at_station or moves_to_station:
        events = [
            "the departure board flickers to a name that was removed years ago",
            "a familiar figure waits beneath the last platform lamp, holding the missing half of the letter",
            "the Vienna train is already breathing at the platform, though dawn is still hours away",
            "a porter recognizes the handwriting on your letter and goes pale",
            "the station clock is frozen at the exact hour named in the letter",
        ]
        return events[variant % len(events)], "the last platform", events

    if any(word in action_lower for word in ("margin", "symbol", "letter", "envelope", "seal")):
        events = [
            "the sealed letter reveals a confession written in the margins after the sender disappeared",
            "the wax seal, held to the light, hides a second, smaller message beneath it",
        ]
        return events[variant % len(events)], context.get("location") or "the bookshop doorway", events

    events = [
        "a brass train token is wedged beneath the threshold, stamped with tomorrow's date",
        "the loose doorplate lifts in the wind, revealing a thread of blue paper from the letter",
        "fresh saltwater darkens the doorstep in the shape of a single footprint pointing inland",
        "a pressed flower slips from between two books, matching the one folded into the letter",
        "the bell above the door rings once on its own, with no wind to explain it",
    ]
    return events[variant % len(events)], context.get("location") or "the bookshop doorway", events


# ---------------------------------------------------------------------------
# INTENT-ACTION BINDING
#
# A question about *why* something happened (motive, history, backstory) is a
# fundamentally different kind of action than physically searching a room. It
# must be answered from a lore/reveal pool, never from the physical-item pool
# — otherwise "take a risk to learn why..." gets matched by the word "take"
# and re-spawns a physical object instead of answering the question asked.
# ---------------------------------------------------------------------------
LORE_INTENT_PATTERNS = (
    "why", "reason", "understand why", "learn why", "history of", "the past",
    "meaning of", "backstory", "motive", "explain", "what happened to",
    "risk to learn", "find out why", "who they were", "what changed them",
)


def _is_lore_question(action_lower: str) -> bool:
    """True when the player is asking about motive/backstory rather than a physical object."""
    return any(pattern in action_lower for pattern in LORE_INTENT_PATTERNS)


LORE_REVEAL_SETS: dict[str, list[str]] = {
    "noir": [
        "the caller waited because they were the one who filed the case away ten years ago, and only now found the nerve to reopen it",
        "the silence was never guilt — naming the caller sooner would have gotten you both killed",
        "an old report surfaces the real reason: the case was never closed, only hidden from you specifically",
    ],
    "romance": [
        "the sender waited ten years because your last reply never arrived — it was intercepted before it reached Vienna",
        "the silence was a promise: to wait until it was finally safe to write again, and that day only just came",
        "the ten years make sense the moment you realize the sender was waiting for you to stop searching for someone else",
    ],
    "lighthouse": [
        "the previous keeper vanished the night they realized the beam was never meant to guide ships in — it called them out",
        "the warning about midnight exists because that is the hour the strait remembers every ship it has ever taken",
        "the journal explains it plainly: the light was never for sailors. It was for something coming home",
    ],
    "scifi": [
        "the silent galaxy was never empty; it went quiet the moment it heard a signal identical to your own voice, decades ago",
        "mission control's twelve-hour delay was deliberate — someone wanted this signal to reach you and no one else first",
        "the impossible signal is a recording of this exact conversation, sent from a version of the relay not yet built",
    ],
    "fantasy": [
        "the map exists because your grandmother walked that road once and chose to erase it from every other chart in the kingdom",
        "the stone hums because it remembers a promise your grandmother made to whatever waits at the road's end",
        "the road was hidden because the last person who found it was never seen leaving — only arriving, again and again",
    ],
    "bakery": [
        "the midnight customer keeps returning because the last thing said between you was never finished, thirty years ago",
        "the bakery only opens at midnight because that is the one hour the customer is still allowed to remember being alive",
        "the recipe they keep asking for was the last thing anyone made for them — making it again is the only way they can leave in peace",
    ],
}


def _lore_consequence(profile: dict, variant: int) -> tuple[str, list[str]]:
    """Return a thematic, motive-level reveal instead of a physical-object spawn."""
    pool = LORE_REVEAL_SETS.get(profile["kind"])
    if not pool:
        pool = [
            f"the truth behind {profile['thread']} finally comes into focus, recontextualizing everything before this",
            f"a memory or record surfaces that explains {profile['thread']} in a way you did not expect",
        ]
    return pool[variant % len(pool)], pool


def _consequence_transition(action_lower: str, consequence: str, kind: str, turn_offset: int = 0) -> str:
    """Pick a sentence that leads into the consequence. Purely diegetic — no
    meta-commentary like the literal word 'consequence' should appear here."""
    transitions = [
        f"The detail answers your search: {consequence}",
        f"Your attention catches the change. {consequence.capitalize()}",
        f"A small movement breaks the stillness—{consequence}",
        f"Something shifts at the edge of your vision: {consequence}",
        f"The room itself seems to answer you: {consequence}",
        f"You almost miss it, then can't unsee it: {consequence}",
        f"It takes a second look to be sure, but there it is: {consequence}",
    ]
    index = (_seed_for(action_lower + consequence + kind) + turn_offset) % len(transitions)
    return transitions[index]


def _progression_close(action: str, consequence: str, profile: dict, variant: int, turn_offset: int = 0) -> str:
    """Close a fallback turn with a concrete changed situation, not a fixed template.

    None of these lines use meta language like "the consequence" — they describe
    the in-world state change directly, so the narration never announces its own
    mechanics to the player.
    """
    object_name = profile.get("object", "the clue")
    place = profile.get("place", "the current scene")
    thread = profile.get("thread", "the unresolved mystery")
    closers = [
        f"Now {object_name} is no longer background detail; it demands an immediate decision.",
        f"{place.capitalize()} feels different now — danger and opportunity both in plain sight.",
        f"Whatever happens next, {thread} cannot wait any longer.",
        f"Staying passive here would let the situation move on without you.",
        f"The scene has shifted around you, and the next move must deal with {thread} directly.",
        f"A physical trace remains at {place}, giving you something concrete to act on.",
        f"There is no going back to how {place} looked a moment ago.",
        f"You will have to decide what to do about {object_name} before it changes again.",
        f"Whoever — or whatever — caused this is still close enough to matter.",
    ]
    seed_text = f"{action}|{consequence}|{object_name}|{place}|{thread}|{variant}"
    index = (_seed_for(seed_text) + turn_offset) % len(closers)
    return closers[index]



def _is_passive_action(action_lower: str) -> bool:
    """Actions that spend time without investigating a new story lead."""
    return any(re.search(rf"\b{re.escape(word)}\b", action_lower) for word in (
        "sleep", "rest", "wait", "camp", "nap", "sit", "stay", "do nothing",
    ))


def _choice_repeats_action(choice: str, action_lower: str) -> bool:
    """Reject suggestions that would make the player repeat the just-finished beat."""
    choice_lower = choice.lower()
    if not action_lower:
        return False

    # A kitchen/recipe search is especially easy to loop because the bakery
    # location template always contains that option.
    if ("kitchen" in action_lower or "recipe" in action_lower) and (
        "kitchen" in choice_lower or "recipe" in choice_lower
    ):
        return True

    action_words = set(re.findall(r"[a-z0-9]+", action_lower))
    choice_words = set(re.findall(r"[a-z0-9]+", choice_lower))
    verbs = {"search", "look", "examine", "study", "inspect", "investigate"}
    if action_words & verbs and choice_words & verbs:
        # Avoid variants such as “search more thoroughly in the same spot”.
        same_subject = action_words & choice_words - {"the", "a", "for", "in", "at", "to"}
        if len(same_subject) >= 2 or "same spot" in choice_lower:
            return True

    return False


def _story_profile(context: dict, narration: str = "") -> dict:
    """Turn the premise and current scene into small, reusable story signals.

    This keeps the offline/quota fallback from collapsing every premise into the
    same generic adventure. The hosted model still has the full prompt; these
    signals are specifically for graceful local fallback behaviour.
    """
    text = " ".join(
        str(context.get(key, ""))
        for key in ("world_prompt", "story_summary", "last_narration", "location", "active_characters")
    ) + " " + str(narration or "")
    lower = text.lower()
    profiles = [
        ("noir", ("detective", "investigator", "neon", "case file", "crime", "rain"),
         "the case file", "the rain-slick street", "the caller's identity"),
        ("romance", ("letter", "vienna", "bookshop", "train", "coastal town"),
         "the sealed letter", "the bookshop doorway", "why the sender waited ten years"),
        ("lighthouse", ("lighthouse", "keeper", "island", "beam", "journal", "coast"),
         "the locked journal", "the lighthouse stairwell", "what happened to the previous keeper"),
        ("scifi", ("space", "station", "signal", "galaxy", "mission control", "relay"),
         "the impossible signal", "the relay station", "who is transmitting from the silent galaxy"),
        ("fantasy", ("cartographer", "kingdom", "map", "forest", "stone", "magic"),
         "the humming stone key", "the old forest road", "where the impossible map leads"),
        ("bakery", ("bakery", "bread", "dawn", "midnight", "customer", "alive"),
         "the midnight customer's story", "the bakery kitchen", "why the regular returns after death"),
    ]
    for name, signals, object_name, place, thread in profiles:
        if any(re.search(rf"\b{re.escape(signal)}\b", lower) for signal in signals):
            return {"kind": name, "object": object_name, "place": place, "thread": thread}
    return {
        "kind": "custom", "object": "the latest clue", "place": context.get("location") or "the current scene",
        "thread": "what the new discovery means",
    }


def _active_character_names(context: dict) -> list[str]:
    """Normalize prompt-formatted or state-formatted active characters."""
    value = context.get("active_characters", "")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    names = []
    for line in str(value or "").splitlines():
        character = line.strip().lstrip("-*").strip()
        if character and character.lower() != "none":
            names.append(character)
    return names


def _dialogue_subject(context: dict, profile: dict) -> str | None:
    """Return a character that has actually been established in the scene."""
    active_characters = _active_character_names(context)
    if active_characters:
        return active_characters[0]

    # The curated lighthouse opening establishes this role even when the
    # fallback is called before the first narrative-context update is saved.
    if profile.get("kind") == "lighthouse":
        return "Mara Venn, the boat captain who brought you ashore"
    if profile.get("kind") == "bakery":
        return "the midnight customer"
    return None


def _generate_dynamic_choices(full_prompt: str, narration: str) -> list:
    """Generate 3 dynamic, context-aware choices based on the current scene."""
    context = _extract_context_from_prompt(full_prompt)
    raw_action = _extract_player_action(full_prompt)
    action = re.sub(r"^\(Game start\)\s*", "", raw_action, flags=re.IGNORECASE).strip()
    action = action or "take stock of your surroundings"
    action_lower = action.lower()
    
    # Build a rich set of possible actions based on context
    choices_pool = []
    
    profile = _story_profile(context, narration)
    current_choices = context.get("last_choices", "")
    already_discovered = context.get("already_discovered", "")
    story_text = " ".join(str(context.get(key, "")) for key in (
        "world_prompt", "story_summary", "recent_turns", "recent_narrations", "last_narration"
    )).lower()

    # Location-based choices
    location = context.get("location", "").lower()
    if "city" in location or "street" in location or "alley" in location:
        choices_pool.extend([
            "search the nearby alleys for a hidden entrance",
            "ask a passerby about the sealed letter",
            "look for a vantage point to survey the area",
            "follow the sound of music from a nearby bar",
            "check the notice board at the town square"
        ])
    elif "bookshop" in location or "library" in location or "study" in location:
        choices_pool.extend([
            "examine the bookshelves for a hidden compartment",
            "read the oldest journal on the desk",
            "look for a secret passage behind a bookcase",
            "search for a map in the archives",
            "study the strange symbols in the margins"
        ])
    elif "lighthouse" in location or "island" in location or "coast" in location:
        choices_pool.extend([
            "climb to the top of the lighthouse",
            "examine the locked journal left behind",
            "search the shore for clues",
            "check the weather station",
            "look for a boat to leave the island"
        ])
    elif "space" in location or "station" in location or "relay" in location:
        choices_pool.extend([
            "decrypt the incoming signal",
            "check the station's logs",
            "attempt to contact mission control",
            "run a diagnostic on the relay system",
            "explore the emergency escape pods"
        ])
    elif "forest" in location or "woods" in location or "map" in location:
        choices_pool.extend([
            "follow the path that appears on the map",
            "search for the grandmother's old hideout",
            "study the stones that hum",
            "look for travelers who know the road",
            "set up camp and wait for dawn"
        ])
    elif "bakery" in location or "shop" in location or "market" in location:
        choices_pool.extend([
            "question the mysterious regular customer",
            "search the kitchen for hidden recipes",
            "check the back room for a secret",
            "ask about the history of the place",
            "offer a pastry to a stranger for information",
            "leave the kitchen and follow the sound outside",
            "wait by the front door for the midnight customer"
        ])
    else:
        # Generic but varied choices based on story elements
        if "letter" in story_text:
            choices_pool.extend([
                "read the sealed letter carefully",
                "examine the envelope for hidden marks",
                "search for the sender's identity",
                "pursue the new clue",
                "follow the letter's instructions",
                "question the stranger who sent it"
            ])
        elif "clue" in narration.lower() or "mystery" in context.get("story_summary", "").lower():
            choices_pool.extend([
                "investigate the clue thoroughly",
                "question the person who knows more",
                "search for related evidence",
                "track down the source of the clue",
                "piece together what you know"
            ])
        else:
            choices_pool.extend([
                "explore the area more carefully",
                "talk to someone nearby",
                f"investigate {profile['object']}",
                f"trace the mystery of {profile['thread']}",
                f"follow the newest lead from {profile['place']}"
            ])
    
    # Add action-specific choices
    if any(word in action_lower for word in ("talk", "ask", "question", "speak")):
        choices_pool.extend([
            "ask a deeper, more personal question",
            "search for someone else to speak with",
            "examine the person's belongings",
            "look for a letter or note they dropped",
            "follow them discreetly"
        ])
    elif any(word in action_lower for word in ("search", "look", "examine", "study")):
        choices_pool.extend([
            "ask someone about what you found",
            "try to use what you discovered",
            "look for a hidden compartment",
            "check the opposite direction",
            "leave this place and follow the newest lead"
        ])
    elif any(word in action_lower for word in ("take", "grab", "open", "use")):
        choices_pool.extend([
            "examine the item more closely",
            "ask someone about the item's origin",
            "try to use the item in a different way",
            "search for where the item came from",
            "hide the item and wait"
        ])
    elif any(word in action_lower for word in ("follow", "track", "pursue")):
        choices_pool.extend([
            "follow the trail more cautiously",
            "look for tracks or signs left behind",
            "ask others if they saw someone pass",
            "take a shortcut to intercept",
            "wait and see if they return"
        ])
    
    # Add character-based choices
    active_chars = context.get("active_characters", "")
    if active_chars and "None" not in active_chars:
        choices_pool.extend([
            "ask about the character's history",
            "offer help in exchange for information",
            "watch the character from a distance",
            "confront the character directly",
            "search for evidence they left behind"
        ])
    
    # Add inventory-based choices
    inventory = context.get("inventory", "")
    if inventory and "None" not in inventory:
        items = [i.strip() for i in inventory.split(",") if i.strip()]
        for item in items[:2]:
            choices_pool.extend([
                f"examine the {item} more closely",
                f"use the {item} in a new way",
                f"ask someone about the {item}",
                f"hide the {item} safely"
            ])
    
    # Add flag-based choices
    flags = context.get("flags", "")
    if flags and "None" not in flags:
        if "found_clue" in flags:
            choices_pool.extend([
                "follow the clue you found",
                "search for more clues like it",
                "tell someone about your discovery"
            ])

        if "met_stranger" in flags:
            choices_pool.extend([
                "find the stranger again",
                "search for the stranger's past",
                "watch for the stranger's return"
            ])

    # The current scene is more authoritative than generic location keywords.
    # These options make the three cards visibly respond to what just happened.
    choices_pool.extend([
        f"inspect the consequence near {profile['place']}",
        f"ask what {profile['object']} is hiding",
        f"take a risk to learn {profile['thread']}",
    ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_choices = []
    for choice in choices_pool:
        # Normalize for comparison
        norm = " ".join(re.findall(r"[a-z0-9]+", choice.lower()))
        if (norm and norm not in seen and norm not in current_choices
                and not _choice_repeats_action(choice, action_lower)
                and not _consequence_already_known(choice, already_discovered)):
            seen.add(norm)
            unique_choices.append(choice)
    
    # If we still don't have enough, add more generic but varied choices
    if len(unique_choices) < 3:
        generic_fallbacks = [
            "examine the environment for overlooked details",
            "approach someone who seems to know something",
            "consider the clues you have gathered",
            "look for a way to gain an advantage",
            "search for a hidden entrance or exit",
            "ask about the history of this place",
            "try a completely unexpected approach"
        ]
        for choice in generic_fallbacks:
            norm = " ".join(re.findall(r"[a-z0-9]+", choice.lower()))
            if norm not in seen:
                unique_choices.append(choice)
                if len(unique_choices) >= 3:
                    break

    # A passive turn should open the scene back up rather than offering more
    # versions of waiting, camping, or staying put.
    if _is_passive_action(action_lower):
        unique_choices = [
            choice for choice in unique_choices
            if not any(word in choice.lower() for word in ("wait", "camp", "stay", "rest", "sleep"))
        ]
        for choice in (
            f"leave {profile['place']} and follow the newest lead",
            f"look for the person connected to {profile['thread']}",
            f"use the change near {profile['object']} to move forward",
        ):
            norm = " ".join(re.findall(r"[a-z0-9]+", choice.lower()))
            if norm not in seen:
                unique_choices.append(choice)
                seen.add(norm)
            if len(unique_choices) >= 3:
                break
    
    # If still less than 3, use the most basic fallbacks
    while len(unique_choices) < 3:
        final_fallbacks = [
            "explore your surroundings",
            "talk to someone nearby", 
            "look for a useful object"
        ]
        for choice in final_fallbacks:
            norm = " ".join(re.findall(r"[a-z0-9]+", choice.lower()))
            if norm not in seen:
                unique_choices.append(choice)
                if len(unique_choices) >= 3:
                    break

    # Once the player has searched the kitchen/recipes, keep at least one card
    # that changes location or follows the new consequence. Rotation below must
    # not hide every escape route behind three more in-room suggestions.
    if "kitchen" in action_lower or "recipe" in action_lower:
        unique_choices = [
            choice for choice in unique_choices
            if "kitchen" not in choice.lower()
            and "recipe" not in choice.lower()
            and "wait by the front door" not in choice.lower()
        ]
        escape_choice = "leave the kitchen and follow the newest lead"
        if escape_choice not in unique_choices:
            unique_choices.insert(0, escape_choice)
        for choice in (
            "ask the midnight customer what changed",
            "follow the sound beyond the bakery",
            "check the back room and keep moving",
        ):
            if len(unique_choices) >= 3:
                break
            if choice not in unique_choices:
                unique_choices.append(choice)

    # Keep the strongest premise-specific lead visible while still rotating the
    # other cards by scene text.
    if "letter" in story_text and "pursue the new clue" in unique_choices:
        unique_choices.remove("pursue the new clue")
        unique_choices.insert(0, "pursue the new clue")

    # Rotate the pool by the scene text so repeated fallback calls do not return
    # the same first three cards forever.
    if len(unique_choices) > 3:
        offset = _seed_for(narration + action + profile["kind"]) % len(unique_choices)
        if unique_choices[0] == "pursue the new clue":
            offset = 0
        unique_choices = unique_choices[offset:] + unique_choices[:offset]

    if ("kitchen" in action_lower or "recipe" in action_lower) and not any(
        "leave" in choice.lower() or "follow" in choice.lower()
        for choice in unique_choices[:3]
    ):
        unique_choices[2] = "leave the kitchen and follow the newest lead"
    return unique_choices[:3]


def _consequence_already_known(consequence: str, already_discovered: str) -> bool:
    """Return True if the consequence describes something already flagged as found."""
    if not already_discovered or already_discovered.strip().lower() in ("", "none", "- none"):
        return False
    # Tokenise both strings and check for 4+ shared word overlap.
    import re as _re
    c_words = set(_re.findall(r"[a-z0-9]+", consequence.lower()))
    for line in already_discovered.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line or line.lower() == "none":
            continue
        d_words = set(_re.findall(r"[a-z0-9]+", line.lower()))
        if len(c_words & d_words) >= 4:
            return True
    return False


def _fallback_turn(full_prompt: str, last_consequences: list | None = None) -> dict:
    """Generic, prompt-aware fallback used when Gemini is unavailable or over quota."""
    context = _extract_context_from_prompt(full_prompt)
    raw_action = _extract_player_action(full_prompt)
    action = re.sub(r"^\(Game start\)\s*", "", raw_action, flags=re.IGNORECASE).strip()
    action = re.sub(r"^(?:i|we|you)\s+(?=\S)", "", action, flags=re.IGNORECASE).strip()
    action = action or "take stock of your surroundings"
    action_lower = action.lower()

    previous_count = sum(
        1 for old in context["recent_actions"]
        if " ".join(re.findall(r"[a-z0-9]+", old.lower()))
        == " ".join(re.findall(r"[a-z0-9]+", action_lower))
    )
    # turn_offset grows by one every time the fallback engine is invoked (the
    # ring buffer is appended to on every fallback turn), giving a reliable,
    # ever-increasing rotation signal even when the action text repeats and
    # the hash-based variant below would otherwise collide.
    turn_offset = len(last_consequences or [])
    variant = (_seed_for(action + context["last_narration"]) + previous_count + turn_offset) % 7
    is_lore_question = _is_lore_question(action_lower)

    if raw_action.lower().startswith("(game start)"):
        premise = context["world_prompt"] or action
        profile = _story_profile(context)
        opening_signals = [
            f"a fresh sign points toward {profile['object']}",
            f"someone nearby reacts when you mention {profile['thread']}",
            f"a hidden detail appears at {profile['place']}",
            f"{profile['object'].capitalize()} changes in a way no ordinary object should",
            f"a stranger arrives carrying news about {profile['thread']}",
            f"a reflection reveals the next route through {profile['place']}",
            f"a quiet warning is scratched beside {profile['object']}"
        ]
        signal = opening_signals[variant % len(opening_signals)]
        opening_companion = ""
        if profile["kind"] == "lighthouse" and "Mara Venn" not in premise:
            opening_companion = (
                " Mara Venn, the boat captain who brought you ashore, remains in the "
                "lantern room doorway, watching the journal. "
            )
        narration = (
            f"{premise.strip()} The scene gathers around one wrong detail: {signal}. "
            f"{opening_companion}"
            "Whatever brought you here is already moving, quietly and deliberately. "
            "You can read the clue before it disappears, approach the person who seems "
            "to know more than they should, or walk straight toward the place the clue "
            "is pointing."
        )
        choices = [
            f"examine {profile['object']} closely",
            f"search {profile['place']} for the hidden detail",
            f"follow the lead about {profile['thread']}",
        ]
        image_prompt = premise[:180]
        location_update = profile["place"]
        thread_update = profile["thread"]
        active_characters_update = (
            ["Mara Venn (boat captain)"]
            if profile["kind"] == "lighthouse"
            else []
        )
    else:
        profile = _story_profile(context)
        repeated_note = (
            "You try the same approach again, but the world does not reset. This time, "
            if previous_count else ""
        )

        # Location-specific pools are checked FIRST so that moving to a bar,
        # alley, or other new room never loops back to a street/case-file clue.
        current_location = context.get("location", "").lower()
        location_consequence_map: dict[str, list[str]] = {
            "bar": [
                "the bartender slides a folded note down the counter without meeting your eyes",
                "a patron in the corner booth recognises your face and ducks behind a newspaper",
                "the jukebox skips and lands on a song the victim used to request every Friday",
            ],
            "alley": [
                "a fire-escape ladder has been freshly forced, its rust scraped bright silver",
                "a witness crouches behind the dumpster, clutching a matchbook from the crime scene",
                "footprints in the puddles lead toward a bricked-up door that is slightly ajar",
            ],
            "diner": [
                "a server quietly passes you a napkin with a name and a room number on it",
                "the cook shuts the kitchen window the moment you enter, locking eyes with you first",
                "someone has circled tonight's date in the newspaper left on your stool",
            ],
            "office": [
                "a desk drawer has been searched recently — the pens are all pointing the same way",
                "the phone log shows a call placed to a number that was disconnected two years ago",
                "a framed photo on the shelf has been turned face-down and recently picked back up",
            ],
            "rooftop": [
                "a rope tied to the parapet is still damp — someone used it to descend tonight",
                "a second set of fresh boot-prints crosses the gravel and stops at the edge",
                "a spent shell casing rocks in the wind, balanced on the ledge",
            ],
            "hotel": [
                "the room-service tray outside the next door holds two untouched glasses",
                "a key card slides under your door — no number, no name, only a floor: three",
                "the hallway mirror reflects a figure at the far end who is not there when you turn",
            ],
            "lantern": [
                "the old logbook falls open to an entry the keeper never finished",
                "the lenses magnify a distant ship that was never on the charts",
                "frost forms on the glass in the shape of a human hand pressed from inside",
            ],
            "relay": [
                "a secondary terminal boots itself and begins displaying coordinates",
                "the backup beacon fires once, then goes silent before the timestamp registers",
                "a handwritten note taped inside the panel reads: do not answer the second signal",
            ],
        }

        # Match current location to a specific pool.
        location_consequences: list[str] | None = None
        for loc_key, loc_pool in location_consequence_map.items():
            if loc_key in current_location:
                location_consequences = loc_pool
                break

        consequence_sets = {
            "noir": [
                "a witness steps from a doorway and says the caller is watching this street",
                "a passing headlight exposes a blood-marked clue beneath the rain",
                "the phone in your pocket buzzes — a second message from the blocked number",
            ],
            "romance": [],  # selected from the current location below
            "lighthouse": [
                "the journal opens to a tide chart marked with tomorrow's date",
                "the beam turns by itself and illuminates a boat where no boat should be",
                "saltwater rises through the stairwell, carrying the previous keeper's key",
            ],
            "scifi": [
                "the signal answers with a precise copy of your own voice",
                "the relay maps a second station hidden inside the dead transmission",
                "the station loses artificial gravity just as the message becomes readable",
            ],
            "fantasy": [
                "the stone key warms and redraws one road on the impossible map",
                "the forest rearranges its trees, opening a path beneath blue fireflies",
                "a royal seal appears in the soil, buried beside footprints made by no human",
            ],
            "bakery": [
                "the oven rings though nothing is baking, and a warm loaf bears your name",
                "the midnight customer leaves a recipe written in ink that is still wet",
                "the back room door opens onto the bakery as it looked thirty years ago",
            ],
        }
        # Intent-action binding: a "why"/backstory/motive question is answered
        # from the lore-reveal pool, never the physical-item pool. This check
        # runs FIRST because otherwise phrases like "take a risk to learn why
        # ..." get matched by the word "take" further down and re-spawn a
        # physical object instead of answering the question that was asked.
        if is_lore_question:
            consequence, _active_pool = _lore_consequence(profile, variant)
            consequence_place = current_location or profile["place"]
        elif profile["kind"] == "romance":
            consequence, consequence_place, _active_pool = _romance_consequence(context, action_lower, variant)
        elif location_consequences:
            # Location-specific pool wins: this prevents the bar/alley loop.
            consequence = location_consequences[variant % len(location_consequences)]
            consequence_place = current_location or profile["place"]
            _active_pool = location_consequences
        else:
            default_pool = [
                f"a concealed detail near {profile['place']} points toward {profile['thread']}",
                f"a witness appears with one fact about {profile['object']} they should not know",
                f"the scene changes around {profile['object']}, exposing both danger and a useful clue",
            ]
            # NOTE: use `or` rather than dict.get's default, since some kinds
            # (e.g. "romance") map to an intentionally empty list in
            # consequence_sets — dict.get's default only applies when the key
            # is *missing*, not when its value is falsy, which previously
            # left _active_pool empty and silently disabled the anti-repeat
            # rotation guards below for those kinds.
            consequences = consequence_sets.get(profile["kind"]) or default_pool
            consequence = consequences[variant % len(consequences)]
            consequence_place = profile["place"]
            _active_pool = consequences

        # Flag guard: if this consequence re-describes something the player
        # already discovered, rotate to the next entry in whichever pool was used.
        already_discovered = context.get("already_discovered", "")
        if _consequence_already_known(consequence, already_discovered):
            for _offset in range(1, len(_active_pool)):
                _candidate = _active_pool[(variant + _offset) % len(_active_pool)]
                if not _consequence_already_known(_candidate, already_discovered):
                    consequence = _candidate
                    break

        # Ring-buffer guard: rotate past any consequence already used in recent
        # fallback turns so we never repeat the same sentence back-to-back.
        _used = set(last_consequences or [])
        if consequence in _used and len(_active_pool) > 1:
            for _offset in range(1, len(_active_pool)):
                _candidate = _active_pool[(variant + _offset) % len(_active_pool)]
                if _candidate not in _used and not _consequence_already_known(
                    _candidate, already_discovered
                ):
                    consequence = _candidate
                    break

        clean_action = action.rstrip(".!?")

        # Build narration around the player's action and a concrete world change.
        # No fixed "discovery changes..." bridge is used anywhere.
        close = _progression_close(clean_action, consequence, profile, variant, turn_offset)

        if is_lore_question:
            # Dedicated thematic-reveal template. Checked before the movement/
            # dialogue/item templates below so a phrase like "take a risk to
            # learn why..." is bound to the reveal, not the item template
            # (which would otherwise match on the word "take").
            narration = (
                f"{repeated_note}You {clean_action}. "
                f"{consequence[0].upper()}{consequence[1:]}. "
                f"{close}"
            )
        elif any(re.search(rf"\b{re.escape(word)}\b", action_lower) for word in (
            "back", "return", "path", "go", "head", "walk", "follow", "leave"
        )):
            dest = consequence_place if consequence_place != profile["place"] else (
                current_location or profile["place"]
            )
            narration = (
                f"{repeated_note}You {clean_action}. "
                f"The place you left recedes; {dest} fills the frame instead. "
                f"Here, {consequence}. "
                f"{close}"
            )
        elif any(re.search(rf"\b{re.escape(word)}\b", action_lower) for word in (
            "talk", "ask", "call", "say", "tell", "speak", "answer"
        )):
            subject = _dialogue_subject(context, profile)
            if subject:
                narration = (
                    f"{repeated_note}Your words land harder than expected. {subject} "
                    f"hesitates, then reveals that {consequence}. Their reaction changes "
                    f"the balance of the scene. {close}"
                )
            else:
                narration = (
                    f"{repeated_note}Your question hangs in the air. After a tense pause, "
                    f"the scene reveals that {consequence}. The new information changes "
                    f"the balance of the scene. {close}"
                )
        elif any(re.search(rf"\b{re.escape(word)}\b", action_lower) for word in (
            "drink", "eat", "take", "grab", "use", "open", "touch", "pick"
        )):
            narration = (
                f"{repeated_note}You {clean_action}. The object refuses to remain passive: "
                f"{consequence}. Its response alters what is physically possible here. "
                f"{close}"
            )
        elif _is_passive_action(action_lower):
            narration = (
                f"{repeated_note}You {clean_action}. Time slips by, but the world does not "
                f"wait with you: {consequence}. {close}"
            )
        else:
            transition = _consequence_transition(
                action_lower, consequence, profile["kind"], turn_offset
            )
            narration = (
                f"{repeated_note}You {clean_action}. {transition}. {close}"
            )

        _genre_visuals = {
            "noir":       "rain-soaked neon street at night; wet pavement reflections; shadows",
            "romance":    "dim coastal midnight light; sealed letter; fog and lamplight",
            "lighthouse": "isolated lighthouse at night; crashing dark waves; rotating beam",
            "scifi":      "deep-space relay station interior; blinking terminals; star field",
            "fantasy":    "enchanted forest road; blue fireflies; ancient glowing stone",
            "bakery":     "moonlit bakery interior; warm bread on shelves; glowing oven",
        }
        _visual_context = _genre_visuals.get(profile["kind"], "atmospheric scene; moody lighting")
        # Carry the previous scene's key visual beat forward so two
        # consecutive images share a continuous thread instead of reading as
        # two unrelated establishing shots.
        _previous_beat = " ".join((context.get("last_narration") or "").split()[:16])
        if _previous_beat:
            image_prompt = (
                f"{consequence_place}; continuing directly from: {_previous_beat}; "
                f"{_visual_context}; now showing: {consequence}"
            )
        else:
            image_prompt = f"{consequence_place}; {_visual_context}; {consequence}"
        choices = _generate_dynamic_choices(full_prompt, narration)
        profile = _story_profile(context, narration)
        location_update = consequence_place
        thread_update = profile["thread"]
        active_characters_update = _active_character_names(context)
        if not active_characters_update and profile["kind"] == "lighthouse":
            active_characters_update = ["Mara Venn (boat captain)"]

    _discovered_flags = {}
    _consequence_for_flag = locals().get("consequence", "")
    if _consequence_for_flag:
        _slug_words = re.findall(r"[a-z0-9]+", _consequence_for_flag.lower())[:6]
        if _slug_words:
            _discovered_flags[f"discovered_{'_'.join(_slug_words)}"] = True

    return {
        "narration": narration,
        "stats_delta": {},
        "add_items": [],
        "remove_items": [],
        "flags": _discovered_flags,
        "relationships": {},
        "add_quests": [],
        "complete_quests": [],
        "discovered_locations": [],
        "narrative_context_updates": {
            "location": location_update,
            "active_characters": active_characters_update,
            "current_goal": thread_update,
            "open_threads": [thread_update],
        },
        "choices_hint": choices[:3],
        "image_prompt": _clean_image_prompt(image_prompt),
        "game_over": False,
        # Carries the consequence sentence back to game_engine so it can be
        # stored in state["last_consequences"] and prevent repetition.
        "_fallback_consequence": locals().get("consequence", ""),
    }


def summarize(history_text: str) -> str:
    """Compress recent story turns into a short rolling summary (1-3 sentences)."""
    try:
        client = _get_client()
    except RuntimeError as exc:
        print(f"[llm_client] Gemini unavailable, using fallback summary: {exc}")
        return _fallback_summary(history_text)

    prompt = (
        "Summarize the following interactive story so far into 2-3 concise "
        "sentences. Preserve key facts: important items, character names, "
        "locations, and unresolved plot threads. Do not add new events.\n\n"
        f"STORY SO FAR:\n{history_text}\n\nSUMMARY:"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        return response.text.strip()
    except Exception as exc:
        reason = "quota/provider error" if _is_quota_error(exc) else "Gemini request failed"
        print(f"[llm_client] {reason}, using fallback summary: {exc}")
        traceback.print_exc()
        return _fallback_summary(history_text)


def _fallback_summary(history_text: str) -> str:
    lines = []
    for raw_line in history_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1].strip()
        lines.append(line)

    if not lines:
        return "The story continues."

    snippet = " ".join(lines[-4:]).strip()
    if len(snippet) > 240:
        snippet = snippet[:237].rstrip() + "..."
    return snippet