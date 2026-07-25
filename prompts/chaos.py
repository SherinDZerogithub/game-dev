"""
chaos.py

Controllable randomness: every so often, injects a "twist event" fragment
into the prompt, nudging the LLM to bend the story in an unforeseen but
tonally consistent direction. The chance creeps up slowly the longer the
game runs, mirroring how stories naturally gain momentum.
"""

import os
import re

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

with open(os.path.join(PROMPTS_DIR, "chaos_prompt.txt"), "r", encoding="utf-8") as f:
    _RAW_CHAOS_TEXT = f.read().strip()

# chaos_prompt.txt holds multiple *alternative* twist directives (each its
# own "TWIST EVENT:" block). Previously the whole file was read as one
# CHAOS_FRAGMENT string and both directives were concatenated into the
# prompt together. That doubled-up, unlabelled instruction text was long
# and repetitive enough that the model sometimes echoed it back verbatim as
# "narration" instead of writing original prose (i.e. the raw instruction
# leaked into the player-facing output). Splitting into distinct blocks and
# picking ONE at random per trigger keeps each injected instruction short
# and reduces that echo risk, while also giving twists more variety.
CHAOS_FRAGMENTS = [
    block.strip()
    for block in re.split(r"(?=^TWIST EVENT:)", _RAW_CHAOS_TEXT, flags=re.MULTILINE)
    if block.strip()
]

# Backward-compatible alias (some callers/tests may still reference the old
# single-string name); kept as the first block so it's never empty.
CHAOS_FRAGMENT = CHAOS_FRAGMENTS[0] if CHAOS_FRAGMENTS else _RAW_CHAOS_TEXT

# Base chance per turn that a twist fires in standard mode.
NORMAL_CHAOS_CHANCE = 0.12

# Twist Mode dials this way up and shortens how often memory gets summarized,
# so the story unravels faster on purpose.
CHAOS_MODE_CHANCE = 0.45


def maybe_trigger_chaos(chaos_mode: bool, turn_count: int) -> str | None:
    """Return one twist prompt fragment to inject this turn, or None."""
    chance = CHAOS_MODE_CHANCE if chaos_mode else NORMAL_CHAOS_CHANCE
    drift_bonus = min(0.2, turn_count * 0.005)  # small creeping increase
    if (chance + drift_bonus) > 0 and _roll(chance + drift_bonus):
        return _pick_fragment()
    return None


def _pick_fragment() -> str:
    import random
    return random.choice(CHAOS_FRAGMENTS) if CHAOS_FRAGMENTS else CHAOS_FRAGMENT


def _roll(threshold: float) -> bool:
    import random
    return random.random() < threshold