"""
chaos.py

Controllable randomness: every so often, injects a "twist event" fragment
into the prompt, nudging the LLM to bend the story in an unforeseen but
tonally consistent direction. The chance creeps up slowly the longer the
game runs, mirroring how stories naturally gain momentum.
"""

import os

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

with open(os.path.join(PROMPTS_DIR, "chaos_prompt.txt"), "r", encoding="utf-8") as f:
    CHAOS_FRAGMENT = f.read().strip()

# Base chance per turn that a twist fires in standard mode.
NORMAL_CHAOS_CHANCE = 0.12

# Twist Mode dials this way up and shortens how often memory gets summarized,
# so the story unravels faster on purpose.
CHAOS_MODE_CHANCE = 0.45


def maybe_trigger_chaos(chaos_mode: bool, turn_count: int) -> str | None:
    """Return the twist prompt fragment to inject this turn, or None."""
    chance = CHAOS_MODE_CHANCE if chaos_mode else NORMAL_CHAOS_CHANCE
    drift_bonus = min(0.2, turn_count * 0.005)  # small creeping increase
    if (chance + drift_bonus) > 0 and _roll(chance + drift_bonus):
        return CHAOS_FRAGMENT
    return None


def _roll(threshold: float) -> bool:
    import random
    return random.random() < threshold
