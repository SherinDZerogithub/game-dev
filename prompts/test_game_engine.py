import unittest
from unittest.mock import patch

from prompts import game_engine
from prompts import llm_client


class NarrativeContextTests(unittest.TestCase):
    def test_prompt_contains_story_memory_and_bridge_instruction(self):
        state = {
            **game_engine.deepcopy(game_engine.DEFAULT_STATE),
            "narrative_context": {
                "location": "Moonrise Bridge",
                "active_characters": ["Pip"],
                "current_goal": "return the silver bell",
                "open_threads": ["Pip heard singing below the bridge"],
            },
        }
        prompt = game_engine._assemble_prompt(
            state,
            [{"role": "narrator", "text": "Pip points below the bridge."}],
            "ask him about it",
            None,
        )

        self.assertIn("Moonrise Bridge", prompt)
        self.assertIn("Pip heard singing below the bridge", prompt)
        self.assertIn("show a concrete consequence", prompt)
        self.assertIn("Continuity instruction", prompt)

    @patch("prompts.game_engine.image_gen.generate_scene_image", return_value=None)
    def test_turn_result_persists_narrative_context(self, _image):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        result = {
            "narration": "Pip explains that the river is singing to the bell.",
            "state_updates": {},
            "narrative_context_updates": {
                "location": "under Moonrise Bridge",
                "active_characters": ["Pip", "river spirit"],
                "current_goal": "return the silver bell",
                "open_threads": ["why the river knows the heroine's name"],
            },
            "choices_hint": [],
            "game_over": False,
        }

        game_engine._apply_turn_result(state, result)

        self.assertEqual(state["narrative_context"]["location"], "under Moonrise Bridge")
        self.assertIn("river spirit", state["narrative_context"]["active_characters"])

    @patch("prompts.game_engine.image_gen.generate_scene_image", return_value=None)
    def test_turn_result_never_returns_blank_narration(self, _image):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)

        scene = game_engine._apply_turn_result(state, {"narration": "   "})

        self.assertIn("scene changes", scene["narration"])
        self.assertTrue(scene["narration"].strip())

    @patch("prompts.game_engine.image_gen.generate_scene_image", return_value=None)
    def test_choice_fallback_deduplicates_before_limiting(self, _image):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["action_history"] = ["inspect the locked gate"]

        scene = game_engine._apply_turn_result(
            state,
            {
                "narration": "The gate shudders, revealing a narrow gap.",
                "choices_hint": [
                    "Inspect the gate",
                    "inspect the gate",
                    "Try the narrow gap",
                    "Listen at the hinges",
                ],
            },
        )

        self.assertEqual(len(scene["choices_hint"]), 3)
        self.assertEqual(
            scene["choices_hint"],
            [
                "Inspect the gate",
                "Try the narrow gap",
                "Listen at the hinges",
            ],
        )

    @patch("prompts.game_engine.image_gen.generate_scene_image", return_value=None)
    def test_choice_fallback_does_not_sniff_letter_keyword(self, _image):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[1]["opening"]
        state["action_history"] = ["watch the rain"]

        scene = game_engine._apply_turn_result(
            state,
            {
                "narration": "Rain gathers in the doorway while the street empties.",
                "choices_hint": ["Watch the street"],
            },
        )

        self.assertNotIn("read the letter carefully", scene["choices_hint"])
        self.assertTrue(any(choice.lower() == "watch the street" for choice in scene["choices_hint"]))
        self.assertTrue(any("watch the rain" in choice.lower() for choice in scene["choices_hint"]))

    def test_repetition_checker_catches_soft_rephrasing_and_short_blocks(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["narration_history"] = [
            "The discovery changes what the next step can be near the old bridge."
        ]

        self.assertTrue(
            game_engine._result_repeats_recent_story(
                {
                    "narration": "The discovery changes what your next step can be beside the old bridge."
                },
                state,
            )
        )

        state["narration_history"] = [
            "A cold wind rattles the shutters as the lantern burns low."
        ]
        self.assertTrue(
            game_engine._result_repeats_recent_story(
                {
                    "narration": "The lantern burns low while the cold wind rattles nearby shutters."
                },
                state,
            )
        )

    def test_fallback_understands_recent_turn_format_and_hidden_path(self):
        prompt = game_engine._assemble_prompt(
            game_engine.deepcopy(game_engine.DEFAULT_STATE),
            [
                {"role": "narrator", "text": "A trail of silver pawprints slips away from the old road."},
                {"role": "player", "text": "follow the road"},
            ],
            "i search for a hidden path",
            None,
        )

        context = llm_client._extract_context_from_prompt(prompt)
        result = llm_client._fallback_turn(prompt)

        self.assertIn("silver pawprints", context["last_narration"])
        self.assertIn("hidden path", result["narration"].lower())
        self.assertNotIn("responds to your", result["narration"].lower())

    def test_fallback_reads_player_input_from_engine_prompt(self):
        prompt = game_engine._assemble_prompt(
            game_engine.deepcopy(game_engine.DEFAULT_STATE),
            [],
            "follow the old road",
            None,
        )

        result = llm_client._fallback_turn(prompt)

        self.assertIn("road", result["narration"].lower())
        self.assertNotIn("You decide to .", result["narration"])

    def test_fallback_uses_story_prose_for_open_actions(self):
        prompt = game_engine._assemble_prompt(
            game_engine.deepcopy(game_engine.DEFAULT_STATE),
            [{"role": "narrator", "text": "The sealed letter warms in your coat pocket."}],
            "investigate the consequence",
            None,
        )

        result = llm_client._fallback_turn(prompt)
        lower = result["narration"].lower()

        self.assertNotIn("free-form action", lower)
        self.assertNotIn("predefined menu option", lower)
        self.assertIn("pursue the new clue", result["choices_hint"])

    def test_romance_search_stays_at_bookshop_doorway(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[1]["opening"]
        state["narrative_context"]["location"] = "the bookshop doorway"
        prompt = game_engine._assemble_prompt(
            state, [], "check the opposite direction", None
        )

        result = llm_client._fallback_turn(prompt)
        lower = result["narration"].lower()

        self.assertNotIn("station clock", lower)
        self.assertNotIn("last platform", lower)
        self.assertNotIn("teleport", lower)
        self.assertEqual(
            result["narrative_context_updates"]["location"],
            "the bookshop doorway",
        )
        self.assertIn("bookshop doorway", result["image_prompt"].lower())
        self.assertNotIn("station", result["image_prompt"].lower())

    def test_romance_margin_action_images_the_letter(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[1]["opening"]
        state["narrative_context"]["location"] = "the bookshop doorway"
        prompt = game_engine._assemble_prompt(
            state, [], "study the strange symbols in the margins", None
        )

        result = llm_client._fallback_turn(prompt)

        self.assertIn("confession", result["narration"].lower())
        # Image prompt must reference the actual location and the letter,
        # but NOT Vienna-station imagery since the player is at the bookshop.
        self.assertIn("sealed letter", result["image_prompt"].lower())
        self.assertIn("bookshop doorway", result["image_prompt"].lower())
        self.assertNotIn("station", result["image_prompt"].lower())

    def test_romance_reaches_station_only_when_action_moves_there(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[1]["opening"]
        state["narrative_context"]["location"] = "the bookshop doorway"
        prompt = game_engine._assemble_prompt(
            state, [], "walk to the station", None
        )

        result = llm_client._fallback_turn(prompt)

        self.assertNotIn("bookshop doorway grows quiet", result["narration"].lower())
        self.assertEqual(
            result["narrative_context_updates"]["location"],
            "the last platform",
        )

    def test_passive_rest_does_not_repeat_discovery_bridge(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[4]["opening"]
        state["narrative_context"]["location"] = "a forest tent"
        state["narration_history"] = [
            "The discovery changes what the next step can be: it points toward the old forest road."
        ]
        prompt = game_engine._assemble_prompt(state, [], "sleep in the tent", None)

        result = llm_client._fallback_turn(prompt)

        self.assertNotIn(
            "the discovery changes what the next step can be",
            result["narration"].lower(),
        )
        self.assertIn("time slips by", result["narration"].lower())

    def test_kitchen_search_does_not_suggest_the_same_recipe_search(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[5]["opening"]
        state["narrative_context"]["location"] = "the bakery kitchen"
        prompt = game_engine._assemble_prompt(
            state, [], "search the kitchen for hidden recipes", None
        )

        result = llm_client._fallback_turn(prompt)
        choices = [choice.lower() for choice in result["choices_hint"]]

        self.assertFalse(any("search the kitchen" in choice or "recipe" in choice for choice in choices))
        self.assertTrue(any("leave" in choice or "follow" in choice for choice in choices))

    def test_lighthouse_dialogue_names_the_established_boat_captain(self):
        state = game_engine.deepcopy(game_engine.DEFAULT_STATE)
        state["world_prompt"] = game_engine.LORE_PRESETS[2]["opening"]
        prompt = game_engine._assemble_prompt(
            state, [], "ask what the locked journal is hiding", None
        )

        result = llm_client._fallback_turn(prompt)
        lower = result["narration"].lower()

        self.assertIn("mara venn", lower)
        self.assertIn("boat captain", lower)
        self.assertNotIn("the listener", lower)
        self.assertIn(
            "mara venn",
            " ".join(result["narrative_context_updates"]["active_characters"]).lower(),
        )

    def test_presets_are_interview_safe(self):
        presets = game_engine.list_presets()
        self.assertGreaterEqual(len(presets), 5)
        for p in presets:
            text = (p["label"] + " " + p["opening"]).lower()
            # none of the old candy/witch themes
            for banned in ("candy", "sugar", "witch", "gummy", "pastry", "lollipop"):
                self.assertNotIn(banned, text, f"preset {p['id']} contains '{banned}'")


if __name__ == "__main__":
    unittest.main()
