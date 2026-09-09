#!/usr/bin/env python3

import unittest

import run_round6_direct_answer as direct


class DirectAnswerTests(unittest.TestCase):
    def test_direct_instruction_is_frozen(self):
        self.assertIn("Do not perform any further grounding", direct.DIRECT_ANSWER_INSTRUCTION)
        self.assertIn("<answer>", direct.DIRECT_ANSWER_INSTRUCTION)

    def test_grounding_only_is_strict_violation(self):
        parsed = direct.parse_direct_output('<grounding>{"bbox_2d":[0,0,1,1],"source":"original_image"}</grounding>')
        self.assertTrue(parsed["grounding_present"])
        self.assertTrue(parsed["protocol_violation"])
        self.assertFalse(parsed["strict_valid_answer"])
        self.assertFalse(parsed["lenient_valid_answer"])

    def test_grounding_and_answer_preserves_lenient_answer(self):
        parsed = direct.parse_direct_output(
            '<grounding>{"bbox_2d":[0,0,1,1],"source":"original_image"}</grounding><answer>blue</answer>'
        )
        self.assertTrue(parsed["protocol_violation"])
        self.assertFalse(parsed["strict_valid_answer"])
        self.assertTrue(parsed["lenient_valid_answer"])
        self.assertEqual(parsed["parsed_answer"], "blue")

    def test_answer_only_is_valid_in_both_modes(self):
        parsed = direct.parse_direct_output("<answer>blue</answer>")
        self.assertFalse(parsed["grounding_present"])
        self.assertTrue(parsed["strict_valid_answer"])
        self.assertTrue(parsed["lenient_valid_answer"])

    def test_retention_arm_mapping(self):
        episode = {"crops": [{"gt_coverage": value} for value in (0.1, 0.9, 0.2, 0.8, 0.0, 0.3)]}
        self.assertEqual(direct.retained_indices("F_full_direct", "x", episode), [1, 2, 3, 4, 5, 6])
        self.assertEqual(direct.retained_indices("O_oracle_top2_direct", "x", episode), [2, 4])
        self.assertEqual(direct.retained_indices("R_recent_top2_direct", "x", episode), [5, 6])
        self.assertEqual(len(direct.retained_indices("X_random_top2_direct", "x", episode)), 2)

    def test_banned_key_scan(self):
        found = direct.recursively_find_banned_keys({"safe": [{"gt_answer": "x"}]}, {"gt_answer"})
        self.assertEqual(found, {"gt_answer"})


if __name__ == "__main__":
    unittest.main()
