#!/usr/bin/env python3
"""Read-only structural tests; these tests never load the model or write outputs."""

from __future__ import annotations

import unittest
from dataclasses import asdict

import run_round6_memory_surgery as surgery


class Round6MemorySurgeryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.bundles, cls.episodes = surgery.build_cohort()
        cls.sample_id = sorted(cls.bundles)[0]

    def test_cohort_partition(self) -> None:
        self.assertEqual(len(self.rows), 300)
        eligible = [row for row in self.rows if row["eligible"]]
        self.assertEqual(len(eligible), len(self.bundles))
        self.assertTrue(any(row["cohort"] == "primary_prefix_hit" for row in eligible))

    def test_sanitized_prefix_has_no_evaluation_keys(self) -> None:
        forbidden = {"gt_answer", "gt_bbox", "category", "gt_coverage", "iou_with_gt"}
        bundle = asdict(self.bundles[self.sample_id])
        self.assertFalse(forbidden.intersection(bundle))
        for row in bundle["turns"] + bundle["crops"]:
            self.assertFalse(forbidden.intersection(row))

    def test_retention_plans(self) -> None:
        episode = self.episodes[self.sample_id]
        for arm in surgery.ARMS:
            retained = surgery.retained_indices(arm, self.sample_id, episode)
            expected = 6 if arm in {"A_full", "D_force_answer_r6"} else 2
            self.assertEqual(len(retained), expected)
            self.assertEqual(retained, sorted(set(retained)))
        self.assertEqual(surgery.retained_indices("C_recent_top2", self.sample_id, episode), [5, 6])

    def test_eviction_changes_images_not_assistant_text(self) -> None:
        prefix = self.bundles[self.sample_id]
        episode = self.episodes[self.sample_id]
        plans = {}
        for arm in ("A_full", "B_oracle_top2", "C_recent_top2", "E_random_top2"):
            retained = surgery.retained_indices(arm, self.sample_id, episode)
            plans[arm] = surgery.InterventionPlan(
                sample_id=self.sample_id,
                condition=arm,
                retained_crop_indices=retained,
                evicted_crop_indices=[index for index in range(1, 7) if index not in retained],
                forced_answer=False,
                prefix_snapshot_sha256=surgery.sha256_json(asdict(prefix)),
            )
        assistant_text = None
        image_counts = {}
        for arm, plan in plans.items():
            _, _, messages, _ = surgery.reconstruct_messages(prefix, plan)
            texts = [message["content"] for message in messages if message["role"] == "assistant"]
            assistant_text = texts if assistant_text is None else assistant_text
            self.assertEqual(texts, assistant_text)
            image_counts[arm] = sum(
                item.get("type") == "image"
                for message in messages if isinstance(message.get("content"), list)
                for item in message["content"] if isinstance(item, dict)
            )
        self.assertEqual(image_counts["A_full"], 7)
        self.assertEqual(image_counts["B_oracle_top2"], 3)
        self.assertEqual(image_counts["C_recent_top2"], 3)
        self.assertEqual(image_counts["E_random_top2"], 3)

    def test_force_answer_only_appends_user_instruction(self) -> None:
        prefix = self.bundles[self.sample_id]
        retained = list(range(1, 7))
        plain = surgery.InterventionPlan(
            self.sample_id, "A_full", retained, [], False,
            surgery.sha256_json(asdict(prefix)),
        )
        forced = surgery.InterventionPlan(
            self.sample_id, "D_force_answer_r6", retained, [], True,
            surgery.sha256_json(asdict(prefix)),
        )
        _, _, plain_messages, _ = surgery.reconstruct_messages(prefix, plain)
        _, _, forced_messages, _ = surgery.reconstruct_messages(prefix, forced)
        self.assertEqual(plain_messages[:-1], forced_messages[:-1])
        self.assertEqual(
            [m["content"] for m in plain_messages if m["role"] == "assistant"],
            [m["content"] for m in forced_messages if m["role"] == "assistant"],
        )
        self.assertIn(surgery.FORCE_ANSWER_INSTRUCTION, forced_messages[-1]["content"][-1]["text"])


if __name__ == "__main__":
    unittest.main()
