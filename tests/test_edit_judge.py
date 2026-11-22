"""The judge's verdict (scripts/pipeline/edit_judge.py: verdict_of), without GPU, AI or pictures."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "pipeline"))
from edit_judge import verdict_of  # noqa: E402


class Verdict(unittest.TestCase):
    def test_numbers_decide(self):
        self.assertEqual(verdict_of({"kept": 98, "part": 82}, 85, 70)[0], "pass")
        self.assertEqual(verdict_of({"kept": 80, "part": 82}, 85, 70)[0], "redo")
        self.assertEqual(verdict_of({"kept": 98, "part": 60}, 85, 70)[0], "redo")

    def test_ai_not_seeing_the_part_is_only_a_warning_when_the_numbers_pass_clearly(self):
        # cora's tiara, Nov 19: in every render, the AI said "not there" (1/10)
        verdict, why, warnings = verdict_of({"kept": 98, "part": 82, "ai": {"part_is_there": False, "score": 1, "why": "no tiara"}}, 85, 70)
        self.assertEqual(verdict, "pass")
        self.assertEqual(why, [])
        self.assertTrue(any("not a redo" in w for w in warnings))
        self.assertTrue(any("AI score 1" in w for w in warnings))

    def test_ai_not_seeing_the_part_is_a_redo_when_the_numbers_are_borderline(self):
        for best in ({"kept": 88, "part": 90}, {"kept": 98, "part": 72}):
            best["ai"] = {"part_is_there": False, "score": 2}
            verdict, why, _ = verdict_of(best, 85, 70)
            self.assertEqual(verdict, "redo", best)
            self.assertTrue(any("does not see" in w for w in why))


if __name__ == "__main__":
    unittest.main()
