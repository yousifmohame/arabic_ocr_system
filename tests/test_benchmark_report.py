import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_preprocessing.py"
SPEC = importlib.util.spec_from_file_location("compare_preprocessing", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkOracleTests(unittest.TestCase):
    def test_oracle_ranks_by_ground_truth_but_does_not_change_runtime_selection(self):
        page = {
            "selected_variant": "original",
            "candidates": [
                {"name": "original", "metrics": {"cer": 0.44, "normalized_cer": 0.40, "wer": 0.56}},
                {"name": "perspective", "metrics": {"cer": 0.25, "normalized_cer": 0.22, "wer": 0.41}},
            ],
        }
        result = MODULE._benchmark_oracle(page)
        self.assertEqual(result["runtime_selected_candidate"], "original")
        self.assertFalse(result["ground_truth_used_for_runtime_selection"])
        self.assertEqual(result["best_candidates_by_metric"]["cer"]["candidate"], "perspective")
        self.assertFalse(result["best_candidates_by_metric"]["cer"]["runtime_selection_matches_benchmark_oracle"])

    def test_no_ground_truth_metrics_returns_none(self):
        self.assertIsNone(MODULE._benchmark_oracle({"selected_variant": "original", "candidates": []}))


if __name__ == "__main__":
    unittest.main()
