"""
Unit Tests for the Adversarial Quantitative Reasoning & Statistical Evaluation Suite.
Verifies:
1. 210+ benchmark problem generation and strict JSON Schema validation
2. 100% pass rate of verified reference solutions across all 6 domains
3. Deterministic detection/rejection of canonical LLM statistical & ML antipattern traps
Compatible with both `pytest` and `python3 -m unittest`.
"""

import unittest
import numpy as np

from src.adversarial_traps import (
    trap_base_rate_neglect_posterior,
    trap_pre_split_target_encoding_leakage,
    trap_roc_auc_and_uncorrected_p_values,
    trap_spurious_levels_regression,
    trap_uncentered_vif_without_intercept,
)
from src.benchmark_catalog import generate_210_problem_catalog, validate_catalog_against_schema
from src.grading_harness import (
    NAIVE_AGENT_TRAP_REGISTRY,
    REFERENCE_SOLVER_REGISTRY,
    evaluate_solver_suite,
)
from src.reference_solutions import (
    solve_bayesian_Diagnostic_posterior,
    solve_leak_free_target_encoding_eval,
    solve_multiple_testing_and_skewed_metric,
    solve_nonstationary_timeseries_regression,
    solve_vif_diagnostics,
    solve_welch_heteroscedastic_test,
)


class TestAdversarialQuantEvalSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problems = generate_210_problem_catalog(seed=2026)

    def test_catalog_has_over_200_problems_and_validates_json_schema(self) -> None:
        """Verify catalog contains 210 problems (35 per domain) and passes JSON Schema validation."""
        self.assertGreaterEqual(len(self.problems), 200)
        self.assertEqual(len(self.problems), 210)
        validated_count = validate_catalog_against_schema(self.problems)
        self.assertEqual(validated_count, 210)

    def test_reference_solutions_achieve_100_percent_pass_rate(self) -> None:
        """Verify ground-truth reference solver suite achieves 210/210 (100%) pass rate."""
        scorecard = evaluate_solver_suite(self.problems, REFERENCE_SOLVER_REGISTRY)
        self.assertEqual(scorecard["total_problems"], 210)
        self.assertEqual(scorecard["total_passed"], 210)
        self.assertEqual(scorecard["overall_pass_rate"], 1.0)

    def test_adversarial_traps_are_caught_and_rejected_by_grading_harness(self) -> None:
        """Verify naive LLM antipattern solvers fail across the adversarial benchmark."""
        trap_scorecard = evaluate_solver_suite(self.problems, NAIVE_AGENT_TRAP_REGISTRY)
        self.assertLess(float(trap_scorecard["overall_pass_rate"]), 0.05)

    def test_vif_collinearity_detection_accuracy(self) -> None:
        """Verify exact VIF calculation detects severe multicollinearity (VIF > 10)."""
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 1, 50)
        x2 = rng.normal(0, 1, 50)
        x3 = 2.0 * x1 - 1.5 * x2 + rng.normal(0, 0.05, 50)
        X = np.column_stack([x1, x2, x3])
        res = solve_vif_diagnostics(X, vif_threshold=10.0)
        self.assertTrue(res["methodological_flag"])
        self.assertGreater(float(res["primary_metric"]), 50.0)

    def test_benjamini_hochberg_blocks_uncorrected_false_positives(self) -> None:
        """Verify BH-FDR correction prevents false-positive p-hacking when min(p) = 0.015 across 25 tests."""
        raw_ps = [0.015] + [0.45] * 24
        y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        y_scores = [0.9, 0.3, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        ref = solve_multiple_testing_and_skewed_metric(raw_ps, y_true, y_scores, fdr_q=0.05)
        trap = trap_roc_auc_and_uncorrected_p_values(raw_ps, y_true, y_scores, fdr_q=0.05)
        self.assertFalse(ref["methodological_flag"])  # 0.015 * 25 = 0.375 > 0.05 (not significant after FDR)
        self.assertTrue(trap["methodological_flag"])  # Naive trap falsely claims significance


if __name__ == "__main__":
    unittest.main()
