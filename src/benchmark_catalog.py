"""
Generates and validates the 210-problem Adversarial Quantitative Reasoning & Statistical
Evaluation Benchmark (35 parameterized problems across each of the 6 domains = 210 total).
Validates every problem record against `schemas/problem_schema.json`.
"""

import json
from pathlib import Path
from typing import Dict, List
import jsonschema
import numpy as np

from src.adversarial_traps import (
    trap_base_rate_neglect_posterior,
    trap_pooled_variance_student_ttest,
    trap_pre_split_target_encoding_leakage,
    trap_roc_auc_and_uncorrected_p_values,
    trap_spurious_levels_regression,
    trap_uncentered_vif_without_intercept,
)
from src.reference_solutions import (
    solve_bayesian_Diagnostic_posterior,
    solve_leak_free_target_encoding_eval,
    solve_multiple_testing_and_skewed_metric,
    solve_nonstationary_timeseries_regression,
    solve_vif_diagnostics,
    solve_welch_heteroscedastic_test,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "problem_schema.json"
CATALOG_PATH = REPO_ROOT / "benchmarks" / "problems_210.json"


def generate_210_problem_catalog(seed: int = 2026) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    problems: List[Dict[str, object]] = []

    # 1. Domain PROB_BAYES (35 problems)
    for idx in range(1, 36):
        prior = round(float(rng.uniform(0.001, 0.035)), 5)
        sens = round(float(rng.uniform(0.88, 0.99)), 4)
        fpr = round(float(rng.uniform(0.01, 0.08)), 4)
        k_tests = 1 if idx <= 20 else 2
        params = {
            "prior_prevalence": prior,
            "sensitivity": sens,
            "false_positive_rate": fpr,
            "n_consecutive_positive": k_tests,
        }
        gt = solve_bayesian_Diagnostic_posterior(**params)
        trap = trap_base_rate_neglect_posterior(**params)
        problems.append(
            {
                "problem_id": f"PROB_BAYES_{idx:03d}",
                "domain": "PROB_BAYES",
                "difficulty": "adversarial" if k_tests == 2 else "hard",
                "title": f"Skewed Prior Bayesian Updating under {prior*100:.2f}% Prevalence (Case {idx})",
                "prompt": (
                    f"Given base-rate prevalence {prior}, sensitivity {sens}, and false-positive rate {fpr} "
                    f"across {k_tests} independent positive screening test(s), compute exact posterior P(D=1|T)."
                ),
                "targeted_antipattern": "Base-rate fallacy (ignoring skewed prior prevalence when computing posterior).",
                "input_parameters": params,
                "ground_truth": gt,
                "tolerance": 1e-4,
                "adversarial_trap_output": trap,
            }
        )

    # 2. Domain INFER_STATS (35 problems)
    for idx in range(1, 36):
        # Unequal sample sizes and heteroscedastic variances where pooled Student t-test diverges from Welch's t-test
        n_a, n_b = 15, 65
        mean_shift = float(rng.uniform(0.8, 1.8))
        sample_a = [round(float(x), 5) for x in rng.normal(loc=mean_shift, scale=3.2, size=n_a)]
        sample_b = [round(float(x), 5) for x in rng.normal(loc=0.0, scale=0.65, size=n_b)]
        params = {"sample_a": sample_a, "sample_b": sample_b, "alpha": 0.05}
        gt = solve_welch_heteroscedastic_test(**params)
        trap = trap_pooled_variance_student_ttest(**params)
        problems.append(
            {
                "problem_id": f"INFER_STATS_{idx:03d}",
                "domain": "INFER_STATS",
                "difficulty": "hard",
                "title": f"Heteroscedastic Two-Sample Inference with Imbalanced Sample Sizes (Case {idx})",
                "prompt": (
                    "Evaluate two-sided hypothesis test H0: mu_A = mu_B under severe variance inequality "
                    "(n_A=15, sigma_A~3.2 vs n_B=65, sigma_B~0.65) using Welch-Satterthwaite degrees of freedom."
                ),
                "targeted_antipattern": "Applying homoscedastic pooled-variance Student t-test under unequal variances and imbalanced n.",
                "input_parameters": params,
                "ground_truth": gt,
                "tolerance": 1e-4,
                "adversarial_trap_output": trap,
            }
        )

    # 3. Domain MULTICOLL_VIF (35 problems)
    for idx in range(1, 36):
        n_rows = 45
        x1 = rng.normal(10.0, 2.0, size=n_rows)
        x2 = rng.normal(-5.0, 1.5, size=n_rows)
        # Construct near-collinear linear combination x3 = 1.5 * x1 - 0.8 * x2 + small noise
        noise_scale = float(rng.uniform(0.08, 0.35))
        x3 = 1.5 * x1 - 0.8 * x2 + rng.normal(0.0, noise_scale, size=n_rows)
        mat = np.column_stack([x1, x2, x3]).round(5).tolist()
        params = {"design_matrix": mat, "vif_threshold": 10.0}
        gt = solve_vif_diagnostics(**params)
        trap = trap_uncentered_vif_without_intercept(**params)
        problems.append(
            {
                "problem_id": f"MULTICOLL_VIF_{idx:03d}",
                "domain": "MULTICOLL_VIF",
                "difficulty": "adversarial",
                "title": f"Multicollinearity Diagnostics & Maximum VIF Calculation (Case {idx})",
                "prompt": (
                    "Compute the exact maximum Variance Inflation Factor (VIF) across all predictors "
                    "in the 3-variable design matrix with an explicit intercept term, and flag if max(VIF) >= 10."
                ),
                "targeted_antipattern": "Computing pairwise correlation instead of multi-predictor R_j^2 with intercept.",
                "input_parameters": params,
                "ground_truth": gt,
                "tolerance": 1e-3,
                "adversarial_trap_output": trap,
            }
        )

    # 4. Domain TIMESERIES_NS (35 problems)
    for idx in range(1, 36):
        T = 60
        # Generate two independent I(1) random walks with drift that trigger spurious correlation in levels
        eps_x = rng.normal(0.15, 1.0, size=T)
        eps_y = rng.normal(0.18, 1.0, size=T)
        rw_x = [round(float(v), 5) for v in np.cumsum(eps_x)]
        rw_y = [round(float(v), 5) for v in np.cumsum(eps_y)]
        params = {"series_x": rw_x, "series_y": rw_y, "alpha": 0.05}
        gt = solve_nonstationary_timeseries_regression(**params)
        trap = trap_spurious_levels_regression(**params)
        problems.append(
            {
                "problem_id": f"TIMESERIES_NS_{idx:03d}",
                "domain": "TIMESERIES_NS",
                "difficulty": "adversarial",
                "title": f"Non-Stationary I(1) Time-Series Spurious Regression Defense (Case {idx})",
                "prompt": (
                    "Given two integrated I(1) time-series processes, apply first-differencing prior to "
                    "OLS regression and return the two-sided slope p-value and significance flag at alpha=0.05."
                ),
                "targeted_antipattern": "Granger-Newbold spurious regression on raw non-stationary I(1) levels.",
                "input_parameters": params,
                "ground_truth": gt,
                "tolerance": 1e-4,
                "adversarial_trap_output": trap,
            }
        )

    # 5. Domain LEAKAGE_TRAP (35 problems)
    for idx in range(1, 36):
        n_total = 60
        split_idx = 40
        # High-cardinality categories where test categories memorise targets if encoded before split
        cats = [f"C_{i % 18:02d}" for i in range(n_total)]
        rng.shuffle(cats)
        feats = [round(float(v), 5) for v in rng.normal(0.0, 1.5, size=n_total)]
        targets = [int(v) for v in rng.integers(0, 2, size=n_total)]
        params = {
            "categories": cats,
            "features": feats,
            "targets": targets,
            "split_index": split_idx,
            "smoothing_m": 5.0,
        }
        gt = solve_leak_free_target_encoding_eval(**params)
        trap = trap_pre_split_target_encoding_leakage(**params)
        problems.append(
            {
                "problem_id": f"LEAKAGE_TRAP_{idx:03d}",
                "domain": "LEAKAGE_TRAP",
                "difficulty": "adversarial",
                "title": f"Zero-Leakage Partitioning vs Pre-Split Target Encoding (Case {idx})",
                "prompt": (
                    f"Partition dataset at index {split_idx}, fit smoothed target encoding (m=5.0) and feature "
                    "standardization strictly on [:split_index], and compute out-of-sample test RMSE."
                ),
                "targeted_antipattern": "Fitting target encoding and feature scaling on the full dataset prior to train/test split.",
                "input_parameters": params,
                "ground_truth": gt,
                "tolerance": 1e-4,
                "adversarial_trap_output": trap,
            }
        )

    # 6. Domain MULTI_TEST_PR (35 problems)
    for idx in range(1, 36):
        # 25 hypothesis tests under the global null where min(p) < 0.05 occurs by chance, but BH-FDR rejects 0
        raw_ps = sorted([round(float(v), 5) for v in rng.uniform(0.012, 0.95, size=25)])
        raw_ps[0] = round(float(rng.uniform(0.008, 0.038)), 5)  # Uncorrected false positive trap!
        # Skewed binary classification (4 positives out of 80 samples = 5% prevalence)
        y_true = [1, 1, 1, 1] + [0] * 76
        # High false positives at top ranks: ROC-AUC looks high (~0.85), but PR-AUC is low (~0.25)
        pos_scores = [round(float(v), 5) for v in rng.uniform(0.55, 0.78, size=4)]
        neg_scores = [round(float(v), 5) for v in np.concatenate([rng.uniform(0.60, 0.92, size=8), rng.uniform(0.01, 0.50, size=68)])]
        y_scores = pos_scores + neg_scores
        params = {
            "raw_p_values": raw_ps,
            "y_true": y_true,
            "y_scores": y_scores,
            "fdr_q": 0.05,
        }
        gt = solve_multiple_testing_and_skewed_metric(**params)
        trap = trap_roc_auc_and_uncorrected_p_values(**params)
        problems.append(
            {
                "problem_id": f"MULTI_TEST_PR_{idx:03d}",
                "domain": "MULTI_TEST_PR",
                "difficulty": "adversarial",
                "title": f"PR-AUC under Skewed Priors & Benjamini-Hochberg FDR Control (Case {idx})",
                "prompt": (
                    "Compute exact Area Under the Precision-Recall Curve (PR-AUC) on the skewed binary predictions "
                    "and determine whether any hypothesis survives Benjamini-Hochberg FDR control at q=0.05."
                ),
                "targeted_antipattern": "Reporting ROC-AUC instead of PR-AUC under severe imbalance and claiming significance without multiple testing correction.",
                "input_parameters": params,
                "ground_truth": gt,
                "tolerance": 1e-4,
                "adversarial_trap_output": trap,
            }
        )

    return problems


def validate_catalog_against_schema(problems: List[Dict[str, object]]) -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for prob in problems:
        validator.validate(prob)
    return len(problems)


def save_catalog() -> Path:
    problems = generate_210_problem_catalog(seed=2026)
    count = validate_catalog_against_schema(problems)
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(problems, indent=2), encoding="utf-8")
    print(f"Generated and schema-validated {count} adversarial problems at {CATALOG_PATH}")
    return CATALOG_PATH


if __name__ == "__main__":
    save_catalog()
