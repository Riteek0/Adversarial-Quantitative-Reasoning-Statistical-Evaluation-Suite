"""
Automated Grading Harness that evaluates candidate solver implementations against
deterministically defined runtime criteria, numerical tolerances, methodological flags,
and JSON Schema constraints across all 210 benchmark problems.
"""

import argparse
import json
from typing import Callable, Dict, List
import numpy as np

from src.adversarial_traps import (
    trap_base_rate_neglect_posterior,
    trap_pooled_variance_student_ttest,
    trap_pre_split_target_encoding_leakage,
    trap_roc_auc_and_uncorrected_p_values,
    trap_spurious_levels_regression,
    trap_uncentered_vif_without_intercept,
)
from src.benchmark_catalog import (
    CATALOG_PATH,
    generate_210_problem_catalog,
    save_catalog,
    validate_catalog_against_schema,
)
from src.reference_solutions import (
    solve_bayesian_Diagnostic_posterior,
    solve_leak_free_target_encoding_eval,
    solve_multiple_testing_and_skewed_metric,
    solve_nonstationary_timeseries_regression,
    solve_vif_diagnostics,
    solve_welch_heteroscedastic_test,
)


REFERENCE_SOLVER_REGISTRY: Dict[str, Callable[..., Dict[str, float | bool]]] = {
    "PROB_BAYES": solve_bayesian_Diagnostic_posterior,
    "INFER_STATS": solve_welch_heteroscedastic_test,
    "MULTICOLL_VIF": solve_vif_diagnostics,
    "TIMESERIES_NS": solve_nonstationary_timeseries_regression,
    "LEAKAGE_TRAP": solve_leak_free_target_encoding_eval,
    "MULTI_TEST_PR": solve_multiple_testing_and_skewed_metric,
}

NAIVE_AGENT_TRAP_REGISTRY: Dict[str, Callable[..., Dict[str, float | bool]]] = {
    "PROB_BAYES": trap_base_rate_neglect_posterior,
    "INFER_STATS": trap_pooled_variance_student_ttest,
    "MULTICOLL_VIF": trap_uncentered_vif_without_intercept,
    "TIMESERIES_NS": trap_spurious_levels_regression,
    "LEAKAGE_TRAP": trap_pre_split_target_encoding_leakage,
    "MULTI_TEST_PR": trap_roc_auc_and_uncorrected_p_values,
}


def grade_single_problem(
    problem: Dict[str, object],
    solver_fn: Callable[..., Dict[str, float | bool]],
) -> Dict[str, object]:
    """Execute candidate solver against a single benchmark problem and verify numerical & methodological match."""
    params = problem["input_parameters"]  # type: ignore
    gt = problem["ground_truth"]  # type: ignore
    tol = float(problem["tolerance"])  # type: ignore

    try:
        candidate_out = solver_fn(**params)
        cand_metric = float(candidate_out["primary_metric"])
        cand_flag = bool(candidate_out["methodological_flag"])
        gt_metric = float(gt["primary_metric"])
        gt_flag = bool(gt["methodological_flag"])

        metric_match = abs(cand_metric - gt_metric) <= tol
        flag_match = cand_flag == gt_flag
        passed = bool(metric_match and flag_match)
        return {
            "problem_id": problem["problem_id"],
            "domain": problem["domain"],
            "passed": passed,
            "metric_error": round(abs(cand_metric - gt_metric), 6),
            "flag_match": flag_match,
        }
    except Exception as exc:
        return {
            "problem_id": problem["problem_id"],
            "domain": problem["domain"],
            "passed": False,
            "error": str(exc),
        }


def evaluate_solver_suite(
    problems: List[Dict[str, object]],
    solver_registry: Dict[str, Callable[..., Dict[str, float | bool]]],
) -> Dict[str, object]:
    """Grade a solver registry across all 210 problems and aggregate per-domain scorecards."""
    validate_catalog_against_schema(problems)

    domain_totals: Dict[str, int] = {}
    domain_passed: Dict[str, int] = {}
    total_passed = 0

    for prob in problems:
        dom = str(prob["domain"])
        solver_fn = solver_registry[dom]
        res = grade_single_problem(prob, solver_fn)
        domain_totals[dom] = domain_totals.get(dom, 0) + 1
        if res["passed"]:
            domain_passed[dom] = domain_passed.get(dom, 0) + 1
            total_passed += 1

    per_domain_summary = {
        dom: {
            "total": domain_totals[dom],
            "passed": domain_passed.get(dom, 0),
            "pass_rate": round(domain_passed.get(dom, 0) / max(domain_totals[dom], 1), 4),
        }
        for dom in sorted(domain_totals.keys())
    }

    return {
        "total_problems": len(problems),
        "total_passed": total_passed,
        "overall_pass_rate": round(total_passed / max(len(problems), 1), 4),
        "per_domain_breakdown": per_domain_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial Quantitative Evaluation Suite Grading Harness")
    parser.add_argument("--generate-catalog", action="store_true", help="Generate and validate 210-problem JSON catalog")
    parser.add_argument("--run-evaluation", action="store_true", help="Run grading harness on reference vs naive agent solvers")
    args = parser.parse_args()

    if args.generate_catalog or not CATALOG_PATH.exists():
        save_catalog()

    problems = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    if args.run_evaluation:
        ref_scorecard = evaluate_solver_suite(problems, REFERENCE_SOLVER_REGISTRY)
        trap_scorecard = evaluate_solver_suite(problems, NAIVE_AGENT_TRAP_REGISTRY)
        report = {
            "schema_validated_problems": len(problems),
            "reference_solutions_scorecard": ref_scorecard,
            "naive_llm_antipattern_scorecard": trap_scorecard,
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
