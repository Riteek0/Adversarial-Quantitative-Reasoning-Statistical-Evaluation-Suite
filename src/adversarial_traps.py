"""
Canonical Flawed Agent Implementations (Adversarial Traps).
Demonstrates the exact statistical and methodological mistakes made by naive LLMs/agents
on each of the 6 benchmark categories.
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def trap_base_rate_neglect_posterior(
    prior_prevalence: float,
    sensitivity: float,
    false_positive_rate: float,
    n_consecutive_positive: int = 1,
) -> Dict[str, float | bool]:
    """ANTIPATTERN: Confuses P(D|T) with sensitivity or ignores skewed base-rate prior (sets prior=0.5)."""
    naive_posterior = float(sensitivity) / (float(sensitivity) + float(false_positive_rate))
    return {
        "primary_metric": round(naive_posterior, 6),
        "methodological_flag": bool(naive_posterior > 0.50),
    }


def trap_pooled_variance_student_ttest(
    sample_a: List[float] | np.ndarray,
    sample_b: List[float] | np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float | bool]:
    """ANTIPATTERN: Uses pooled-variance Student's t-test (equal_var=True) under severe heteroscedasticity."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    res = stats.ttest_ind(a, b, equal_var=True)
    p_val = float(res.pvalue)
    return {
        "primary_metric": round(p_val, 6),
        "methodological_flag": bool(p_val < alpha),
    }


def trap_uncentered_vif_without_intercept(
    design_matrix: List[List[float]] | np.ndarray,
    vif_threshold: float = 10.0,
) -> Dict[str, float | bool]:
    """ANTIPATTERN: Computes correlation or OLS without an intercept term, misestimating VIF."""
    X = np.asarray(design_matrix, dtype=float)
    corr = np.corrcoef(X, rowvar=False)
    max_off_diag = float(np.max(np.abs(corr - np.eye(X.shape[1]))))
    naive_metric = 1.0 / max(1.0 - max_off_diag, 1e-6)
    return {
        "primary_metric": round(naive_metric, 6),
        "methodological_flag": False,
    }


def trap_spurious_levels_regression(
    series_x: List[float] | np.ndarray,
    series_y: List[float] | np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float | bool]:
    """ANTIPATTERN: Regresses raw non-stationary I(1) levels directly without first-differencing."""
    x = np.asarray(series_x, dtype=float)
    y = np.asarray(series_y, dtype=float)
    _, _, _, p_val, _ = stats.linregress(x, y)
    return {
        "primary_metric": round(float(p_val), 6),
        "methodological_flag": bool(p_val < alpha),
    }


def trap_pre_split_target_encoding_leakage(
    categories: List[str],
    features: List[float],
    targets: List[int],
    split_index: int,
    smoothing_m: float = 5.0,
) -> Dict[str, float | bool]:
    """ANTIPATTERN: Computes target encoding and standardization across the entire dataset BEFORE splitting."""
    cats = np.asarray(categories, dtype=str)
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)

    # Leaky global target mean using test labels
    df_all = pd.DataFrame({"c": cats, "y": y})
    global_means = df_all.groupby("c")["y"].transform("mean").to_numpy(dtype=float)
    x_global_scaled = (x - np.mean(x)) / (np.std(x) + 1e-9)

    X_all = np.column_stack([np.ones(len(x)), x_global_scaled, global_means])
    w, _, _, _ = np.linalg.lstsq(X_all, y, rcond=None)
    preds_te = np.clip(X_all[split_index:] @ w, 0.0, 1.0)
    leaky_rmse = float(np.sqrt(np.mean((y[split_index:] - preds_te) ** 2)))

    return {
        "primary_metric": round(leaky_rmse, 6),
        "methodological_flag": False,
    }


def trap_roc_auc_and_uncorrected_p_values(
    raw_p_values: List[float],
    y_true: List[int],
    y_scores: List[float],
    fdr_q: float = 0.05,
) -> Dict[str, float | bool]:
    """
    ANTIPATTERN: Reports inflated ROC-AUC instead of PR-AUC under severe class imbalance,
    and claims significance whenever min(raw_p_values) < 0.05 without multiple testing correction.
    """
    p_arr = np.asarray(raw_p_values, dtype=float)
    uncorrected_sig = bool(np.any(p_arr < fdr_q))

    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    pos_scores = scores[y_t == 1]
    neg_scores = scores[y_t == 0]
    # Mann-Whitney U statistic for ROC-AUC
    u_stat, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    roc_auc = float(u_stat / (len(pos_scores) * len(neg_scores)))

    return {
        "primary_metric": round(roc_auc, 6),
        "methodological_flag": uncorrected_sig,
    }
