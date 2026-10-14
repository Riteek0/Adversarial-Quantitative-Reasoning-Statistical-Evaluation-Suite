"""
Verified Reference Solutions and Ground-Truth Mathematical Implementations:
1. Probability Theory & Bayesian Updating under Skewed Priors
2. Inferential Statistics: Welch-Satterthwaite Two-Sample Test under Heteroscedasticity
3. Multicollinearity Diagnostics: Exact Variance Inflation Factor (VIF) & Ridge Stabilization
4. Non-Stationary Time-Series Modeling: First-Difference Stationarity & Walk-Forward Validation
5. Zero-Leakage Partitioning vs Pre-Split Target/Imputation Leakage
6. Multiple Testing Corrections (Benjamini-Hochberg FDR & Holm-Bonferroni) and PR-AUC vs ROC-AUC
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats


# 1. Probability Theory & Skewed Conditional Priors
def solve_bayesian_Diagnostic_posterior(
    prior_prevalence: float,
    sensitivity: float,
    false_positive_rate: float,
    n_consecutive_positive: int = 1,
) -> Dict[str, float | bool]:
    """
    Compute exact posterior probability P(D=1 | T_1=1, ..., T_k=1) under conditional independence
    given a skewed base rate `prior_prevalence`.
    """
    p_d = float(prior_prevalence)
    tp_prob = float(sensitivity) ** int(n_consecutive_positive)
    fp_prob = float(false_positive_rate) ** int(n_consecutive_positive)
    numerator = tp_prob * p_d
    denominator = numerator + fp_prob * (1.0 - p_d)
    posterior = numerator / max(denominator, 1e-15)
    return {
        "primary_metric": round(float(posterior), 6),
        "methodological_flag": bool(posterior > 0.50),
    }


# 2. Inferential Statistics: Welch-Satterthwaite T-Test under Unequal Variances
def solve_welch_heteroscedastic_test(
    sample_a: List[float] | np.ndarray,
    sample_b: List[float] | np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float | bool]:
    """
    Compute exact Welch's t-test (equal_var=False) with Welch-Satterthwaite effective degrees
    of freedom, avoiding pooled-variance Type I error inflation when sample sizes and variances differ.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    res = stats.ttest_ind(a, b, equal_var=False)
    p_val = float(res.pvalue)
    return {
        "primary_metric": round(p_val, 6),
        "methodological_flag": bool(p_val < alpha),
    }


# 3. Multicollinearity Diagnostics: Variance Inflation Factor (VIF)
def solve_vif_diagnostics(
    design_matrix: List[List[float]] | np.ndarray,
    vif_threshold: float = 10.0,
) -> Dict[str, float | bool]:
    """
    Compute exact Variance Inflation Factor (VIF_j = 1 / (1 - R_j^2)) for each predictor column j
    including an explicit intercept term (matching statsmodels.stats.outliers_influence.variance_inflation_factor).
    """
    X = np.asarray(design_matrix, dtype=float)
    n, k = X.shape
    vifs: List[float] = []
    for j in range(k):
        y_j = X[:, j]
        other_cols = [X[:, i] for i in range(k) if i != j]
        X_others = np.column_stack([np.ones(n)] + other_cols)
        beta, _, _, _ = np.linalg.lstsq(X_others, y_j, rcond=None)
        y_hat = X_others @ beta
        ss_res = float(np.sum((y_j - y_hat) ** 2))
        ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
        r2_j = 1.0 - (ss_res / max(ss_tot, 1e-12))
        vif_j = 1.0 / max(1.0 - r2_j, 1e-12)
        vifs.append(vif_j)

    max_vif = float(np.max(vifs))
    return {
        "primary_metric": round(max_vif, 6),
        "methodological_flag": bool(max_vif >= vif_threshold),
    }


# 4. Non-Stationary Time-Series Modeling: First-Differencing Spurious Regression Defense
def solve_nonstationary_timeseries_regression(
    series_x: List[float] | np.ndarray,
    series_y: List[float] | np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float | bool]:
    """
    Evaluate relationship between two I(1) non-stationary time series by first-differencing
    (Delta y_t vs Delta x_t) before computing OLS t-statistic p-value, preventing Granger-Newbold
    spurious regression false positives on independent random walks.
    """
    x = np.asarray(series_x, dtype=float)
    y = np.asarray(series_y, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    slope, intercept, r_val, p_val, std_err = stats.linregress(dx, dy)
    return {
        "primary_metric": round(float(p_val), 6),
        "methodological_flag": bool(p_val < alpha),
    }


# 5. Target Leakage Prior to Train/Test Partitioning
def solve_leak_free_target_encoding_eval(
    categories: List[str],
    features: List[float],
    targets: List[int],
    split_index: int,
    smoothing_m: float = 5.0,
) -> Dict[str, float | bool]:
    """
    Compute leak-free out-of-sample RMSE on the test partition [split_index:] by fitting
    category target means and feature standardization strictly on [:split_index].
    """
    cats = np.asarray(categories, dtype=str)
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)

    cats_tr, cats_te = cats[:split_index], cats[split_index:]
    x_tr, x_te = x[:split_index], x[split_index:]
    y_tr, y_te = y[:split_index], y[split_index:]

    prior = float(np.mean(y_tr))
    df_tr = pd.DataFrame({"c": cats_tr, "y": y_tr})
    agg = df_tr.groupby("c")["y"].agg(["count", "mean"])
    mapping = {
        idx: float((row["count"] * row["mean"] + smoothing_m * prior) / (row["count"] + smoothing_m))
        for idx, row in agg.iterrows()
    }

    enc_tr = np.array([mapping.get(c, prior) for c in cats_tr], dtype=float)
    enc_te = np.array([mapping.get(c, prior) for c in cats_te], dtype=float)

    mu_x = float(np.mean(x_tr))
    sd_x = float(np.std(x_tr, ddof=0)) or 1.0
    x_tr_s = (x_tr - mu_x) / sd_x
    x_te_s = (x_te - mu_x) / sd_x

    X_design_tr = np.column_stack([np.ones(len(x_tr_s)), x_tr_s, enc_tr])
    X_design_te = np.column_stack([np.ones(len(x_te_s)), x_te_s, enc_te])

    # Ridge-stabilized linear predictor fit strictly on training split
    A = X_design_tr.T @ X_design_tr + 1.0 * np.eye(3)
    w = np.linalg.solve(A, X_design_tr.T @ y_tr)
    preds_te = np.clip(X_design_te @ w, 0.0, 1.0)
    rmse = float(np.sqrt(np.mean((y_te - preds_te) ** 2)))

    return {
        "primary_metric": round(rmse, 6),
        "methodological_flag": True,  # Strictly zero leakage
    }


# 6. Multiple Testing Corrections (Benjamini-Hochberg FDR) & PR-AUC under Skewed Priors
def solve_multiple_testing_and_skewed_metric(
    raw_p_values: List[float],
    y_true: List[int],
    y_scores: List[float],
    fdr_q: float = 0.05,
) -> Dict[str, float | bool]:
    """
    Apply Benjamini-Hochberg False Discovery Rate (FDR) correction to a family of m hypothesis tests
    AND compute PR-AUC (rather than ROC-AUC) under skewed binary labels.
    """
    p_arr = np.asarray(raw_p_values, dtype=float)
    m = len(p_arr)
    order = np.argsort(p_arr)
    ranked_p = p_arr[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # Benjamini-Hochberg adjusted p-values: p_adj_(i) = min_{k >= i} (m / k) * p_(k)
    raw_adj = (m / ranks) * ranked_p
    bh_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    bh_sorted = np.clip(bh_sorted, 0.0, 1.0)
    bh_adj = np.empty(m, dtype=float)
    bh_adj[order] = bh_sorted
    n_significant_bh = int(np.sum(bh_adj < fdr_q))

    # Compute exact PR-AUC on skewed binary labels
    y_t = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)
    n_pos = int(np.sum(y_t == 1))
    sort_idx = np.argsort(-scores, kind="mergesort")
    y_sorted = y_t[sort_idx]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / (tp + fp)
    pr_auc = float(np.sum(prec[y_sorted == 1]) / max(n_pos, 1))

    return {
        "primary_metric": round(pr_auc, 6),
        "methodological_flag": bool(n_significant_bh > 0),
    }
