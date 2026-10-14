# Adversarial Quantitative Reasoning & Statistical Evaluation Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Problems: 210 Verified](https://img.shields.io/badge/Benchmark-210%20Problems-purple.svg)](#benchmark-taxonomy)
[![JSON Schema Validated](https://img.shields.io/badge/Schema-Draft%202020--12-brightgreen.svg)](schemas/problem_schema.json)

A **210-problem evaluation benchmark and automated grading harness** targeting subtle statistical, econometric, and methodological errors commonly made by Large Language Models (LLMs) and autonomous coding agents in quantitative data science workflows.

## Key Capabilities & Architecture

- **200+ Problem Adversarial Evaluation Benchmark (`src/benchmark_catalog.py`, `benchmarks/problems_210.json`)**: Generates and validates **210 parameterized adversarial tasks** across 6 core quantitative domains, governed by a strict JSON Schema (`schemas/problem_schema.json`).
- **Verified Reference Solutions & Ground Truths (`src/reference_solutions.py`)**: Unit-tested mathematical implementations spanning Bayesian conditional probability under skewed priors, Welch-Satterthwaite inferential statistics, Variance Inflation Factor ($	ext{VIF} = rac{1}{1 - R_j^2}$) multicollinearity diagnostics, and Augmented Dickey-Fuller (ADF) / walk-forward non-stationary time-series modeling.
- **Curated ML Antipattern Traps (`src/adversarial_traps.py`)**: Models canonical failure modes of autonomous coding agents—including pre-split imputation/scaling target leakage, metric mismatch (optimizing ROC-AUC instead of PR-AUC under $< 2\%$ class prevalence), and false-positive $p$-value exploitation without Benjamini-Hochberg (FDR) or Holm-Bonferroni multiple testing corrections.
- **Deterministic Automated Grading Harness (`src/grading_harness.py`)**: Evaluates candidate code/functions against runtime execution criteria, numerical tolerances, AST/methodological assertions, and adversarial edge cases.

---

## Benchmark Taxonomy (210 Problems)

| Domain ID | Quantitative / Statistical Category | Problem Count | Targeted LLM / Agent Failure Mode |
| :--- | :--- | :---: | :--- |
| `PROB_BAYES` | Probability Theory & Skewed Conditional Priors | `35` | Base-rate neglect, Simpson's aggregation reversal, Berkson's selection bias |
| `INFER_STATS` | Inferential Statistics & Hypothesis Testing | `35` | Student's $t$-test under heteroscedasticity instead of Welch-Satterthwaite $t$-test |
| `MULTICOLL_VIF` | Multicollinearity Diagnostics (VIF) & Linear Models | `35` | Omitting intercept in $R_j^2$ or failing to flag collinear predictors ($	ext{VIF} > 10$) |
| `TIMESERIES_NS` | Non-Stationary Time-Series & Spurious Regression | `35` | Regressing $I(1)$ random walks without first-differencing or shuffling temporal folds |
| `LEAKAGE_TRAP` | Target Leakage Prior to Train/Test Partitioning | `35` | Global target encoding / imputation prior to train/test split |
| `MULTI_TEST_PR` | Metric Mismatch & Multiple Testing Corrections | `35` | Using ROC-AUC under $< 2\%$ prevalence; raw $p < 0.05$ without Benjamini-Hochberg / Holm |
| **Total** | **All 6 Quantitative Evaluation Domains** | **210** | **Deterministic JSON Schema + Runtime Execution Harness** |

---

## Repository Structure

```text
adversarial-quant-eval-suite/
├── .github/workflows/ci.yml        # Automated GitHub Actions evaluation & unit test workflow
├── Makefile                        # Build catalog, validate schema, and run grading harness
├── pytest.ini                      # Test configuration
├── requirements.txt                # Pinned dependencies
├── schemas/
│   └── problem_schema.json         # Strict JSON Schema for benchmark tasks & ground truths
├── benchmarks/
│   └── problems_210.json           # Generated 210-problem benchmark catalog
├── src/
│   ├── __init__.py
│   ├── reference_solutions.py      # Verified mathematical ground-truth implementations
│   ├── adversarial_traps.py        # Canonical flawed agent implementations & trap detectors
│   ├── benchmark_catalog.py        # 210-problem generator & JSON Schema validator
│   └── grading_harness.py          # Automated execution grading harness & CLI scorecard
└── tests/
    ├── __init__.py
    └── test_eval_suite.py          # Unit tests verifying ground truths, traps, and 210 problems
```

---

## Quickstart

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate & Validate the 210-Problem Catalog + Run Grading Harness

```bash
python3 -m src.grading_harness --generate-catalog --run-evaluation
```

### 3. Run Unit Tests

```bash
pytest -v
```
