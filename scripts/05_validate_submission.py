from __future__ import annotations

import ast
import json
import os
import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*", category=UserWarning)

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from sales_analytics.metrics import compute_metrics
from sales_analytics.preprocessing import get_baseline_tabular_feature_columns, select_baseline_tabular_frame


EXPECTED_METRIC_COLUMNS = [
    "model",
    "split",
    "pr_auc",
    "roc_auc",
    "precision_at_k",
    "lift_at_k",
    "f1",
    "precision",
    "recall",
    "brier",
    "threshold",
]
EXPECTED_EXPLAINABILITY_FILES = [
    "tabular_rf_permutation.csv",
    "combined_coefficients.csv",
    "text_terms.csv",
    "text_terms.png",
    "partial_dependence/pdp_delivery_delay_days_clipped.png",
    "partial_dependence/pdp_freight_ratio.png",
    "partial_dependence/pdp_review_score.png",
]
FORBIDDEN_PATH_FRAGMENT = "/Users/kammatiaditya/"


def check(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def validate_python_syntax(failures: list[str]) -> None:
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"Syntax error in {path}: {exc}")
            print(f"[FAIL] Syntax error in {path}: {exc}")


def validate_processed_data(failures: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = PROJECT_ROOT / "data/processed"
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    check(list(train_df.columns) == list(val_df.columns) == list(test_df.columns), "Processed splits share the same schema.", failures)
    check(len(train_df.columns) == 46, "Processed splits contain 46 columns.", failures)
    check(not train_df["order_id"].duplicated().any(), "Train split has unique order_id values.", failures)
    check(not val_df["order_id"].duplicated().any(), "Validation split has unique order_id values.", failures)
    check(not test_df["order_id"].duplicated().any(), "Test split has unique order_id values.", failures)
    check(set(train_df["order_id"]).isdisjoint(set(val_df["order_id"])), "Train and validation splits are disjoint.", failures)
    check(set(train_df["order_id"]).isdisjoint(set(test_df["order_id"])), "Train and test splits are disjoint.", failures)
    check(set(val_df["order_id"]).isdisjoint(set(test_df["order_id"])), "Validation and test splits are disjoint.", failures)
    check(pd.to_datetime(train_df["score_time"]).is_monotonic_increasing, "Train split is temporally ordered by score_time.", failures)
    check(pd.to_datetime(val_df["score_time"]).is_monotonic_increasing, "Validation split is temporally ordered by score_time.", failures)
    check(pd.to_datetime(test_df["score_time"]).is_monotonic_increasing, "Test split is temporally ordered by score_time.", failures)
    return train_df, val_df, test_df


def validate_summary_files(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, failures: list[str]) -> None:
    summary = json.loads((PROJECT_ROOT / "final_outputs/dataset_summary.json").read_text(encoding="utf-8"))
    check("n_processed_columns" in summary, "Dataset summary reports n_processed_columns.", failures)
    check("n_model_input_features_baseline" in summary, "Dataset summary reports baseline input-feature count.", failures)
    check(summary["n_processed_columns"] == len(train_df.columns), "Dataset summary matches processed-column count.", failures)
    check(summary["n_model_input_features_baseline"] == len(get_baseline_tabular_feature_columns()), "Dataset summary matches baseline model feature count.", failures)

    best_model = json.loads((PROJECT_ROOT / "final_outputs/best_model_summary.json").read_text(encoding="utf-8"))
    required_keys = {
        "overall_validation_best_model",
        "overall_test_best_model",
        "recommended_model",
        "recommendation_reason",
        "recommended_model_test_metrics",
        "n_total_models_evaluated",
    }
    check(required_keys.issubset(best_model.keys()), "Best-model summary exposes the final canonical fields.", failures)


def validate_metric_files(failures: list[str]) -> pd.DataFrame:
    baseline_metrics = pd.read_csv(PROJECT_ROOT / "final_outputs/metrics_baselines.csv")
    check(list(baseline_metrics.columns) == EXPECTED_METRIC_COLUMNS, "Baseline metric schema is correct.", failures)
    metrics = baseline_metrics.copy()
    for column in ["pr_auc", "roc_auc", "precision_at_k", "f1", "precision", "recall", "brier"]:
        check(metrics[column].between(0, 1).all(), f"Metric column {column} stays within [0, 1].", failures)
    check((metrics["lift_at_k"] >= 0).all(), "Lift@10% values are non-negative.", failures)
    check(metrics["model"].nunique() == 6, "Exactly six baseline models are evaluated across the phase-1 ladder.", failures)
    return metrics


def validate_models_and_metrics(test_df: pd.DataFrame, failures: list[str]) -> None:
    model_paths = sorted((PROJECT_ROOT / "models").rglob("*.joblib"))
    check(len(model_paths) == 6, "All expected baseline joblib artifacts are present.", failures)
    for path in model_paths:
        try:
            joblib.load(path)
        except Exception as exc:
            failures.append(f"Could not load {path}: {exc}")
            print(f"[FAIL] Could not load {path}: {exc}")

    baseline_model = joblib.load(PROJECT_ROOT / "models/baselines/tabular_rf.joblib")
    best_model = json.loads((PROJECT_ROOT / "final_outputs/best_model_summary.json").read_text(encoding="utf-8"))
    expected = best_model["recommended_model_test_metrics"]
    y_test = test_df["target_repeat_within_180d"]
    baseline_features = select_baseline_tabular_frame(test_df)
    proba = baseline_model.predict_proba(baseline_features)[:, 1]
    recomputed = compute_metrics(y_test, proba, threshold=float(expected["threshold"]))
    all_match = True
    for key in ["pr_auc", "roc_auc", "precision_at_k", "lift_at_k", "f1", "precision", "recall", "brier", "threshold"]:
        all_match = all_match and abs(float(recomputed[key]) - float(expected[key])) < 1e-12
    check(all_match, "Recommended-model test metrics exactly reproduce from the saved joblib model.", failures)


def validate_explainability_outputs(failures: list[str]) -> None:
    explainability_dir = PROJECT_ROOT / "final_outputs/explainability"
    for relative_path in EXPECTED_EXPLAINABILITY_FILES:
        check((explainability_dir / relative_path).exists(), f"Explainability artifact exists: {relative_path}", failures)

    permutation = pd.read_csv(explainability_dir / "tabular_rf_permutation.csv")
    forbidden_features = {"target_repeat_within_180d", "order_id", "customer_unique_id", "score_time", "review_text"}
    check(forbidden_features.isdisjoint(set(permutation["feature"])), "Permutation importance contains only real tabular model features.", failures)
    check(set(permutation["feature"]) == set(get_baseline_tabular_feature_columns()), "Permutation importance covers the exact baseline feature set.", failures)
    probe_paths = sorted(explainability_dir.rglob("*probe*"))
    check(not probe_paths, "No probe explainability artifacts remain in final_outputs.", failures)


def validate_notebooks_and_links(failures: list[str]) -> None:
    notebook_paths = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")) + [PROJECT_ROOT / "submission/Final_Submission_Notebook.ipynb"]
    for path in notebook_paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
        executed = all(cell.get("execution_count") is not None for cell in code_cells)
        has_errors = any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
        check(executed, f"{path.relative_to(PROJECT_ROOT)} has all code cells executed.", failures)
        check(not has_errors, f"{path.relative_to(PROJECT_ROOT)} has no error outputs.", failures)
        check(FORBIDDEN_PATH_FRAGMENT not in path.read_text(encoding="utf-8"), f"{path.relative_to(PROJECT_ROOT)} contains no absolute local paths.", failures)

    for path in sorted(PROJECT_ROOT.rglob("*.md")):
        check(FORBIDDEN_PATH_FRAGMENT not in path.read_text(encoding="utf-8"), f"{path.relative_to(PROJECT_ROOT)} contains no absolute local paths.", failures)


def validate_repo_cleanliness(failures: list[str]) -> None:
    pycache_dirs = list(PROJECT_ROOT.rglob("__pycache__"))
    check(not pycache_dirs, "No __pycache__ directories remain in the repo tree.", failures)
    check(not (PROJECT_ROOT / "models/advanced").exists(), "No advanced-model directory remains in the phase-1 repo.", failures)
    check(not (PROJECT_ROOT / "final_outputs/metrics_advanced.csv").exists(), "No advanced-metrics file remains in the phase-1 repo.", failures)
    check(not (PROJECT_ROOT / "scripts/03_train_advanced.py").exists(), "No advanced-training script remains in the phase-1 repo.", failures)
    check(not (PROJECT_ROOT / "final_outputs/explainability/advanced_tabular_permutation.csv").exists(), "No advanced explainability artifact remains in the phase-1 repo.", failures)


def main() -> int:
    failures: list[str] = []
    validate_python_syntax(failures)
    train_df, val_df, test_df = validate_processed_data(failures)
    validate_summary_files(train_df, val_df, test_df, failures)
    validate_metric_files(failures)
    validate_models_and_metrics(test_df, failures)
    validate_explainability_outputs(failures)
    validate_notebooks_and_links(failures)
    validate_repo_cleanliness(failures)

    if failures:
        print("\nValidation failed with the following issues:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed: the repo matches the final submission checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
