from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from sales_analytics.config import load_config
from sales_analytics.data_loading import load_raw_tables
from sales_analytics.features import build_order_level_features
from sales_analytics.preprocessing import get_baseline_tabular_feature_columns
from sales_analytics.split import temporal_split
from sales_analytics.target import build_customer_first_order_cohort
from sales_analytics.utils import ensure_dir, save_json, set_seed


def main():
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    set_seed(config["seed"])
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    results_dir = PROJECT_ROOT / config["data"]["results_dir"]
    ensure_dir(processed_dir)
    ensure_dir(results_dir)

    tables = load_raw_tables(config, PROJECT_ROOT)
    cohort = build_customer_first_order_cohort(tables, repeat_window_days=config["target"]["repeat_window_days"])
    modeling_df = build_order_level_features(tables, cohort).sort_values("score_time").reset_index(drop=True)
    train_df, val_df, test_df = temporal_split(
        modeling_df,
        time_col="score_time",
        train_fraction=config["split"]["train_fraction"],
        val_fraction=config["split"]["val_fraction"],
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    summary = {
        "n_modeling_rows": int(len(modeling_df)),
        "n_processed_columns": int(len(modeling_df.columns)),
        "n_model_input_features_baseline": int(len(get_baseline_tabular_feature_columns())),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "target_rate_all": float(modeling_df["target_repeat_within_180d"].mean()),
        "target_rate_train": float(train_df["target_repeat_within_180d"].mean()),
        "target_rate_val": float(val_df["target_repeat_within_180d"].mean()),
        "target_rate_test": float(test_df["target_repeat_within_180d"].mean()),
        "score_time_min": str(modeling_df["score_time"].min().date()),
        "score_time_max": str(modeling_df["score_time"].max().date()),
        "missing_text_rate": float((1 - modeling_df["text_present"].mean())),
        "average_review_score": float(modeling_df["review_score"].mean()),
        "median_total_price": float(modeling_df["total_price"].median()),
        "median_delivery_days": float(modeling_df["delivery_days"].median()),
    }
    save_json(summary, results_dir / "dataset_summary.json")
    print(pd.Series(summary))

if __name__ == "__main__":
    main()
