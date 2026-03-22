# Rubric Alignment

## Literature Review
- [notebooks/01_Literature_Review.ipynb](../notebooks/01_Literature_Review.ipynb): explains why retention prediction is a stronger framing than review-score prediction and justifies the baseline ladder from prior work.

## Dataset Quality and EDA
- [notebooks/02_EDA.ipynb](../notebooks/02_EDA.ipynb): links class imbalance, skew, temporal structure, geographic variation, and text behavior directly to downstream preprocessing and modeling choices.
- [final_outputs/dataset_summary.json](../final_outputs/dataset_summary.json): provides the final dataset dimensions, split sizes, target rates, and missing-text statistics used throughout the project.
- [submission/figures](figures): contains the baseline EDA figures used in both the notebook and report.

## Preprocessing
- [notebooks/03_Preprocessing.ipynb](../notebooks/03_Preprocessing.ipynb): documents missing-value review, outlier treatment, categorical encoding, and feature scaling in separate evaluator-friendly notebook cells.
- [data/processed/train.csv](../data/processed/train.csv): is the cleaned modeling split used to demonstrate the preprocessing decisions.

## Feature Engineering
- [notebooks/04_Feature_Engineering.ipynb](../notebooks/04_Feature_Engineering.ipynb): documents the raw-to-feature mapping and explains why each engineered feature family matters for repeat purchase.
- [src/sales_analytics/features.py](../src/sales_analytics/features.py): contains the reusable feature-construction logic for the customer-level dataset.
- [data/processed/train.csv](../data/processed/train.csv): shows the final engineered modeling schema that the baseline models actually consume.

## Baseline ML Model
- [notebooks/05_Baseline_ML_Model.ipynb](../notebooks/05_Baseline_ML_Model.ipynb): assembles the final feature matrix, trains the baseline models, compares validation performance, and reports the held-out test evaluation.
- [final_outputs/metrics_baselines.csv](../final_outputs/metrics_baselines.csv): is the canonical baseline metric table for validation and test evaluation.
- [final_outputs/best_model_summary.json](../final_outputs/best_model_summary.json): records the best validation baseline, best held-out test baseline, and final recommended phase-1 model.
- [submission/Viva_QA.md](Viva_QA.md): captures the reasoning needed to defend the methodology and results in oral questioning.

## Repository and Code Quality
- [README.md](../README.md): gives the problem statement, leakage-safe framing, notebook order, and cleaned submission structure in a portable form.
- [scripts](../scripts): contains the reproducible phase-1 pipeline and final validation script.
- [src](../src): contains the modular baseline implementation used by the scripts and notebooks.
- [configs/default.yaml](../configs/default.yaml): centralizes the project configuration.
- [scripts/05_validate_submission.py](../scripts/05_validate_submission.py): performs the final integrity check for syntax, notebook structure, paths, and submission cleanup.

## Viva Readiness
- [submission/Viva_QA.md](Viva_QA.md): gives concise, defensible answers for the likely technical questions.
- [notebooks](../notebooks): provide deeper backup material if the evaluator asks for more detail on any section.

