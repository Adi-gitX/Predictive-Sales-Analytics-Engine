# Predictive Sales Analytics Engine

## Open First
Start with [submission/Final_Submission_Notebook.ipynb](submission/Final_Submission_Notebook.ipynb). It is the simplest evaluator-facing artifact and presents the phase-1 baseline submission from problem framing through final conclusion.

## Problem
This project predicts whether a customer will make another purchase within 180 days after the first delivered order using the Olist Brazilian e-commerce dataset. The task is framed as a CRM retention problem rather than a sentiment or review-score benchmark.

## Approach
The pipeline builds a leakage-safe first-order customer cohort, aggregates raw business tables into customer-level features, compares classical baseline models, and uses baseline explainability to interpret the final recommendation.

## Why This Is Leakage-Safe
Each modeling row represents only the customer's first delivered order. The scoring time is defined as the later of delivery time and review creation time, and the target only checks for purchases after that point. Temporal train, validation, and test splits preserve chronology instead of randomly mixing future rows into earlier decisions.

## Final Phase-1 Results
- Recommended final model: `tabular_rf`
- Test PR-AUC: `0.0229`
- Test ROC-AUC: `0.5698`
- Precision@10%: `0.0255`
- Lift@10%: `1.5756`

## Recommended Final Model
`tabular_rf` is the phase-1 final recommendation because it is the strongest held-out baseline on PR-AUC while remaining simple, interpretable, and easy to explain.

## Repo Structure
```text
.
├── README.md
├── requirements.txt
├── configs/default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
├── models/
├── final_outputs/
└── submission/
```

## Run Order
```bash
python scripts/01_build_dataset.py
python scripts/02_train_baselines.py
python scripts/04_explain.py
python scripts/05_validate_submission.py
```

## Main Files
- Final notebook: [submission/Final_Submission_Notebook.ipynb](submission/Final_Submission_Notebook.ipynb)
- Section notebooks: [notebooks](notebooks)
- Config: [configs/default.yaml](configs/default.yaml)
- Final summaries: [final_outputs](final_outputs)
- Validation script: [scripts/05_validate_submission.py](scripts/05_validate_submission.py)

## Section Notebook Order
1. [notebooks/01_Literature_Review.ipynb](notebooks/01_Literature_Review.ipynb)
2. [notebooks/02_Dataset_Quality_and_EDA.ipynb](notebooks/02_Dataset_Quality_and_EDA.ipynb)
3. [notebooks/03_Feature_Engineering.ipynb](notebooks/03_Feature_Engineering.ipynb)
4. [notebooks/04_Model_Application_and_Evaluation.ipynb](notebooks/04_Model_Application_and_Evaluation.ipynb)
5. [notebooks/05_Explainability_and_Theoretical_Rigor.ipynb](notebooks/05_Explainability_and_Theoretical_Rigor.ipynb)
6. [submission/Final_Submission_Notebook.ipynb](submission/Final_Submission_Notebook.ipynb)
