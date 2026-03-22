# Viva Q&A

## What is the exact problem statement?
Predict whether a customer will make another purchase within 180 days after the customer's first delivered order.

## Why did you not predict review score or sentiment?
Repeat purchase is more actionable for CRM and retention. It connects directly to business value while still allowing the use of both text and structured data.

## How did you avoid leakage?
I used only the first delivered order for each customer and defined `score_time = max(delivery time, review creation time)`. Only information available up to score time was used to construct features and labels.

## Why did you use temporal splitting instead of random splitting?
This is a time-dependent business prediction task. A random split could leak future behavioral patterns into earlier examples and produce optimistic results.

## Why is accuracy not the main metric?
The dataset is highly imbalanced, with only about 1.79% positives. Accuracy would mostly reward predicting the majority class, so PR-AUC and Lift@10% are more meaningful.

## Which phase-1 model is the final recommendation?
`tabular_rf` is the final phase-1 recommendation because it is the strongest held-out baseline on PR-AUC.

## Why did tabular random forest beat the other baseline models?
The baseline features mix nonlinear business effects such as delivery delay, freight burden, item count, and review signals. Random forest can capture those interactions better than a linear model.

## Why did you still keep text models if they were weaker?
They were useful as baselines and for interpretation. The text-term outputs still reveal customer-experience themes even though text was not the strongest predictive source.

## What are the final test metrics?
- PR-AUC: 0.0229
- ROC-AUC: 0.5698
- Precision@10%: 0.0255
- Lift@10%: 1.5756

## What does Lift@10% = 1.5756 mean?
It means the top 10% of customers ranked by the model contain positives at about 1.58 times the average rate in the full test set. That is useful for targeted CRM campaigns.

## What were the most important final-model features?
Review score, text length, freight ratio, item count, delivery days, payment value, and seller-customer state alignment were among the strongest.

## What are the main limitations?
- Rare target event
- Large fraction of missing review text
- Portuguese text reduces interpretability for non-Portuguese readers
- Phase 1 uses only baseline models, so it favors simplicity over maximum possible performance

## If you had more time, what would you improve?
I would test probability calibration, richer time-aware validation, better Portuguese text normalization, and targeted error analysis on false positives and false negatives.
