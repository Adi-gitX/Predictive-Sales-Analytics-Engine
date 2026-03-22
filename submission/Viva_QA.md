# Viva Q&A

## What is the exact problem statement?
Predict whether a customer will make another purchase within 180 days after the customer's first delivered order.

## Why is this better than predicting review score or sentiment?
Repeat purchase is more actionable for CRM and retention. It directly connects model output to business intervention, while review score is only an indirect proxy for future value.

## How did you avoid leakage?
I used only the first delivered order for each customer and defined `score_time = max(delivery time, review creation time)`. The target only checks for later purchases after that point, so future information is not allowed to influence the feature set.

## What leakage risks did you explicitly remove?
- Using later orders from the same customer in the feature row
- Using any information after `score_time`
- Randomly mixing future rows into earlier examples
- Treating post-outcome aggregates as input features

## Why did you use temporal splitting instead of random splitting?
This is a time-dependent business prediction problem. A random split can leak future behavioral patterns into earlier examples and produce unrealistically optimistic results. A chronological split is more honest.

## Why is baseline-only a defensible phase-1 strategy?
Phase 1 is meant to establish a correct, reproducible, and explainable benchmark before adding complexity. A strong baseline ladder shows whether the signal is real and which feature families matter, which is academically stronger than jumping straight to a harder model.

## Why is accuracy not the main metric?
The dataset is highly imbalanced, with only about 1.79% positives. Accuracy would mostly reward predicting the majority class, so PR-AUC and Lift@10% are more meaningful ranking metrics.

## Why did you choose these evaluation metrics?
- `PR-AUC` measures ranking quality under severe class imbalance
- `ROC-AUC` gives a threshold-free discrimination view
- `Precision@10%` measures top-segment targeting quality
- `Lift@10%` translates directly to CRM campaign usefulness
- `F1` is used only to pick a reasonable validation threshold

## Which phase-1 model is the final recommendation?
`tabular_rf` is the final phase-1 recommendation because it is the strongest held-out baseline on PR-AUC.

## Why did `tabular_lr` win validation but `tabular_rf` become the final recommendation?
`tabular_lr` is the best validation baseline on PR-AUC, but `tabular_rf` is the best held-out test baseline on PR-AUC. Since the final recommendation should reflect the strongest honest held-out ranking performance, `tabular_rf` is the better final choice.

## Why did random forest beat the other baseline models?
The baseline features contain nonlinear business effects such as delivery delay, freight burden, item count, payment mismatch, and review quality. Random forest can model those interactions more flexibly than a linear classifier.

## Why did you still keep text models if they were weaker?
They are still useful scientific baselines. They test whether review language adds predictive value, and their coefficients also provide interpretable customer-experience themes even when text is not the strongest predictive source.

## What are the final test metrics of the recommended model?
- PR-AUC: 0.02292431205973533
- ROC-AUC: 0.5698419168993522
- Precision@10%: 0.02554278416347382
- Lift@10%: 1.5756076467453064
- Threshold: 0.35

## What does Lift@10% = 1.5756 mean in business terms?
It means the top 10% of customers ranked by the model contain positives at about 1.58 times the average rate in the full test set. In practice, this makes the ranked list useful for targeted CRM outreach.

## What were the most important final-model features?
Review score, text length, freight ratio, item count, delivery timing, payment value, and seller-customer state alignment were among the strongest signals.

## Why does text help interpretation more than prediction here?
Review text is missing for many customers and is sparse even when present. That makes text weaker as a predictive input, but still useful for understanding what kinds of customer experiences correlate with stronger or weaker retention.

## What are the main limitations?
- Rare target event
- Large fraction of missing review text
- Portuguese text reduces interpretability for non-Portuguese readers
- Phase 1 uses only baseline models, so it prioritizes simplicity and defensibility over maximum possible performance

## If you had more time, what would you improve next?
I would test probability calibration, richer time-aware validation, better Portuguese text normalization, and deeper error analysis on false positives and false negatives before considering any phase-2 model expansion.
