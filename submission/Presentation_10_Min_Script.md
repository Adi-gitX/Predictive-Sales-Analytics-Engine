# 10-Minute Presentation Script

## Slide 1: Title and Problem
Time: 45 seconds

This project is a predictive sales analytics engine built on the Olist e-commerce dataset.  
The task is to predict whether a customer will purchase again within 180 days after the first delivered order.  
I chose this framing because it is more business-relevant than review-score prediction and directly supports CRM retention decisions.

## Slide 2: Why This Problem Matters
Time: 45 seconds

Retention is usually more valuable than one-time conversion because repeat customers improve lifetime value.  
If we can rank customers by repeat-purchase likelihood after their first order, a business can target the highest-potential segment with discounts, reminders, and loyalty campaigns.

## Slide 3: Dataset and Leakage-Safe Cohort
Time: 1 minute

The project uses the Olist Brazilian e-commerce dataset with customers, orders, items, payments, reviews, products, and sellers.  
I built a customer-level cohort where each row represents only the customer's first delivered order.  
The score time is defined as the maximum of delivery time and review creation time.  
This is important because it prevents future leakage.

## Slide 4: EDA Findings
Time: 1 minute

The key EDA findings were:
- the target is highly imbalanced
- more than half the customers have no review text
- delivery delay, freight burden, and payment behavior look related to repeat purchase
- the raw data is relational, so aggregation quality matters

These findings directly influenced preprocessing and model choice.

## Slide 5: Feature Engineering
Time: 1 minute

I engineered features from six groups:
- review features
- monetary features
- order-complexity features
- delivery features
- categorical context
- log-transformed numeric variables

Examples include freight ratio, payment gap, seller-customer same-state flag, clipped delivery delay, and text-length indicators.

## Slide 6: Baseline Modeling Strategy
Time: 1 minute 15 seconds

I used a clear baseline ladder:
- dummy prior baseline
- review-score logistic regression
- tabular logistic regression
- tabular random forest
- text-only TF-IDF logistic regression
- combined TF-IDF plus tabular logistic regression

This lets me compare simple and interpretable methods honestly before moving beyond phase 1.

## Slide 7: Evaluation Metrics
Time: 45 seconds

Because the target is rare, accuracy is misleading.  
So I focused on:
- PR-AUC
- ROC-AUC
- Precision@10%
- Lift@10%
- F1 for threshold selection

Lift@10% is especially useful because it tells us how concentrated the positives are in the top-ranked customer segment.

## Slide 8: Final Results
Time: 1 minute

The best phase-1 model was `tabular_rf`.

Final test results:
- PR-AUC: 0.0229
- ROC-AUC: 0.5698
- Precision@10%: 0.0255
- Lift@10%: 1.5756

The key conclusion is that structured business features outperform sparse review text on this task even in a baseline-only setup.

## Slide 9: Explainability
Time: 1 minute

I used three kinds of explainability:
- text-term coefficients for the TF-IDF model
- linear coefficients for the combined model
- permutation importance and partial dependence for the final tabular random forest

The most important final-model signals were review score, text length, freight burden, item count, and delivery-related features.  
This makes business sense because repeat purchase is influenced by the quality and friction of the first-order experience.

## Slide 10: Conclusion
Time: 45 seconds

This project delivers a strong phase-1 retention-prediction pipeline with:
- leakage-safe dataset construction
- meaningful feature engineering
- strong baseline comparisons
- a justified final baseline model
- explainability for business interpretation

The final takeaway is simple: for first-order retention prediction in Olist, structured post-purchase business signals carry more predictive value than sparse review text.

## Demo Flow
Time: 1 minute

If asked to demo live, open:
- `../notebooks/01_Literature_Review.ipynb`
- `../notebooks/02_EDA.ipynb`
- `../notebooks/03_Preprocessing.ipynb`
- `../notebooks/04_Feature_Engineering.ipynb`
- `../notebooks/05_Baseline_ML_Model.ipynb`

Show these sections in order:
1. Literature framing
2. EDA highlights
3. Preprocessing choices
4. Feature engineering story
5. Baseline model comparison and final conclusion
