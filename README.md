# ChurnGuard

End-to-end machine learning system for **customer churn prediction, explanation and retention-ROI optimisation**.

## Why this is more than a notebook
| Stage | What it does |
|---|---|
| Feature engineering | Shared `add_features()` used identically in training and serving (no train/serve skew) |
| Modelling | LightGBM tuned with **Optuna** (TPE, stratified CV on PR-AUC) vs a logistic-regression baseline |
| Calibration | Isotonic regression fitted on a separate validation split so scores behave like real probabilities |
| Decision layer | Threshold chosen to **maximise campaign profit** given contact cost and save rate, not F1/accuracy |
| Explainability | **SHAP** global importance plot + top-3 reasons per customer returned by the API |
| Serving | **FastAPI** `/predict` with strict input validation, risk tiers and recommended actions |
| Quality | `pytest` suite covering data, training artifacts, API behaviour and validation; Dockerfile |

Data is split 60/20/20 (train / validation / test). Validation is used for calibration and threshold choice; the test set is touched only for reporting.

## Quick start
```bash
pip install -e ".[dev]"
python -m churnguard.train --n-samples 20000 --n-trials 30 --out artifacts
pytest
uvicorn churnguard.api:app --port 8000
```
Use your own data: `python -m churnguard.train --csv data.csv --target Churn --value-col annual_revenue`
(missing engineered-feature source columns are skipped automatically).

## Results (synthetic data, held-out test set)
| Model | ROC-AUC | PR-AUC | Brier | Log-loss |
|---|---|---|---|---|
| Logistic Regression | 0.886 | 0.748 | 0.1145 | 0.361 |
| LightGBM (tuned) | 0.893 | 0.755 | 0.1121 | 0.355 |
| LightGBM + isotonic calibration | 0.893 | 0.741 | 0.1126 | 0.355 |

**Business impact** (contact cost $15, 35% save rate): chosen threshold 0.14 targets ~2,000 of 4,000 test customers; net profit ~$106.7K vs ~$85.4K contacting everyone. Top-decile lift 3.15x.

**Honest notes**
- Results are on *synthetic* data with engineered non-linear effects; expect different numbers on real data. The boosting advantage over logistic regression is modest here and is larger when real data has stronger interactions.
- The raw LightGBM scores were already well calibrated, so isotonic calibration mainly acts as a safeguard (it slightly lowers PR-AUC because it creates ties). Keep it if you need trustworthy probabilities for the profit calculation.
- Correlation is not causation: SHAP explains the model's predictions, not what would change a customer's behaviour. Validate retention offers with an A/B test (uplift modelling is a natural next step).

## API example
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
 "customers":[{"tenure_months":3,"contract":"Month-to-month","internet_service":"Fiber",
 "monthly_charge":105,"num_addons":0,"support_tickets_90d":4,"late_payments_12m":3,
 "avg_weekly_usage_hrs":2,"usage_change_pct":-50,"price_increase_pct":15,
 "autopay":"No","payment_method":"Cash"}]}'
```
Returns `churn_probability`, `risk_tier`, `recommended_action` and `top_drivers` (SHAP).

## Layout
```
src/churnguard/  data.py  features.py  train.py  model.py  api.py
tests/           test_pipeline.py
artifacts/       metrics.json + plots (model file created by training)
```

## Ideas to extend
Uplift modelling for who is *persuadable*, survival analysis (time-to-churn), drift monitoring, a Streamlit dashboard, MLflow experiment tracking.
