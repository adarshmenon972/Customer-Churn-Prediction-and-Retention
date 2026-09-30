# Key Project Details

| Field | Details |
|---|---|
| **Project name** | ChurnGuard - End-to-End Customer Churn Prediction & Retention-ROI Optimisation System |
| **Brief description** | Predicts which subscription customers are likely to leave, explains *why* for each customer, and picks the contact threshold that maximises retention-campaign profit (not just accuracy). Includes hyperparameter tuning, probability calibration, a REST API for real-time scoring, automated tests and a Docker image. |
| **Problem solved** | Retaining an existing customer is far cheaper than acquiring a new one, but contacting everyone wastes budget. The system ranks customers by churn risk, tells the retention team the top reasons per customer, and recommends who is worth contacting given contact cost and expected save rate. |
| **AI/ML technologies & tools** | **Modelling:** LightGBM (gradient-boosted trees), Logistic Regression baseline. **Tuning:** Optuna (Bayesian/TPE search, stratified 3-fold CV on PR-AUC). **Calibration:** Isotonic regression. **Explainability:** SHAP (TreeExplainer, global + per-customer drivers). **Feature engineering / preprocessing:** pandas, NumPy, scikit-learn pipelines. **Decision optimisation:** cost-sensitive threshold selection on campaign profit. **Serving:** FastAPI + Pydantic validation, Uvicorn. **MLOps:** joblib model artifacts, pytest test suite, Docker. **Visualisation:** Matplotlib. |
| **Key results (synthetic data, held-out test set)** | ROC-AUC 0.893 (vs 0.886 logistic baseline); PR-AUC 0.755; top-decile lift 3.15x; retention-campaign net profit ~$106.7K with the optimised threshold vs ~$85.4K when contacting everyone (+25%). |
| **GitHub repository link** | `https://github.com/<your-username>/churnguard` (replace after you push - see below) |

## Publish to GitHub
```bash
cd churnguard
git init && git add . && git commit -m "ChurnGuard: churn prediction + retention ROI optimisation"
git branch -M main
git remote add origin https://github.com/<your-username>/churnguard.git
git push -u origin main
```
