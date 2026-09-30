import os

import pytest
from fastapi.testclient import TestClient

from churnguard import api
from churnguard.data import TARGET, generate_synthetic
from churnguard.train import run

SAMPLE = {
    "tenure_months": 3, "contract": "Month-to-month", "internet_service": "Fiber",
    "monthly_charge": 105.0, "num_addons": 0, "support_tickets_90d": 4,
    "late_payments_12m": 3, "avg_weekly_usage_hrs": 2.0, "usage_change_pct": -50.0,
    "price_increase_pct": 15.0, "autopay": "No", "payment_method": "Cash",
}
SAFE = {**SAMPLE, "tenure_months": 60, "contract": "Two year", "internet_service": "DSL",
        "monthly_charge": 60.0, "num_addons": 4, "support_tickets_90d": 0, "late_payments_12m": 0,
        "avg_weekly_usage_hrs": 20.0, "usage_change_pct": 10.0, "price_increase_pct": 0.0,
        "autopay": "Yes", "payment_method": "Card"}


def test_data_generation():
    df = generate_synthetic(3000)
    assert 0.18 < df[TARGET].mean() < 0.34
    assert df.isna().sum().sum() == 0


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    metrics = run(n_samples=5000, n_trials=3, out_dir=str(out))
    return out, metrics


def test_training_artifacts_and_quality(trained):
    out, metrics = trained
    for f in ["churn_model.joblib", "metrics.json", "shap_summary.png", "calibration.png"]:
        assert (out / f).exists()
    assert metrics["lightgbm_calibrated"]["roc_auc"] > 0.75
    assert metrics["lightgbm_calibrated"]["brier"] <= metrics["lightgbm_raw"]["brier"] + 0.01


def test_api_end_to_end(trained):
    out, _ = trained
    os.environ["CHURN_MODEL_PATH"] = str(out / "churn_model.joblib")
    api.load_model.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/predict", json={"customers": [SAMPLE, SAFE]})
    assert r.status_code == 200
    risky, safe = r.json()["predictions"]
    assert risky["churn_probability"] > safe["churn_probability"]
    assert len(risky["top_drivers"]) == 3


def test_api_validation():
    client = TestClient(api.app)
    bad = {**SAMPLE, "contract": "Weekly"}
    assert client.post("/predict", json={"customers": [bad]}).status_code == 422
