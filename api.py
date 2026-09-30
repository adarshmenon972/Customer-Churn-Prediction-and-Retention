"""FastAPI service.  Run:  uvicorn churnguard.api:app --port 8000

POST /predict  -> churn probability, risk tier, recommended action, SHAP drivers
GET  /health
"""
import os
from functools import lru_cache
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ChurnGuard API", version="1.0.0")


class Customer(BaseModel):
    tenure_months: float = Field(ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["Fiber", "DSL", "None"]
    monthly_charge: float = Field(ge=0)
    num_addons: int = Field(ge=0)
    support_tickets_90d: int = Field(ge=0)
    late_payments_12m: int = Field(ge=0)
    avg_weekly_usage_hrs: float = Field(ge=0)
    usage_change_pct: float
    price_increase_pct: float = Field(ge=0)
    autopay: Literal["Yes", "No"]
    payment_method: str


class PredictRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=1000)
    explain: bool = True


ACTIONS = {
    "High": "Proactive retention call + tailored offer (e.g. contract upgrade discount)",
    "Medium": "Automated engagement: usage tips, loyalty perk, satisfaction survey",
    "Low": "No action - monitor",
}


@lru_cache(maxsize=2)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)


def get_model():
    path = os.environ.get("CHURN_MODEL_PATH", "artifacts/churn_model.joblib")
    try:
        return load_model(path)
    except FileNotFoundError:
        raise HTTPException(503, f"Model not found at {path}. Run: python -m churnguard.train")


@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok", "decision_threshold": model.threshold}


@app.post("/predict")
def predict(req: PredictRequest):
    model = get_model()
    df = pd.DataFrame([c.model_dump() for c in req.customers])
    probs = model.predict_proba(df)
    drivers = model.explain(df) if req.explain else [None] * len(df)
    results = []
    for p, d in zip(probs, drivers):
        tier = model.risk_tier(float(p))
        item = {"churn_probability": round(float(p), 4), "risk_tier": tier,
                "recommended_action": ACTIONS[tier]}
        if d is not None:
            item["top_drivers"] = d
        results.append(item)
    return {"predictions": results}
