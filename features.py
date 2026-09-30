"""Feature engineering shared by training and serving (single source of truth)."""
import numpy as np
import pandas as pd


def _has(df: pd.DataFrame, *cols) -> bool:
    return all(c in df.columns for c in cols)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if _has(df, "tenure_months"):
        df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    if _has(df, "monthly_charge", "num_addons"):
        df["charge_per_addon"] = df["monthly_charge"] / (df["num_addons"] + 1)
    if _has(df, "support_tickets_90d", "tenure_months"):
        df["tickets_per_month"] = df["support_tickets_90d"] / np.minimum(df["tenure_months"], 3).clip(lower=1)
    if _has(df, "usage_change_pct", "price_increase_pct"):
        df["price_shock_on_disengaged"] = ((df["price_increase_pct"] > 0) & (df["usage_change_pct"] < -20)).astype(int)
    if _has(df, "late_payments_12m", "autopay"):
        df["late_and_manual_pay"] = ((df["late_payments_12m"] >= 2) & (df["autopay"] == "No")).astype(int)
    if _has(df, "monthly_charge", "tenure_months"):
        df["lifetime_spend"] = df["monthly_charge"] * df["tenure_months"]
    return df
