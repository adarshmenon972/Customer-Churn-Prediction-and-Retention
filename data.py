"""Data generation / loading.

The synthetic generator produces a realistic subscription-business dataset with
non-linear effects and interactions, so gradient boosting has something real to
learn that a plain logistic regression cannot fully capture.
"""
import numpy as np
import pandas as pd

TARGET = "churn"
VALUE = "annual_margin"  # yearly profit we lose if the customer leaves


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate_synthetic(n: int = 20000, seed: int = 42, churn_rate: float = 0.26) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.35, 0.20])
    autopay = rng.choice(["Yes", "No"], n, p=[0.45, 0.55])
    payment = np.where(
        autopay == "Yes",
        rng.choice(["Card", "Bank debit"], n),
        rng.choice(["Cash", "Cheque", "UPI"], n),
    )

    mtm = contract == "Month-to-month"
    tenure = np.where(mtm, rng.gamma(2.0, 8.0, n), rng.gamma(4.0, 10.0, n)).clip(1, 72).round()
    num_addons = rng.integers(0, 6, n)
    base = pd.Series(internet).map({"Fiber": 80, "DSL": 55, "None": 25}).to_numpy()
    monthly = (base + 6 * num_addons + rng.normal(0, 7, n)).clip(15, 200).round(2)

    tickets = rng.poisson(np.where(internet == "Fiber", 1.8, 1.0), n)
    late = rng.poisson(np.where(autopay == "Yes", 0.2, 1.1), n)
    usage = rng.gamma(2.5, 5.0, n).round(1)
    usage_change = rng.normal(0, 25, n).clip(-90, 60).round(1)
    price_inc = np.where(rng.random(n) < 0.3, rng.uniform(3, 25, n), 0.0).round(1)

    z = (
        -0.045 * tenure
        + 1.25 * mtm
        - 0.55 * (contract == "Two year")
        + 0.45 * (internet == "Fiber")
        + 0.018 * (monthly - 60)
        + 0.30 * tickets
        + 0.35 * late
        - 0.045 * usage
        - 0.018 * usage_change
        + 0.045 * price_inc
        - 0.45 * (autopay == "Yes")
        - 0.12 * num_addons
        + 0.9 * ((tenure <= 6) & mtm)               # early-life churn
        + 0.6 * ((internet == "Fiber") & mtm)       # interaction
        + 0.035 * price_inc * (usage_change < -20)  # price shock on disengaging users
        # --- non-linear effects that a linear model cannot capture ---
        + 1.1 * ((contract == "One year") & (tenure >= 11) & (tenure <= 14))   # renewal cliff
        + 1.3 * ((contract == "Two year") & (tenure >= 23) & (tenure <= 26))
        + 1.2 * (tickets >= 4)                                                  # ticket threshold
        - 1.0 * ((usage > 12) & (usage < 28))                                   # engagement sweet spot
        + 1.0 * ((usage_change < -40) & (tenure > 12))                          # sudden disengagement of loyal users
        + 0.9 * ((price_inc > 15) & (internet == "Fiber"))                      # price-sensitive segment
        + 0.8 * ((late >= 3) & (autopay == "No") & (monthly > 90))              # payment stress on big bills
        - 0.9 * ((num_addons >= 4) & (tenure > 24))                             # sticky bundled customers
    )
    z = z + rng.normal(0, 0.6, n)

    lo, hi = -10.0, 10.0  # bisection: shift intercept to hit the target churn rate
    for _ in range(40):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > churn_rate:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(z + (lo + hi) / 2)
    churn = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "tenure_months": tenure,
        "contract": contract,
        "internet_service": internet,
        "monthly_charge": monthly,
        "num_addons": num_addons,
        "support_tickets_90d": tickets,
        "late_payments_12m": late,
        "avg_weekly_usage_hrs": usage,
        "usage_change_pct": usage_change,
        "price_increase_pct": price_inc,
        "autopay": autopay,
        "payment_method": payment,
        VALUE: (monthly * 12 * 0.40).round(2),
        TARGET: churn,
    })


def load_csv(path: str, target: str = TARGET, value_col: str | None = None,
             id_cols: tuple = ("customer_id", "customerID")) -> pd.DataFrame:
    """Load your own churn CSV. Target may be Yes/No or 1/0. If no value column is
    given, annual value is approximated from `monthly_charge` (or set to 100)."""
    df = pd.read_csv(path)
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.lower().isin(["yes", "1", "true", "churn", "churned"])
    df[TARGET] = y.astype(int)
    if target != TARGET:
        df = df.drop(columns=[target])
    if value_col and value_col in df:
        df[VALUE] = df[value_col]
    elif "monthly_charge" in df:
        df[VALUE] = df["monthly_charge"] * 12 * 0.40
    else:
        df[VALUE] = 100.0
    return df.drop(columns=[c for c in id_cols if c in df.columns])
