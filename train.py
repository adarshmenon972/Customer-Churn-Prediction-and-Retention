"""Training pipeline: tuning -> calibration -> cost-based threshold -> explainability.

    python -m churnguard.train --n-samples 20000 --n-trials 30 --out artifacts
    python -m churnguard.train --csv my_data.csv --target Churn --out artifacts
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from .data import TARGET, VALUE, generate_synthetic, load_csv  # noqa: E402
from .features import add_features  # noqa: E402
from .model import ChurnModel, shap_positive_class  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Business assumptions for the retention campaign (edit to match your business)
CAMPAIGN = {"cost_per_contact": 15.0, "save_rate": 0.35}


def make_prep(X: pd.DataFrame, linear: bool) -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num_t = (Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
             if linear else "passthrough")
    return ColumnTransformer(
        [("num", num_t, num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False,
    )


def score_report(y, p):
    return {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "log_loss": round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4),
    }


def campaign_profit(y, p, value, thr):
    """Net profit of contacting everyone with p >= thr."""
    target = p >= thr
    saved = (value[target & (y == 1)] * CAMPAIGN["save_rate"]).sum()
    cost = target.sum() * CAMPAIGN["cost_per_contact"]
    return float(saved - cost), int(target.sum())


def run(n_samples=20000, n_trials=30, out_dir="artifacts", csv=None, target=TARGET,
        value_col=None, seed=42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv, target, value_col) if csv else generate_synthetic(n_samples, seed)
    if "customer_id" in df:
        df = df.drop(columns="customer_id")
    y = df[TARGET].to_numpy()
    value = df[VALUE].to_numpy()
    X = add_features(df.drop(columns=[TARGET, VALUE]))
    print(f"Rows: {len(X):,} | Features: {X.shape[1]} | Churn rate: {y.mean():.1%}")

    # 60 / 20 / 20 stratified split: train (fit), val (calibrate + threshold), test (report)
    idx = np.arange(len(X))
    i_tmp, i_te = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    i_tr, i_va = train_test_split(i_tmp, test_size=0.25, stratify=y[i_tmp], random_state=seed)
    Xtr, Xva, Xte = X.iloc[i_tr], X.iloc[i_va], X.iloc[i_te]
    ytr, yva, yte = y[i_tr], y[i_va], y[i_te]
    vva, vte = value[i_va], value[i_te]

    # 1) Baseline: logistic regression
    lr = Pipeline([("prep", make_prep(Xtr, linear=True)),
                   ("clf", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]

    # 2) LightGBM tuned with Optuna (stratified CV, optimising PR-AUC)
    prep = make_prep(Xtr, linear=False).fit(Xtr)
    names = prep.get_feature_names_out().tolist()
    to_df = lambda d: pd.DataFrame(prep.transform(d), columns=names)  # noqa: E731
    Ttr, Tva, Tte = to_df(Xtr), to_df(Xva), to_df(Xte)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 700),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        clf = LGBMClassifier(**params, subsample_freq=1, random_state=seed, verbose=-1, n_jobs=-1)
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        return cross_val_score(clf, Ttr, ytr, cv=cv, scoring="average_precision").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    print(f"Best CV PR-AUC: {study.best_value:.4f} after {n_trials} trials")
    booster = LGBMClassifier(**study.best_params, subsample_freq=1, random_state=seed,
                             verbose=-1, n_jobs=-1).fit(Ttr, ytr)

    # 3) Isotonic calibration on the validation split
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(booster.predict_proba(Tva)[:, 1], yva)
    p_raw = booster.predict_proba(Tte)[:, 1]
    p_cal = np.clip(calibrator.predict(p_raw), 0, 1)

    # 4) Cost-based decision threshold (maximise campaign profit on validation)
    p_va = np.clip(calibrator.predict(booster.predict_proba(Tva)[:, 1]), 0, 1)
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    profits = [campaign_profit(yva, p_va, vva, t)[0] for t in grid]
    threshold = float(grid[int(np.argmax(profits))])

    prof_model, n_targeted = campaign_profit(yte, p_cal, vte, threshold)
    prof_all, _ = campaign_profit(yte, np.ones_like(p_cal), vte, 0.5)
    top_decile = p_cal >= np.quantile(p_cal, 0.9)
    business = {
        "assumptions": CAMPAIGN,
        "chosen_threshold": threshold,
        "customers_targeted_test": n_targeted,
        "test_profit_model": round(prof_model, 2),
        "test_profit_target_everyone": round(prof_all, 2),
        "test_profit_target_nobody": 0.0,
        "top_decile_lift": round(float(yte[top_decile].mean() / yte.mean()), 2),
        "top_decile_churners_captured": round(float(yte[top_decile].sum() / yte.sum()), 3),
    }

    metrics = {
        "logistic_regression": score_report(yte, p_lr),
        "lightgbm_raw": score_report(yte, p_raw),
        "lightgbm_calibrated": score_report(yte, p_cal),
        "best_params": study.best_params,
        "business": business,
    }

    # 5) Explainability with SHAP (global)
    import shap
    explainer = shap.TreeExplainer(booster)
    sample = Tte.sample(min(2000, len(Tte)), random_state=seed)
    sv = shap_positive_class(explainer, sample)
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    metrics["top_shap_features"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}

    # 6) Plots
    plt.figure(figsize=(5, 4.5))
    for label, p in [("LogReg", p_lr), ("LightGBM raw", p_raw), ("LightGBM calibrated", p_cal)]:
        frac, mean = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        plt.plot(mean, frac, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed churn rate")
    plt.title("Calibration"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "calibration.png", dpi=130); plt.close()

    plt.figure(figsize=(5, 4.5))
    for label, p in [("LogReg", p_lr), ("LightGBM", p_cal)]:
        pr, rc, _ = precision_recall_curve(yte, p)
        plt.plot(rc, pr, label=f"{label} (AP={average_precision_score(yte, p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=130); plt.close()

    plt.figure(figsize=(5.5, 4.5))
    test_profits = [campaign_profit(yte, p_cal, vte, t)[0] for t in grid]
    plt.plot(grid, test_profits)
    plt.axvline(threshold, color="r", ls="--", label=f"chosen = {threshold:.2f}")
    plt.xlabel("Contact customers with churn probability >="); plt.ylabel("Campaign net profit")
    plt.title("Retention campaign profit vs threshold"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "profit_curve.png", dpi=130); plt.close()

    shap.summary_plot(sv, sample, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(out / "shap_summary.png", dpi=130, bbox_inches="tight"); plt.close()

    # 7) Persist
    model = ChurnModel(prep, booster, calibrator, threshold, names, X.columns.tolist())
    joblib.dump(model, out / "churn_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== Test-set results ===")
    print(pd.DataFrame({k: metrics[k] for k in ["logistic_regression", "lightgbm_raw", "lightgbm_calibrated"]}).T)
    print("\n=== Business impact (test set) ===")
    print(json.dumps(business, indent=2))
    print("\nTop drivers (mean |SHAP|):")
    print(imp.head(8).round(3).to_string())
    print(f"\nArtifacts written to {out}/")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--csv", help="Your own churn CSV instead of synthetic data")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--value-col", help="Column with annual customer value/margin")
    a = ap.parse_args()
    run(a.n_samples, a.n_trials, a.out, a.csv, a.target, a.value_col)


if __name__ == "__main__":
    main()
