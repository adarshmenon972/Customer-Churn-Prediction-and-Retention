"""Serving wrapper: preprocessing + LightGBM + isotonic calibration + SHAP explanations."""
import numpy as np
import pandas as pd

from .features import add_features


def shap_positive_class(explainer, X):
    """Return SHAP values for the churn class as a 2-D array, across shap versions."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


class ChurnModel:
    def __init__(self, prep, booster, calibrator, threshold, feature_names, input_cols):
        self.prep = prep
        self.booster = booster
        self.calibrator = calibrator
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)
        self.input_cols = list(input_cols)  # columns present after add_features at fit time
        self._explainer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_explainer"] = None  # rebuilt lazily, keeps the artifact small
        return state

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = add_features(df)
        X = self.prep.transform(feats[self.input_cols])
        return pd.DataFrame(X, columns=self.feature_names)

    def raw_proba(self, df):
        return self.booster.predict_proba(self.transform(df))[:, 1]

    def predict_proba(self, df) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.raw_proba(df)), 0.0, 1.0)

    def risk_tier(self, p: float) -> str:
        if p >= self.threshold:
            return "High"
        return "Medium" if p >= self.threshold / 2 else "Low"

    def explain(self, df, top_k: int = 3):
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.booster)
        X = self.transform(df)
        sv = shap_positive_class(self._explainer, X)
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(sv[i]))[:top_k]
            out.append([
                {
                    "feature": self.feature_names[j],
                    "value": float(X.iloc[i, j]),
                    "impact": round(float(sv[i, j]), 4),
                    "effect": "raises churn risk" if sv[i, j] > 0 else "lowers churn risk",
                }
                for j in order
            ])
        return out
