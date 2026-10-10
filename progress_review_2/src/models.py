"""Predeclared candidates; selection uses validation data only."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from .data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


class HistoryBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, feature="lag_1"):
        self.feature = feature

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return X[self.feature].to_numpy(dtype=float)


def make_candidates(cfg):
    settings = cfg["candidate_settings"]
    seed = cfg["seed"]
    def pipeline(model, attributes=True, scaled=False):
        columns = [("numeric", StandardScaler() if scaled else "passthrough", NUMERIC_FEATURES)]
        if attributes:
            columns.append(("attributes", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES))
        return Pipeline([("prepare", ColumnTransformer(columns, remainder="drop")), ("model", model)])
    return {
        "NaiveLastWeek": HistoryBaseline("lag_1"),
        "Mean4Weeks": HistoryBaseline("mean_4"),
        "Ridge": pipeline(Ridge(**settings["Ridge"]), scaled=True),
        "RandomForestHistory": pipeline(RandomForestRegressor(**settings["RandomForestHistory"], random_state=seed, n_jobs=2), attributes=False),
        "RandomForestAttributes": pipeline(RandomForestRegressor(**settings["RandomForestAttributes"], random_state=seed, n_jobs=2)),
        "HistGradientBoosting": pipeline(HistGradientBoostingRegressor(**settings["HistGradientBoosting"], random_state=seed)),
    }


def predict_nonnegative(model, frame):
    result = np.maximum(0.0, model.predict(frame))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model prediction.")
    return result
