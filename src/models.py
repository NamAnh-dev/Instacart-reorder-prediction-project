"""The machine learning models we compare.

  * Logistic regression: simple linear model, easy to explain (needs scaled features).
  * Random forest: many decision trees trained on random samples, averaged.
  * LightGBM: gradient boosting, trees are added one by one to fix the errors of the previous trees.
    Usually the strongest on tabular data.

LightGBM uses early stopping: training stops when the log loss on the validation users has not
improved for 50 rounds, so we do not have to guess the number of trees.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import SEED

MODEL_NAMES = ["logistic_regression", "random_forest", "lightgbm"]


def make_model(name: str, seed: int = SEED):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "random_forest":
        # max_samples: each tree sees 10% of the rows, which keeps training fast
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=50, max_samples=0.1, n_jobs=-1, random_state=seed
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=2000,  # upper limit, early stopping picks the real number
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            importance_type="gain",
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"unknown model {name!r}, choose from {MODEL_NAMES}")


def fit_model(name: str, model, X_train, y_train, X_valid, y_valid):
    if name == "lightgbm":
        model.fit(
            X_train,
            y_train,
            eval_X=X_valid,  # the validation users, watched by early stopping
            eval_y=y_valid,
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
    else:
        model.fit(X_train, y_train)
    return model


def feature_importance(model, features: list[str]) -> pd.Series:
    """Importance of each feature, scaled to sum to 1.
    Linear model: absolute coefficient (features are standardized, so they are comparable).
    Tree models: total gain of the splits that use the feature."""
    if isinstance(model, Pipeline):  # logistic regression = Pipeline(scaler, model); the last step has coef_
        values = np.abs(model[-1].coef_[0])
    else:
        values = np.asarray(model.feature_importances_, dtype=float)
    if values.sum() > 0:
        values = values / values.sum()
    return pd.Series(values, index=features).sort_values(ascending=False)
