"""Evaluation metrics.

Each user has ONE target order, so "per order" and "per user" mean the same thing here.

Main metric: mean F1 per order (like the Kaggle competition metric, but without its special "None" item)
  * truth      = products the user reordered (can be empty)
  * prediction = products we recommend (can be empty = we recommend nothing)
  * F1 = 2 * TP / (n_predicted + n_true)
  * if both are empty, F1 = 1 (recommending nothing was correct)
  * if the truth is empty but we recommend something, F1 = 0

Ranking metrics for a shelf of k items: precision@k and recall@k.
Row-level metrics: ROC AUC and PR AUC (how well the probabilities rank the pairs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import SEED


def order_f1(df: pd.DataFrame, pred) -> pd.Series:
    """F1 for every user. `df` has columns user_id and label; `pred` is True when recommended."""
    pred = np.asarray(pred, dtype=bool)
    label = df["label"].to_numpy(dtype=bool)
    counts = pd.DataFrame({"tp": label & pred, "n_pred": pred, "n_true": label}).groupby(df["user_id"].to_numpy()).sum()
    denom = counts["n_pred"] + counts["n_true"]
    f1 = np.where(denom == 0, 1.0, 2 * counts["tp"] / denom.where(denom > 0, 1))
    return pd.Series(f1, index=counts.index, name="f1")


def mean_f1(df: pd.DataFrame, pred) -> float:
    return float(order_f1(df, pred).mean())


def bootstrap_ci(values, n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    """95% interval of a mean: resample the values (one per user) with replacement and take percentiles."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def avg_items(df: pd.DataFrame, pred) -> float:
    """Average number of recommended products per user."""
    return float(pd.Series(np.asarray(pred, dtype=bool)).groupby(df["user_id"].to_numpy()).sum().mean())


def top_k_mask(df: pd.DataFrame, score, k: int) -> np.ndarray:
    """True for the k highest scores of each user."""
    rank = (
        pd.Series(np.asarray(score, dtype=float))
        .groupby(df["user_id"].to_numpy())
        .rank(method="first", ascending=False)
    )
    return (rank <= k).to_numpy()


def precision_recall_at_k(df: pd.DataFrame, score, k: int) -> dict:
    """precision@k: share of the shown items that were reordered (averaged over all users).
    A user with fewer than k past products is shown all of them.
    recall@k: share of the reordered items that were shown (users with at least one reorder)."""
    shown = top_k_mask(df, score, k)
    label = df["label"].to_numpy(dtype=bool)
    counts = (
        pd.DataFrame({"tp": label & shown, "n_shown": shown, "n_true": label}).groupby(df["user_id"].to_numpy()).sum()
    )
    has_true = counts["n_true"] > 0
    return {
        f"precision@{k}": float((counts["tp"] / counts["n_shown"]).mean()),
        f"recall@{k}": float((counts["tp"][has_true] / counts["n_true"][has_true]).mean()),
    }


def best_threshold(df: pd.DataFrame, score, thresholds) -> tuple[float, float, dict]:
    """Try every threshold (recommend when score >= t) and keep the one with the best mean F1.
    Returns (threshold, f1, {threshold: f1})."""
    score = np.asarray(score, dtype=float)
    curve = {float(t): mean_f1(df, score >= t) for t in thresholds}
    best = max(curve, key=curve.get)
    return best, curve[best], curve


def evaluate(df: pd.DataFrame, score, threshold: float, k: int, with_ci: bool = False) -> dict:
    """All metrics for one system on one group of users. with_ci=True adds a bootstrap interval for F1."""
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    label = df["label"].to_numpy()
    f1_per_user = order_f1(df, pred)
    result = {"f1": float(f1_per_user.mean()), "avg_items": avg_items(df, pred)}
    if with_ci:
        result["f1_ci_low"], result["f1_ci_high"] = bootstrap_ci(f1_per_user)
    result.update(precision_recall_at_k(df, score, k))
    if len(np.unique(score)) > 2:  # ROC/PR AUC only make sense for real probabilities
        result["roc_auc"] = float(roc_auc_score(label, score))
        result["pr_auc"] = float(average_precision_score(label, score))
    return result
