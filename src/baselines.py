"""Simple rules without machine learning. A model is only useful if it beats them.

Each baseline gives a score to every (user, product) pair. We recommend the pairs with
score >= threshold, and the threshold is chosen on the validation users.
"""

import pandas as pd

from src.config import THRESHOLDS
from src.evaluate import best_threshold


def repeat_last_order(df: pd.DataFrame):
    """Recommend exactly the products of the user's last order."""
    return df["up_in_last_order"].to_numpy(dtype=float)


def purchase_rate(df: pd.DataFrame):
    """Score = share of the user's orders that contain the product."""
    return df["up_order_rate"].to_numpy(dtype=float)


def choose_baselines(valid: pd.DataFrame) -> dict[str, dict]:
    """Tune every baseline on the validation users.
    Returns {name: {"score": function(df) -> scores, "threshold": t, "setting": text}}."""
    rate_t, _, _ = best_threshold(valid, purchase_rate(valid), THRESHOLDS)
    return {
        "repeat_last_order": {"score": repeat_last_order, "threshold": 1.0, "setting": "products of the last order"},
        "purchase_rate": {"score": purchase_rate, "threshold": rate_t, "setting": f"order rate >= {rate_t:.2f}"},
    }
