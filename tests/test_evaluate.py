import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    avg_items,
    best_threshold,
    bootstrap_ci,
    count_per_user,
    evaluate,
    order_f1,
    precision_recall_at_k,
)


@pytest.fixture
def df():
    # user 1: truth {a, b}; user 2: truth {} (reorders nothing); user 3: truth {e}
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3],
            "label": [1, 1, 0, 0, 0, 1, 0],
            "score": [0.9, 0.2, 0.6, 0.1, 0.3, 0.8, 0.7],
        }
    )


def test_count_per_user(df):
    counts = count_per_user(df, df["score"] >= 0.5)
    # user 1: true {a, b}, selected {a, c}, hit {a}; user 2: nothing; user 3: true {e}, selected {e, f}
    assert counts.loc[1].tolist() == [2, 2, 1]
    assert counts.loc[2].tolist() == [0, 0, 0]
    assert counts.loc[3].tolist() == [1, 2, 1]


def test_order_f1_cases(df):
    pred = np.array([1, 0, 1, 0, 0, 1, 1], dtype=bool)
    f1 = order_f1(df, pred)
    assert f1[1] == pytest.approx(2 * 1 / (2 + 2))  # TP=1, 2 predicted, 2 true
    assert f1[2] == 1.0  # nothing predicted, nothing true
    assert f1[3] == pytest.approx(2 * 1 / (2 + 1))


def test_order_f1_predicting_something_when_truth_is_empty_gives_zero(df):
    pred = np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)
    assert order_f1(df, pred)[2] == 0.0


def test_avg_items(df):
    assert avg_items(df, df["score"] >= 0.5) == pytest.approx((2 + 0 + 2) / 3)


def test_precision_recall_at_k(df):
    result = precision_recall_at_k(df, df["score"], k=1)
    # top-1 per user: user1 a (hit), user2 the 0.3 item (miss), user3 e (hit)
    assert result["precision@1"] == pytest.approx(2 / 3)
    # recall over users with a truth: user1 1/2, user3 1/1
    assert result["recall@1"] == pytest.approx((0.5 + 1) / 2)


def test_best_threshold_picks_the_highest_f1(df):
    threshold, f1, curve = best_threshold(df, df["score"], [0.0, 0.5, 0.75, 0.95])
    assert curve[threshold] == f1 == max(curve.values())
    assert threshold == 0.75  # user1 {a}, user2 {} (correct), user3 {e} -> (2/3 + 1 + 1) / 3


def test_evaluate_returns_all_metrics(df):
    result = evaluate(df, df["score"], threshold=0.5, k=2, with_ci=True)
    assert {"f1", "avg_items", "precision@2", "recall@2", "roc_auc", "pr_auc"} <= set(result)
    assert result["f1_ci_low"] <= result["f1"] <= result["f1_ci_high"]


def test_bootstrap_ci_contains_the_mean_and_shrinks_with_more_users():
    rng = np.random.default_rng(0)
    small, large = rng.random(100), rng.random(10_000)
    low, high = bootstrap_ci(small, n_boot=200)
    assert low < small.mean() < high
    low_l, high_l = bootstrap_ci(large, n_boot=200)
    assert high_l - low_l < high - low
