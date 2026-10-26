import numpy as np

from src.recommender import Recommender


class FixedProbabilities:
    """A fake model that always returns the same probabilities (one per row)."""

    def __init__(self, probs):
        self.probs = np.asarray(probs)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probs, self.probs])


def test_threshold_is_compared_before_rounding(serving_dirs):
    rec = Recommender(*serving_dirs)  # threshold 0.2, user 5 has 8 products
    # 0.19996 would become 0.2 after rounding to 4 decimals, but it is below the threshold
    rec.model = FixedProbabilities([0.5, 0.19996, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    items = rec.recommend(5)["items"]
    assert [item["probability"] for item in items] == [0.5, 0.2]
