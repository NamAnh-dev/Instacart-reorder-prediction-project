import pandas as pd

from src.baselines import choose_baselines, purchase_rate, repeat_last_order
from tests.conftest import make_feature_table


def test_repeat_last_order():
    df = pd.DataFrame({"user_id": [1, 1], "up_in_last_order": [1.0, 0.0]})
    assert repeat_last_order(df).tolist() == [1.0, 0.0]


def test_purchase_rate_is_the_order_rate():
    df = pd.DataFrame({"user_id": [1, 1], "up_order_rate": [0.5, 0.1]})
    assert purchase_rate(df).tolist() == [0.5, 0.1]


def test_choose_baselines_returns_two_rules():
    valid = make_feature_table(n_users=20)
    rules = choose_baselines(valid)
    assert set(rules) == {"repeat_last_order", "purchase_rate"}
    for rule in rules.values():
        assert len(rule["score"](valid)) == len(valid)
