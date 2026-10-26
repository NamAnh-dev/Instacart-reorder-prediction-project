import pytest

from src.config import FEATURES
from src.features import add_labels, build_features
from src.preprocess import add_days_since_first_order, build_history, build_labels, build_users
from tests.conftest import A, B, C


@pytest.fixture
def features(raw_tables):
    orders = add_days_since_first_order(raw_tables["orders"])
    history = build_history(orders, raw_tables["prior"])
    return build_features(history).set_index(["user_id", "product_id"])


def test_one_row_per_user_product_pair(features):
    # user 1: A B C, user 2: B C, user 3: A D
    assert len(features) == 7
    assert list(features.columns) == FEATURES
    assert not features.isna().any().any()


def test_user_features(features):
    u1 = features.loc[(1, A)]
    assert u1["u_n_orders"] == 3
    assert u1["u_avg_basket_size"] == 2
    assert u1["u_avg_days_between"] == 7.5  # 15 days / 2 gaps
    assert u1["u_reorder_ratio"] == 0.75  # orders 2-3: A(1) C(0) A(1) B(1)
    assert u1["u_n_products"] == 3


def test_product_features(features):
    a = features.loc[(1, A)]
    assert a["p_n_orders"] == 5  # orders 1, 2, 3, 9, 10
    assert a["p_n_users"] == 2
    assert a["p_reorder_rate"] == pytest.approx(3 / 5)


def test_user_product_features(features):
    a, c = features.loc[(1, A)], features.loc[(1, C)]
    assert (a["up_n_orders"], a["up_order_rate"], a["up_orders_since_last"], a["up_in_last_order"]) == (3, 1, 0, 1)
    # C was bought only in order 2 (day 5); the last order is order 3 (day 15)
    assert c["up_n_orders"] == 1
    assert c["up_orders_since_last"] == 1
    assert c["up_days_since_last"] == 10
    assert c["up_rate_since_first"] == 0.5  # 1 purchase / 2 orders since the first one
    assert c["up_in_last_order"] == 0
    assert features.loc[(1, B), "up_avg_cart_position"] == 2


def test_add_labels(raw_tables):
    orders = add_days_since_first_order(raw_tables["orders"])
    features = build_features(build_history(orders, raw_tables["prior"]))
    df = add_labels(features, build_labels(orders, raw_tables["train"]), build_users(orders))
    labelled = df.set_index(["user_id", "product_id"])["label"]
    assert labelled[(1, A)] == 1
    assert labelled.sum() == 1
    assert (df.loc[df["user_id"] == 3, "split"] == "serving").all()
