import numpy as np
import pandas as pd

from src.preprocess import add_days_since_first_order, build_history, build_labels, build_users, clean, split_users
from tests.conftest import A, B


def test_days_since_first_is_a_running_sum(raw_tables):
    orders = add_days_since_first_order(raw_tables["orders"])
    user1 = orders[orders["user_id"] == 1]["days_since_first"].tolist()
    assert user1 == [0, 5, 15, 22]


def test_history_only_has_prior_orders(raw_tables):
    orders = add_days_since_first_order(raw_tables["orders"])
    history = build_history(orders, raw_tables["prior"])
    assert len(history) == len(raw_tables["prior"])
    assert set(history["order_id"]) == {1, 2, 3, 5, 6, 7, 9, 10, 11}
    row = history[(history["order_id"] == 3) & (history["product_id"] == B)].iloc[0]
    assert (row["user_id"], row["order_number"], row["days_since_first"]) == (1, 3, 15)


def test_labels_are_reordered_products_of_train_orders(raw_tables):
    labels = build_labels(raw_tables["orders"], raw_tables["train"])
    # user 1 reordered A; D was new; user 2 only bought a new product
    assert labels[["user_id", "product_id"]].values.tolist() == [[1, A]]
    assert (labels["label"] == 1).all()


def test_split_is_reproducible_and_covers_every_user():
    users = np.arange(1, 1001)
    first, second = split_users(users, seed=1), split_users(users, seed=1)
    assert first.equals(second)
    assert sorted(first.index) == list(users)
    counts = first.value_counts()
    assert (counts["train"], counts["valid"], counts["test"]) == (700, 150, 150)


def test_kaggle_test_users_are_used_for_serving(raw_tables):
    users = build_users(raw_tables["orders"]).set_index("user_id")
    assert users.loc[3, "split"] == "serving"
    assert set(users.loc[[1, 2], "split"]) <= {"train", "valid", "test"}


def test_clean_drops_duplicated_rows(raw_tables):
    prior = raw_tables["prior"]
    raw_tables["prior"] = pd.concat([prior, prior.iloc[[0]]], ignore_index=True)
    tables = clean(raw_tables)
    assert len(tables["prior"]) == len(prior)
