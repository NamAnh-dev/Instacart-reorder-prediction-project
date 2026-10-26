"""A tiny hand-made Instacart dataset, small enough to check every number by hand.

user 1 (Kaggle train user)          user 2 (Kaggle train user)      user 3 (Kaggle test user)
  order 1: A, B                       order 5: B                      order 9 : A
  order 2: A, C        (+5 days)      order 6: B, C   (+3 days)       order 10: A   (+2 days)
  order 3: A, B        (+10 days)     order 7: C      (+4 days)       order 11: D   (+2 days)
  order 4 (train): A (reorder), D     order 8 (train): E (new item)   order 12 (test): unknown

Product ids: A=10, B=20, C=30, D=40, E=50.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES

A, B, C, D, E = 10, 20, 30, 40, 50


@pytest.fixture
def raw_tables() -> dict[str, pd.DataFrame]:
    orders = pd.DataFrame(
        [
            # order_id, user_id, eval_set, order_number, days_since_prior_order
            (1, 1, "prior", 1, np.nan),
            (2, 1, "prior", 2, 5.0),
            (3, 1, "prior", 3, 10.0),
            (4, 1, "train", 4, 7.0),
            (5, 2, "prior", 1, np.nan),
            (6, 2, "prior", 2, 3.0),
            (7, 2, "prior", 3, 4.0),
            (8, 2, "train", 4, 6.0),
            (9, 3, "prior", 1, np.nan),
            (10, 3, "prior", 2, 2.0),
            (11, 3, "prior", 3, 2.0),
            (12, 3, "test", 4, 9.0),
        ],
        columns=["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"],
    )
    orders["order_dow"], orders["order_hour_of_day"] = 1, 10
    orders = orders.astype(
        {
            "order_id": "int32",
            "user_id": "int32",
            "eval_set": "category",
            "order_number": "int16",
            "days_since_prior_order": "float32",
        }
    )
    # order_id, product_id, add_to_cart_order, reordered
    prior = pd.DataFrame(
        [
            (1, A, 1, 0),
            (1, B, 2, 0),
            (2, A, 1, 1),
            (2, C, 2, 0),
            (3, A, 1, 1),
            (3, B, 2, 1),
            (5, B, 1, 0),
            (6, B, 1, 1),
            (6, C, 2, 0),
            (7, C, 1, 1),
            (9, A, 1, 0),
            (10, A, 1, 1),
            (11, D, 1, 0),
        ],
        columns=["order_id", "product_id", "add_to_cart_order", "reordered"],
    )
    train = pd.DataFrame([(4, A, 1, 1), (4, D, 2, 0), (8, E, 1, 0)], columns=prior.columns)
    products = pd.DataFrame(
        {
            "product_id": [A, B, C, D, E],
            "product_name": ["Apples", "Bananas", "Carrots", "Donuts", "Eggs"],
            "aisle_id": [1, 1, 2, 3, 4],
            "department_id": [1, 1, 1, 2, 3],
        }
    )
    aisles = pd.DataFrame({"aisle_id": [1, 2, 3, 4], "aisle": ["fruit", "vegetables", "bakery", "eggs"]})
    departments = pd.DataFrame({"department_id": [1, 2, 3], "department": ["produce", "bakery", "dairy eggs"]})
    return {
        "orders": orders,
        "prior": prior,
        "train": train,
        "products": products,
        "aisles": aisles,
        "departments": departments,
    }


def make_feature_table(n_users: int = 60, n_products: int = 8, seed: int = 0) -> pd.DataFrame:
    """Random feature table where the label depends on up_order_rate (so a model can learn it)."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(1, n_users + 1):
        for product in range(1, n_products + 1):
            rows.append((user, product))
    df = pd.DataFrame(rows, columns=["user_id", "product_id"])
    for col in FEATURES:
        df[col] = rng.random(len(df)).astype("float32")
    df["up_n_orders"] = rng.integers(1, 10, len(df)).astype("float32")
    df["up_orders_since_last"] = rng.integers(0, 5, len(df)).astype("float32")
    df["label"] = (rng.random(len(df)) < df["up_order_rate"] * 0.6).astype("int8")
    return df


@pytest.fixture
def serving_dirs(tmp_path):
    """A small trained model + serving features, saved like src.train and src.features do."""
    table = make_feature_table()
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(table[FEATURES], table["label"])

    model_dir, serving_dir = tmp_path / "models", tmp_path / "serving"
    model_dir.mkdir()
    serving_dir.mkdir()
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps({"model_name": "logistic_regression", "threshold": 0.2, "features": FEATURES, "top_k": 10})
    )
    table.drop(columns="label").to_parquet(serving_dir / "features.parquet", index=False)
    pd.DataFrame(
        {
            "product_id": range(1, 9),
            "product_name": [f"product {i}" for i in range(1, 9)],
            "aisle": "aisle",
            "department": "department",
        }
    ).to_parquet(serving_dir / "products.parquet", index=False)
    return model_dir, serving_dir
