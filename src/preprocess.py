"""Step 1: clean the raw data and prepare 3 tables for feature engineering.

How the Kaggle data is organised:
  * every user has several "prior" orders (their purchase history),
  * and one last order, which is either in the "train" set (we know what was bought)
    or in the "test" set (Kaggle hides what was bought).

Outputs in data/processed/:
  history.parquet  one row per product in a prior order, with the user and the order timing
  users.parquet    one row per user: eval_set of the last order and the split group
  labels.parquet   products REORDERED in the last order of "train" users (the target)

Run:  python -m src.preprocess
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, SEED, TRAIN_FRACTION, VALID_FRACTION
from src.data import load_all


def add_days_since_first_order(orders: pd.DataFrame) -> pd.DataFrame:
    """Add `days_since_first`: days between the user's first order and this order.

    The data has no real dates, only `days_since_prior_order`. It is empty for the
    first order of each user, so we fill it with 0 and take a running sum.
    """
    orders = orders.sort_values(["user_id", "order_number"]).copy()
    gap = orders["days_since_prior_order"].fillna(0)
    orders["days_since_first"] = gap.groupby(orders["user_id"]).cumsum().astype("float32")
    return orders


def build_history(orders: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    """Prior order items joined with user_id, order_number and days_since_first."""
    order_info = orders.loc[orders["eval_set"] == "prior", ["order_id", "user_id", "order_number", "days_since_first"]]
    history = prior.merge(order_info, on="order_id", how="inner")
    cols = ["user_id", "order_id", "order_number", "days_since_first", "product_id", "add_to_cart_order", "reordered"]
    return history[cols].sort_values(["user_id", "order_number", "add_to_cart_order"]).reset_index(drop=True)


def split_users(user_ids, seed: int = SEED) -> pd.Series:
    """Randomly assign users to "train" / "valid" / "test" (70 / 15 / 15).

    We split by USER (not by row), so all products of one user are in the same group
    and the model is always evaluated on users it has never seen.
    """
    user_ids = np.sort(np.asarray(user_ids))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(user_ids)
    n_train = int(len(shuffled) * TRAIN_FRACTION)
    n_valid = int(len(shuffled) * VALID_FRACTION)
    groups = np.array(["train"] * n_train + ["valid"] * n_valid + ["test"] * (len(shuffled) - n_train - n_valid))
    return pd.Series(groups, index=shuffled, name="split").sort_index()


def build_users(orders: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One row per user. Kaggle "test" users get split = "serving": we do not know their
    next order, so they are the users the API makes recommendations for."""
    last = orders.loc[orders["eval_set"] != "prior", ["user_id", "eval_set"]].astype({"eval_set": "string"})
    users = last.set_index("user_id").sort_index()
    users["split"] = "serving"
    train_users = users.index[users["eval_set"] == "train"]
    users.loc[train_users, "split"] = split_users(train_users, seed)
    return users.reset_index()


def build_labels(orders: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """(user_id, product_id) pairs that were REORDERED in the user's last order.

    We only recommend products the user bought before, so a new product in the last
    order (reordered = 0) can never be predicted and is not a target.
    """
    order_user = orders.loc[orders["eval_set"] == "train", ["order_id", "user_id"]]
    labels = train.loc[train["reordered"] == 1].merge(order_user, on="order_id")
    labels = labels[["user_id", "product_id"]].drop_duplicates()
    labels["label"] = np.int8(1)
    return labels.reset_index(drop=True)


def clean(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Simple data quality checks. Problems are printed and fixed when it is safe."""
    orders, prior, train = tables["orders"], tables["prior"], tables["train"]

    for name, df in [("prior", prior), ("train", train)]:
        n_dup = df.duplicated(["order_id", "product_id"]).sum()
        if n_dup:
            print(f"  {name}: dropping {n_dup:,} duplicated (order_id, product_id) rows")
            df = df.drop_duplicates(["order_id", "product_id"])
        unknown = ~df["order_id"].isin(orders["order_id"])
        if unknown.any():
            print(f"  {name}: dropping {unknown.sum():,} rows with an unknown order_id")
            df = df.loc[~unknown]
        tables[name] = df

    first_missing = orders["days_since_prior_order"].isna() & (orders["order_number"] != 1)
    if first_missing.any():
        print(f"  orders: {first_missing.sum():,} non-first orders have no days_since_prior_order")

    n_last = orders.loc[orders["eval_set"] != "prior"].groupby("user_id").size()
    if (n_last != 1).any():
        raise ValueError("every user must have exactly one train/test order")
    return tables


def run() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading raw files from {RAW_DIR} ...")
    tables = clean(load_all())
    orders = add_days_since_first_order(tables["orders"])

    users = build_users(orders)
    history = build_history(orders, tables["prior"])
    labels = build_labels(orders, tables["train"])

    users.to_parquet(PROCESSED_DIR / "users.parquet", index=False)
    labels.to_parquet(PROCESSED_DIR / "labels.parquet", index=False)
    history.to_parquet(PROCESSED_DIR / "history.parquet", index=False)

    print(f"history: {len(history):,} rows, {history['user_id'].nunique():,} users")
    print(f"labels : {len(labels):,} reordered (user, product) pairs")
    print("users per split:")
    print(users["split"].value_counts().to_string())


if __name__ == "__main__":
    run()
