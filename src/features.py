"""Step 2: feature engineering.

One row = one (user, product) pair, for every product the user bought in a prior order.
These pairs are the CANDIDATES: the model gives each one a probability of being reordered.

Features are built ONLY from the purchase history, with the same code for training and
for serving, so the API sees exactly the same features as the model did in training.
We do not use anything about the next order itself (day, hour, days since the last order)
because the "Buy it again" shelf is built before we know when the user will order.

Outputs:
  data/processed/features.parquet  train / valid / test users, with the label
  data/serving/features.parquet    Kaggle "test" users (next order unknown) for the API
  data/serving/products.parquet    product names, used by the API

Run:  python -m src.features
"""

import pandas as pd

from src.config import FEATURES, PROCESSED_DIR, SERVING_DIR
from src.data import load_products


def user_features(history: pd.DataFrame) -> pd.DataFrame:
    by_order = history.groupby(["user_id", "order_number"]).agg(
        basket_size=("product_id", "size"), days_since_first=("days_since_first", "first")
    )
    users = by_order.groupby("user_id").agg(
        u_n_orders=("basket_size", "size"),
        u_avg_basket_size=("basket_size", "mean"),
        u_last_days=("days_since_first", "max"),  # helper column, not a model feature
    )
    # average gap = total days between the first and the last order / number of gaps
    users["u_avg_days_between"] = users["u_last_days"] / (users["u_n_orders"] - 1).clip(lower=1)

    not_first = history[history["order_number"] > 1]
    users["u_reorder_ratio"] = not_first.groupby("user_id")["reordered"].mean()
    users["u_n_products"] = history.groupby("user_id")["product_id"].nunique()
    return users


def product_features(history: pd.DataFrame) -> pd.DataFrame:
    products = history.groupby("product_id").agg(
        p_n_orders=("order_id", "size"),
        p_reorder_rate=("reordered", "mean"),
        p_avg_cart_position=("add_to_cart_order", "mean"),
    )
    products["p_n_users"] = history.drop_duplicates(["user_id", "product_id"]).groupby("product_id").size()
    return products


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """All features for every (user, product) pair in the history."""
    up = (
        history.groupby(["user_id", "product_id"], sort=False)
        .agg(
            up_n_orders=("order_id", "size"),
            up_first_order=("order_number", "min"),
            up_last_order=("order_number", "max"),
            up_last_days=("days_since_first", "max"),
            up_avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    up = up.merge(user_features(history), left_on="user_id", right_index=True, how="left")
    up = up.merge(product_features(history), left_on="product_id", right_index=True, how="left")

    up["up_order_rate"] = up["up_n_orders"] / up["u_n_orders"]
    up["up_orders_since_last"] = up["u_n_orders"] - up["up_last_order"]
    up["up_days_since_last"] = up["u_last_days"] - up["up_last_days"]
    up["up_rate_since_first"] = up["up_n_orders"] / (up["u_n_orders"] - up["up_first_order"] + 1)
    up["up_in_last_order"] = (up["up_last_order"] == up["u_n_orders"]).astype("int8")

    out = up[["user_id", "product_id"] + FEATURES].copy()
    out[FEATURES] = out[FEATURES].fillna(0).astype("float32")
    return out.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def add_labels(features: pd.DataFrame, labels: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Add `label` (1 = reordered in the last order) and the user's `split` group."""
    df = features.merge(labels, on=["user_id", "product_id"], how="left")
    df["label"] = df["label"].fillna(0).astype("int8")
    return df.merge(users[["user_id", "split"]], on="user_id", how="left")


def run() -> None:
    history = pd.read_parquet(PROCESSED_DIR / "history.parquet")
    users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
    labels = pd.read_parquet(PROCESSED_DIR / "labels.parquet")

    print(f"Building features from {len(history):,} history rows ...")
    features = add_labels(build_features(history), labels, users)
    del history

    # Every reordered product was bought before, so it must be a candidate.
    n_found = features["label"].sum()
    print(f"labels found among candidates: {n_found:,} of {len(labels):,} ({n_found / len(labels):.2%})")

    modeling = features[features["split"] != "serving"].reset_index(drop=True)
    serving = features.loc[features["split"] == "serving", ["user_id", "product_id"] + FEATURES]
    modeling.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    serving.reset_index(drop=True).to_parquet(SERVING_DIR / "features.parquet", index=False)
    load_products().to_parquet(SERVING_DIR / "products.parquet", index=False)

    print(f"modeling rows: {len(modeling):,} (positive rate {modeling['label'].mean():.2%})")
    print(modeling.groupby("split")["user_id"].nunique().rename("users").to_string())
    print(f"serving rows : {len(serving):,} for {serving['user_id'].nunique():,} users")
    print(f"features ({len(FEATURES)}): {', '.join(FEATURES)}")


if __name__ == "__main__":
    run()
