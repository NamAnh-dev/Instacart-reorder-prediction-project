"""Project settings in one place: paths, random seed, feature list, model settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folders. MODEL_DIR and SERVING_DIR can be changed with environment variables
# (the Docker container uses this to read the mounted folders).
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SERVING_DIR = Path(os.getenv("SERVING_DIR", PROJECT_ROOT / "data" / "serving"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42

# Users whose last order is in the Kaggle "train" set are split into 3 groups.
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15  # the rest (0.15) is the test group

# Number of items shown on the "Buy it again" shelf, used for precision@k / recall@k.
TOP_K = 10

# Thresholds tried on the validation users.
THRESHOLDS = [round(0.05 + 0.01 * i, 2) for i in range(46)]  # 0.05, 0.06, ..., 0.50

USER_FEATURES = [
    "u_n_orders",  # number of past orders
    "u_avg_basket_size",  # average number of items per order
    "u_avg_days_between",  # average days between two orders
    "u_reorder_ratio",  # share of items that were reorders (orders 2+)
    "u_n_products",  # number of different products bought
]
PRODUCT_FEATURES = [
    "p_n_orders",  # how many times the product was bought (all users)
    "p_n_users",  # how many different users bought it
    "p_reorder_rate",  # share of purchases that were reorders
    "p_avg_cart_position",  # average position in the cart
]
USER_PRODUCT_FEATURES = [
    "up_n_orders",  # how many times this user bought this product
    "up_order_rate",  # up_n_orders / u_n_orders
    "up_orders_since_last",  # number of orders since the user last bought it
    "up_days_since_last",  # days since the user last bought it
    "up_rate_since_first",  # up_n_orders / orders since the first purchase
    "up_avg_cart_position",  # average cart position for this user
    "up_in_last_order",  # 1 if it was in the user's last order
]
FEATURES = USER_FEATURES + PRODUCT_FEATURES + USER_PRODUCT_FEATURES
