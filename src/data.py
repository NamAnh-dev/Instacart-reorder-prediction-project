"""Load the raw Instacart CSV files.

order_products__prior.csv has about 32 million rows, so we use small integer
types to keep memory low on a normal laptop.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

FILES = {
    "orders": "orders.csv",
    "prior": "order_products__prior.csv",
    "train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

_ORDER_PRODUCT_DTYPES = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}

DTYPES = {
    "orders": {
        "order_id": "int32",
        "user_id": "int32",
        "eval_set": "category",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
        "days_since_prior_order": "float32",  # empty for the first order of a user
    },
    "prior": _ORDER_PRODUCT_DTYPES,
    "train": _ORDER_PRODUCT_DTYPES,
    "products": {"product_id": "int32", "product_name": "string", "aisle_id": "int16", "department_id": "int16"},
    "aisles": {"aisle_id": "int16", "aisle": "string"},
    "departments": {"department_id": "int16", "department": "string"},
}


def load_table(name: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    path = Path(raw_dir) / FILES[name]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle 'Instacart Market Basket Analysis' data "
            "and put the CSV files in data/raw/."
        )
    return pd.read_csv(path, usecols=list(DTYPES[name]), dtype=DTYPES[name])


def load_all(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    return {name: load_table(name, raw_dir) for name in FILES}


def load_products(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Products with aisle and department names."""
    products = load_table("products", raw_dir)
    products = products.merge(load_table("aisles", raw_dir), on="aisle_id", how="left")
    products = products.merge(load_table("departments", raw_dir), on="department_id", how="left")
    return products[["product_id", "product_name", "aisle", "department"]]
