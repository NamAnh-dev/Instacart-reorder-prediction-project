"""Load the trained model and the serving features, and make recommendations.

Used by the API. Features were computed in advance by `python -m src.features`
(batch features), and the model scores them when a request comes in.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, SERVING_DIR


class Recommender:
    def __init__(self, model_dir: Path | str = MODEL_DIR, serving_dir: Path | str = SERVING_DIR):
        model_dir, serving_dir = Path(model_dir), Path(serving_dir)
        self.model = joblib.load(model_dir / "model.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.model_name = self.metadata["model_name"]
        self.threshold = float(self.metadata["threshold"])
        self.features = self.metadata["features"]

        table = pd.read_parquet(serving_dir / "features.parquet")
        self.table = table.set_index("user_id").sort_index()  # sorted index -> fast lookup by user
        self.user_ids = self.table.index.unique().to_numpy()
        self.known_users = set(self.user_ids.tolist())
        self.products = pd.read_parquet(serving_dir / "products.parquet").set_index("product_id")

    def _user_rows(self, user_id: int) -> pd.DataFrame:
        """All (user, product) feature rows of one user. KeyError if the user is unknown."""
        if user_id not in self.known_users:
            raise KeyError(user_id)
        return self.table.loc[[user_id]]

    def _with_product_info(self, df: pd.DataFrame) -> list[dict]:
        df = df.join(self.products, on="product_id")
        df[["product_name", "aisle", "department"]] = df[["product_name", "aisle", "department"]].fillna("unknown")
        return df.to_dict(orient="records")

    def recommend(self, user_id: int, k: int = 10) -> dict:
        """Products with probability >= threshold, best first, at most k.
        The list can be empty: the model then thinks the user will not reorder anything."""
        rows = self._user_rows(user_id)
        prob = self.model.predict_proba(rows[self.features])[:, 1]
        scored = pd.DataFrame({"product_id": rows["product_id"].to_numpy(), "probability": prob})
        # compare the raw probability with the threshold, like in training; round only for display
        chosen = scored[scored["probability"] >= self.threshold].sort_values("probability", ascending=False).head(k)
        chosen["probability"] = chosen["probability"].round(4)
        return {
            "user_id": int(user_id),
            "model_name": self.model_name,
            "threshold": self.threshold,
            "n_candidates": len(scored),
            "items": self._with_product_info(chosen),
        }

    def history(self, user_id: int, k: int = 10) -> list[dict]:
        """The user's most frequently bought products."""
        rows = (
            self._user_rows(user_id)
            .sort_values(["up_n_orders", "up_orders_since_last"], ascending=[False, True])
            .head(k)
        )
        out = pd.DataFrame(
            {
                "product_id": rows["product_id"].to_numpy(),
                "times_bought": rows["up_n_orders"].astype(int).to_numpy(),
                "orders_since_last": rows["up_orders_since_last"].astype(int).to_numpy(),
            }
        )
        return self._with_product_info(out)

    def sample_users(self, n: int = 5, seed: int | None = None) -> list[int]:
        rng = np.random.default_rng(seed)
        return [int(u) for u in rng.choice(self.user_ids, size=min(n, len(self.user_ids)), replace=False)]
