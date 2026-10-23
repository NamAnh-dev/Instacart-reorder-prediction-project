"""Run the whole training pipeline: preprocess -> features -> train.

Run:  python -m src.pipeline
"""

from __future__ import annotations

import time

from src import features, preprocess, train


def run() -> None:
    for name, step in [("1/3 preprocess", preprocess.run), ("2/3 features", features.run), ("3/3 train", train.main)]:
        start = time.time()
        print(f"\n=== Step {name} ===")
        step()
        print(f"--- done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    run()
