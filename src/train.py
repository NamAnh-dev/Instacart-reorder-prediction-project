"""Step 3: train the models, choose the best one on the validation users, test it once.

  1. Baselines: tune each simple rule on the validation users.
  2. Models: fit on the train users, then on the validation users pick the probability
     threshold (recommend when probability >= threshold) with the best mean F1.
  3. The model with the best validation F1 is the final model.
  4. Report every system on the test users. Nothing is changed after this step.

Outputs:
  models/model.joblib            the final model
  models/metadata.json           model name, threshold, feature list, metrics (read by the API)
  reports/metrics.json           all results, threshold curves, training times
  reports/model_comparison.md    comparison table
  reports/feature_importance.csv feature importance of every model
  data/processed/predictions_test.parquet  test predictions, for the error analysis notebook

Run:  python -m src.train
      python -m src.train --sample 0.1 --models logistic_regression lightgbm   (quick trial)
"""

import argparse
import json
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from src.baselines import choose_baselines
from src.config import FEATURES, MODEL_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, THRESHOLDS, TOP_K
from src.evaluate import best_threshold, evaluate
from src.models import MODEL_NAMES, feature_importance, fit_model, make_model


def load_data(sample: float = 1.0, seed: int = SEED) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    if sample < 1.0:
        users = df["user_id"].drop_duplicates().sample(frac=sample, random_state=seed)
        df = df[df["user_id"].isin(users)]
    return {split: part.reset_index(drop=True) for split, part in df.groupby("split")}


def run_baselines(valid: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    rows = []
    for name, rule in choose_baselines(valid).items():
        rows.append(
            {
                "name": name,
                "type": "baseline",
                "setting": rule["setting"],
                "threshold": rule["threshold"],
                "valid": evaluate(valid, rule["score"](valid), rule["threshold"], TOP_K),
                "test": evaluate(test, rule["score"](test), rule["threshold"], TOP_K, with_ci=True),
            }
        )
        print(f"  baseline {name:<20} valid F1 {rows[-1]['valid']['f1']:.4f}  ({rule['setting']})")
    return rows


def run_models(data: dict[str, pd.DataFrame], names: list[str]) -> dict[str, dict]:
    """Fit every model and tune its threshold on the validation users.
    Returns {name: {"row": result row, "model": fitted model, "curve": threshold curve,
                    "importance": feature importance, "p_test": test probabilities}}."""
    train, valid, test = data["train"], data["valid"], data["test"]
    results = {}
    for name in names:
        start = time.time()
        model = fit_model(name, make_model(name), train[FEATURES], train["label"], valid[FEATURES], valid["label"])
        fit_seconds = time.time() - start

        p_valid = model.predict_proba(valid[FEATURES])[:, 1]
        p_test = model.predict_proba(test[FEATURES])[:, 1]
        threshold, _, curve = best_threshold(valid, p_valid, THRESHOLDS)

        row = {
            "name": name,
            "type": "model",
            "setting": f"probability >= {threshold:.2f}",
            "threshold": threshold,
            "fit_seconds": round(fit_seconds, 1),
            "valid": evaluate(valid, p_valid, threshold, TOP_K),
            "test": evaluate(test, p_test, threshold, TOP_K, with_ci=True),
        }
        results[name] = {
            "row": row,
            "model": model,
            "curve": curve,
            "importance": feature_importance(model, FEATURES),
            "p_test": p_test,
        }
        print(
            f"  model    {name:<20} valid F1 {row['valid']['f1']:.4f}  threshold {threshold:.2f}  ({fit_seconds:.0f}s)"
        )
    return results


def comparison_table(rows: list[dict], best: str) -> str:
    lines = [
        f"| System | Setting (chosen on valid) | Valid F1 | Test F1 [95% CI] | Test precision@{TOP_K} | "
        f"Test recall@{TOP_K} | Test ROC AUC | Avg items |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        t = r["test"]
        name = f"**{r['name']}** (final)" if r["name"] == best else r["name"]
        auc = f"{t['roc_auc']:.3f}" if "roc_auc" in t else "-"
        f1 = f"{t['f1']:.3f} [{t['f1_ci_low']:.3f}, {t['f1_ci_high']:.3f}]"
        lines.append(
            f"| {name} | {r['setting']} | {r['valid']['f1']:.4f} | {f1} | "
            f"{t[f'precision@{TOP_K}']:.3f} | {t[f'recall@{TOP_K}']:.3f} | {auc} | {t['avg_items']:.1f} |"
        )
    return "\n".join(lines)


def main(sample: float = 1.0, names: list[str] | None = None) -> dict:
    names = names or MODEL_NAMES
    data = load_data(sample)
    for split in ["train", "valid", "test"]:
        d = data[split]
        print(
            f"{split:<5}: {d['user_id'].nunique():>7,} users, {len(d):>10,} rows, positive rate {d['label'].mean():.2%}"
        )

    print("Baselines:")
    rows = run_baselines(data["valid"], data["test"])
    print("Models:")
    models = run_models(data, names)
    model_rows = [m["row"] for m in models.values()]
    rows += model_rows

    # the final model is the one with the best F1 on the validation users
    best = max(model_rows, key=lambda r: r["valid"]["f1"])
    print(f"Final model: {best['name']} (valid F1 {best['valid']['f1']:.4f}, test F1 {best['test']['f1']:.4f})")

    # --- save the final model and its metadata (the API reads these two files) ---
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best["name"]]["model"], MODEL_DIR / "model.joblib")
    metadata = {
        "model_name": best["name"],
        "threshold": best["threshold"],
        "features": FEATURES,
        "top_k": TOP_K,
        "valid_metrics": best["valid"],
        "test_metrics": best["test"],
        "n_train_rows": len(data["train"]),
        "data_sample": sample,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- reports ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "final_model": best["name"],
        "top_k": TOP_K,
        "sample": sample,
        "results": rows,
        "threshold_curves": {name: m["curve"] for name, m in models.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    importances = pd.DataFrame({name: m["importance"] for name, m in models.items()})
    importances.rename_axis("feature").to_csv(REPORTS_DIR / "feature_importance.csv")
    (REPORTS_DIR / "model_comparison.md").write_text(
        "# Model comparison\n\n"
        f"Metrics on the test users ({data['test']['user_id'].nunique():,} users). "
        "Every setting was chosen on the validation users; the final model has the best validation F1.\n\n"
        + comparison_table(rows, best["name"])
        + "\n"
    )

    test = data["test"]
    predictions = test[["user_id", "product_id", "label"] + FEATURES].copy()
    for name, m in models.items():
        predictions[f"prob_{name}"] = m["p_test"].astype(np.float32)
    predictions.to_parquet(PROCESSED_DIR / "predictions_test.parquet", index=False)

    print(f"Saved model to {MODEL_DIR}, reports to {REPORTS_DIR}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=float, default=1.0, help="fraction of users to use (e.g. 0.1 for a quick run)")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    args = parser.parse_args()
    main(args.sample, args.models)
