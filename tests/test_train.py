import json

import joblib
import numpy as np
import pytest

from src import train
from src.config import FEATURES
from src.models import MODEL_NAMES, feature_importance, fit_model, make_model
from tests.conftest import make_feature_table


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_probabilities(name):
    df = make_feature_table(n_users=300)
    model = fit_model(name, make_model(name), df[FEATURES], df["label"], df[FEATURES], df["label"])
    prob = model.predict_proba(df[FEATURES])[:, 1]
    assert prob.shape == (len(df),)
    assert np.all((prob >= 0) & (prob <= 1))
    importance = feature_importance(model, FEATURES)
    assert importance.sum() == pytest.approx(1.0)


def test_main_saves_the_best_model_and_reports(tmp_path, monkeypatch):
    df = make_feature_table(n_users=90)
    df["split"] = np.select([df["user_id"] <= 60, df["user_id"] <= 75], ["train", "valid"], "test")
    processed = tmp_path / "processed"
    processed.mkdir()
    df.to_parquet(processed / "features.parquet", index=False)
    for name, path in [
        ("PROCESSED_DIR", processed),
        ("MODEL_DIR", tmp_path / "models"),
        ("REPORTS_DIR", tmp_path / "reports"),
    ]:
        monkeypatch.setattr(train, name, path)

    report = train.main(names=["logistic_regression", "lightgbm"])

    metadata = json.loads((tmp_path / "models" / "metadata.json").read_text())
    assert metadata["model_name"] == report["final_model"]
    assert metadata["features"] == FEATURES
    model_rows = [r for r in report["results"] if r["type"] == "model"]
    assert report["final_model"] == max(model_rows, key=lambda r: r["valid"]["f1"])["name"]
    model = joblib.load(tmp_path / "models" / "model.joblib")
    assert model.predict_proba(df[FEATURES]).shape == (len(df), 2)
    assert (tmp_path / "reports" / "model_comparison.md").is_file()
    assert (processed / "predictions_test.parquet").is_file()
