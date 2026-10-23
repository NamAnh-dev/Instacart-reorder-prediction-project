import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(serving_dirs):
    model_dir, serving_dir = serving_dirs
    with TestClient(create_app(model_dir, serving_dir)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_name"] == "logistic_regression"
    assert body["n_users"] == 60


def test_predict_returns_sorted_products_above_threshold(client):
    response = client.post("/predict", json={"user_id": 5, "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 5
    assert body["n_candidates"] == 8
    probs = [item["probability"] for item in body["items"]]
    assert len(probs) <= 3
    assert probs == sorted(probs, reverse=True)
    assert all(p >= body["threshold"] for p in probs)
    if body["items"]:
        assert body["items"][0]["product_name"].startswith("product ")


def test_unknown_user_gives_404(client):
    assert client.post("/predict", json={"user_id": 999}).status_code == 404
    assert client.get("/users/999/history").status_code == 404


def test_invalid_request_gives_422(client):
    assert client.post("/predict", json={"user_id": 5, "k": 0}).status_code == 422
    assert client.post("/predict", json={"user_id": "abc"}).status_code == 422


def test_history_and_sample_users(client):
    history = client.get("/users/5/history", params={"k": 4}).json()
    assert len(history) == 4
    counts = [item["times_bought"] for item in history]
    assert counts == sorted(counts, reverse=True)
    users = client.get("/users/sample", params={"n": 3}).json()
    assert len(set(users)) == 3 and all(1 <= u <= 60 for u in users)
