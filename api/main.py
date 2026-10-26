"""FastAPI service: "Buy it again" recommendations for a user.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from api.schemas import HealthResponse, PastProduct, PredictRequest, PredictResponse
from src.config import MODEL_DIR, SERVING_DIR
from src.recommender import Recommender


def create_app(model_dir: Path | str = MODEL_DIR, serving_dir: Path | str = SERVING_DIR) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # load the model and the features once, when the server starts
        app.state.recommender = Recommender(model_dir, serving_dir)
        yield

    app = FastAPI(
        title="Instacart Buy-It-Again API",
        description="Predicts which products a user will reorder in their next order.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_recommender(request: Request) -> Recommender:
        return request.app.state.recommender

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        rec = get_recommender(request)
        return HealthResponse(
            status="ok", model_name=rec.model_name, threshold=rec.threshold, n_users=len(rec.user_ids)
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        try:
            return get_recommender(request).recommend(body.user_id, body.k)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"user {body.user_id} not found") from None

    @app.get("/users/sample", response_model=list[int])
    def sample_users(request: Request, n: int = Query(5, ge=1, le=50)):
        return get_recommender(request).sample_users(n)

    @app.get("/users/{user_id}/history", response_model=list[PastProduct])
    def user_history(user_id: int, request: Request, k: int = Query(10, ge=1, le=50)):
        try:
            return get_recommender(request).history(user_id, k)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"user {user_id} not found") from None

    return app


app = create_app()
