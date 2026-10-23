"""Request and response formats of the API (validated by Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    user_id: int = Field(..., gt=0, examples=[3])
    k: int = Field(10, ge=1, le=50, description="maximum number of products to return")


class Product(BaseModel):
    product_id: int
    product_name: str
    aisle: str
    department: str


class RecommendedProduct(Product):
    probability: float


class PredictResponse(BaseModel):
    user_id: int
    model_name: str
    threshold: float
    n_candidates: int = Field(description="number of products the user bought before")
    items: list[RecommendedProduct]


class PastProduct(Product):
    times_bought: int
    orders_since_last: int


class HealthResponse(BaseModel):
    status: str
    model_name: str
    threshold: float
    n_users: int
