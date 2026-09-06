"""
FastAPI prediction service for the ecommerce recommendation model.
"""

import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("ecommerce_api")
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/recommendation_model.pkl")

# Module-level model reference, populated at startup by the lifespan handler.
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Loaded model from %s", MODEL_PATH)
        except Exception:
            logger.exception("Failed to load model from %s", MODEL_PATH)
            model = None
    else:
        logger.warning("Model file not found at %s; /predict will return 503 until it is available.", MODEL_PATH)
        model = None
    yield
    # No teardown needed.


app = FastAPI(
    title="Ecommerce Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    base_price: float = Field(..., ge=0, json_schema_extra={"example": 49.99})
    discount_percent: float = Field(..., ge=0, le=100, json_schema_extra={"example": 15.0})
    rating: float = Field(..., ge=0, le=5, json_schema_extra={"example": 4.2})
    num_reviews: int = Field(..., ge=0, json_schema_extra={"example": 128})
    stock_quantity: int = Field(..., ge=0, json_schema_extra={"example": 50})


class PredictResponse(BaseModel):
    recommendation_score: float
    model_version: str = "1.0.0"


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run scripts/train_dummy_model.py and ensure ./models is mounted.",
        )

    features = np.array(
        [[
            request.base_price,
            request.discount_percent,
            request.rating,
            request.num_reviews,
            request.stock_quantity,
        ]]
    )

    prediction = model.predict(features)
    score = float(np.clip(prediction[0], 0, 1))

    return PredictResponse(recommendation_score=score)
