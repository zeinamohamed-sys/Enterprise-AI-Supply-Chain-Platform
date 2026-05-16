from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from ml_engine import execute_pipeline

app = FastAPI(
    title="Enterprise AI Supply Chain & Analytics Platform API",
    description="Microservice API supplying dynamic demand forecasts, collaborative product recommendations, and local rule-based customer text sentiment execution.",
    version="1.0.0"
)

# Global pipeline orchestration layout
try:
    COMPONENTS = execute_pipeline()
except Exception as e:
    # Fallback to prevent startup crashes if data hasn't fully synced yet
    COMPONENTS = None

class ForecastRequest(BaseModel):
    lag_1: float = Field(..., description="Previous day unit sales count")
    lag_2: float = Field(..., description="Two days prior unit sales count")
    roll_7_mean: float = Field(..., description="7-day dynamic rolling average inventory demand")
    month: int = Field(..., ge=1, le=12, description="Calendar month sequence")
    day_of_week: int = Field(..., ge=0, le=6, description="Weekday identifier index (0=Monday, 6=Sunday)")

class ForecastResponse(BaseModel):
    predicted_demand_units: float

class RecommendationResponse(BaseModel):
    product_id: str
    recommended_product_ids: list[str]

class SentimentRequest(BaseModel):
    review_text: str

class SentimentResponse(BaseModel):
    review_text: str
    computed_sentiment: str

@app.get("/")
def read_root():
    return {
        "status": "ONLINE",
        "framework": "FastAPI Microservice Layer",
        "pipeline_state": "ACTIVE" if COMPONENTS is not exists else "PENDING_INITIALIZATION"
    }

@app.post("/api/v1/forecast", response_model=ForecastResponse)
def predict_demand(payload: ForecastRequest):
    if not COMPONENTS:
        raise HTTPException(status_code=503, detail="Machine Learning inference core is currently offline or loading.")
    
    try:
        input_data = pd.DataFrame([{
            "Lag_1": payload.lag_1,
            "Lag_2": payload.lag_2,
            "Roll_7_Mean": payload.roll_7_mean,
            "Month": payload.month,
            "DayOfWeek": payload.day_of_week
        }])
        prediction = COMPONENTS["forecaster"].predict(input_data)[0]
        return ForecastResponse(predicted_demand_units=float(prediction))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(err)}")

@app.get("/api/v1/recommend/{product_id}", response_model=RecommendationResponse)
def get_recommendations(product_id: str):
    if not COMPONENTS:
        raise HTTPException(status_code=503, detail="Recommendation engine remains uninitialized.")
    
    recs = COMPONENTS["recommender"].recommend(product_id)
    return RecommendationResponse(
        product_id=product_id,
        recommended_product_ids=recs
    )

@app.post("/api/v1/sentiment", response_model=SentimentResponse)
def evaluate_sentiment(payload: SentimentRequest):
    if not COMPONENTS:
        raise HTTPException(status_code=503, detail="Semantic rule engine deployment has failed or is loading.")
    
    sentiment = COMPONENTS["sentiment_analyzer"].predict_sentiment(payload.review_text)
    return SentimentResponse(
        review_text=payload.review_text,
        computed_sentiment=sentiment
  )
  
