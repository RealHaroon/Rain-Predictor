from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import RainPrediction, CityRequest, RainFeatures
from .model_loader import load_model_and_features
from .openmeteo_client import geocode_city, fetch_hourly_archive
from .feature_builder import build_features_from_hourly


app = FastAPI(title="Rain Predictor API")

# Allow Streamlit to call the API (CORS) [web:212]
origins = [
    "http://localhost",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# Keep this only for debugging/power-users (requires providing ALL features correctly).
@app.post("/predict", response_model=RainPrediction)
def predict(payload: RainFeatures):
    model, feature_names = load_model_and_features()

    # Build a row with the exact training feature set.
    row = {name: 0.0 for name in feature_names}
    row.update(payload.features)

    X = pd.DataFrame([row], columns=feature_names)

    try:
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e!r}")

    return RainPrediction(probability=proba, will_rain_24h=proba >= 0.5)


# Recommended endpoint: UI sends city; API fetches history and builds lag/rolling features.
@app.post("/predict/by-city", response_model=RainPrediction)
def predict_by_city(req: CityRequest):
    model, feature_names = load_model_and_features()

    try:
        lat, lon = geocode_city(req.city, req.country_code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Geocoding failed: {e!r}")

    # Fetch enough history to compute 24h lags/rollings robustly.
   # Historical Weather API can have a delay for the newest days. [web:302]
    end = date.today() - timedelta(days=7)   # safe buffer
    start = end - timedelta(days=7)          # fetch 1 week ending 7 days ago


    try:
        hourly = fetch_hourly_archive(lat, lon, start.isoformat(), end.isoformat())
        feat_df = build_features_from_hourly(hourly)  # should return exactly 1 row
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature build failed: {e!r}")

    # Align columns to training features.
    missing = [c for c in feature_names if c not in feat_df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Feature mismatch. Missing columns: {missing[:10]}... total_missing={len(missing)}",
        )

    X = feat_df[feature_names].copy()

    if X.isna().any().any():
        raise HTTPException(
            status_code=400,
            detail="Feature mismatch: NaNs present after selecting training feature columns.",
        )

    proba = float(model.predict_proba(X)[:, 1][0])
    return RainPrediction(probability=proba, will_rain_24h=proba >= 0.5)
