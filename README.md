# Rain Predictor (Karachi) — ML + FastAPI + Streamlit

A purely ML-based rain prediction project that trains a classifier on historical hourly weather for Karachi, wraps the trained model in a FastAPI inference API, and provides a simple Streamlit UI to query predictions.

> Current version focuses on Karachi. Multi-city generalization is planned.

***

## Features

- End-to-end ML pipeline:
  - Pull historical hourly data (Open‑Meteo)
  - Build supervised dataset (labels + lag/rolling features)
  - Train and evaluate a baseline model
  - Save model artifact (`joblib`) + metrics (`metrics.json`)
- FastAPI backend:
  - `POST /predict/by-city` (recommended): City → fetch recent history → build features → predict
  - `POST /predict` (debug): Accepts a full feature vector (advanced usage)
- Streamlit UI:
  - Minimal UI to predict “rain in next 24 hours” for a city

***

## Project structure

```text
rain-predictor/
  api/
    main.py
    model_loader.py
    openmeteo_client.py
    feature_builder.py
    schemas.py

  src/
    data/
      fetch_openmeteo.py
      build_dataset.py
    models/
      train.py

  ui/
    streamlit_app.py

  data/
    raw/
      karachi_hourly.csv
    processed/
      train_table.csv

  artifacts/
    model.joblib
    metrics.json

  notebooks/
    01_exploration.ipynb
    03_training_evaluation.ipynb

  requirements.txt
  README.md
```

***

## Setup

### 1) Install dependencies

Create a virtual environment (recommended) or install globally.

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, a minimal set is:

```bash
pip install pandas numpy scikit-learn joblib requests fastapi uvicorn streamlit
```

### 2) Run FastAPI

From the project root:

```bash
uvicorn api.main:app --reload
```

API docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 3) Run Streamlit

In a second terminal:

```bash
streamlit run ui/streamlit_app.py
```

***

## ML pipeline (training)

### Step 1 — Fetch historical data

This downloads hourly weather history and saves it to `data/raw/karachi_hourly.csv`.

```bash
python src/data/fetch_openmeteo.py
```

### Step 2 — Build supervised dataset

Creates labels like “will it rain in the next 24 hours” and generates lag/rolling features.

```bash
python src/data/build_dataset.py
```

Outputs:
- `data/processed/train_table.csv`

### Step 3 — Train model

Trains a baseline classifier, evaluates it with time-aware splitting, and saves artifacts.

```bash
python src/models/train.py
```

Outputs:
- `artifacts/model.joblib`
- `artifacts/metrics.json`

***

## API usage

### Recommended: predict by city

This is the correct production-style endpoint because it builds lag/rolling features automatically from recent weather history.

**Request**
```http
POST /predict/by-city
Content-Type: application/json

{
  "city": "Karachi",
  "country_code": "PK"
}
```

**Response**
```json
{
  "probability": 0.021,
  "will_rain_24h": false
}
```

### Debug endpoint: predict from features

`POST /predict` is intended for advanced use only. The model was trained on many engineered features (lags/rolling), so sending only a few “current weather” inputs will produce unreliable results unless you provide the full feature set.

***

## Notes / Known limitations

- **Karachi-only training**: The model is trained on Karachi historical data, so accuracy outside Karachi may degrade. Multi-city training is a planned improvement.
- **Archive data delay**: Historical (archive) sources can lag behind current dates. The API may use a safe time window or recent-history strategy depending on implementation.
- **Baseline model**: Logistic Regression is used initially for speed and interpretability. More advanced models (GBMs, calibration, threshold tuning) can be added later.

***

## Roadmap (future improvements)

- Train on multiple cities (Pakistan-wide) + add lat/lon/elevation as features
- Add Forecast-based “recent hours” fetch for live predictions (better than using delayed historical archive)
- Add proper probability calibration + threshold selection
- Add monitoring: data drift checks, retrain scripts, experiment tracking
- Better UI: show feature snapshot, recent rain trend, and confidence explanation

***

## License

Choose a license (MIT is common for portfolio projects). Add a `LICENSE` file if you want it public/open source.

***

If you want, paste your current `requirements.txt` (or let me infer it from your imports) and I’ll format it cleanly for the repo, plus add “Quickstart” commands exactly matching your filenames.