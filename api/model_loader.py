from pathlib import Path
import json
import joblib

MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")

_model = None
_feature_names = None

def load_model_and_features():
    global _model, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _feature_names is None:
        m = json.loads(METRICS_PATH.read_text())
        _feature_names = m["test"]["features"]
    return _model, _feature_names
