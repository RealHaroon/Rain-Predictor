from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


DATA_PATH = Path("data/processed/train_table.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")

HORIZON_HOURS=24

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Missing {DATA_PATH}. Run Step 3 First')
    
    df= pd.read_csv(DATA_PATH)
    df['time']=pd.to_datetime(df['time'])
    df=df.sort_values('time').reset_index(drop=True)

    y=df['will_rain_24h'].astype(int)

    drop_cols={'time','will_rain_24h','rain_next_24h_mm'}
    feature_cols=[c  for c  in df.columns if c not in drop_cols]
    X=df[feature_cols]

    test_start=df['time'].max() - pd.Timedelta(days=180)

    train_end=test_start - pd.Timedelta(hours=HORIZON_HOURS)

    train_mask=df['time'] <= train_end
    test_mask=df['time'] >= test_start

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    pipe = Pipeline(
        steps=[
            ('scaler',StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, class_weight='balanced'))
        ]
    )

    tscv=TimeSeriesSplit(n_splits=5)
    cv=cross_validate(
        pipe,
        X_train,
        y_train,
        cv=tscv,
        scoring={'roc_auc':'roc_auc', 'avg_precision':'average_precision','bal_acc':'balanced_accuracy'},
        n_jobs=-1,
        return_train_score=False
    )

    
    pipe.fit(X_train, y_train)

    
    proba=pipe.predict_proba(X_test)[:, 1]
    pred=(proba >= 0.5).astype(int)

    test_metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_avg_precision": float(average_precision_score(y_test, proba)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": feature_cols,
        "test_start": str(test_start),
        "train_end": str(train_end),
    }

    artifacts_dir=MODEL_PATH.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(
        {"cv": {k: [float(x) for x in v] for k, v in cv.items() if k.startswith("test_")},
         "test": test_metrics},
        indent=2
    ))

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()