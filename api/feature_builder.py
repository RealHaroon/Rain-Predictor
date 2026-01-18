from __future__ import annotations
import pandas as pd

def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> None:
    for l in lags:
        df[f"{col}_lag_{l}h"] = df[col].shift(l)

def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> None:
    for w in windows:
        df[f"{col}_rollsum_{w}h"] = df[col].rolling(window=w, min_periods=w).sum().shift(1)
        df[f"{col}_rollmean_{w}h"] = df[col].rolling(window=w, min_periods=w).mean().shift(1)

def build_features_from_hourly(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # time features
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["dayofyear"] = df["time"].dt.dayofyear

    # lags
    add_lag_features(df, "rain", [1, 3, 6, 12, 24])
    add_lag_features(df, "precipitation", [1, 3, 6, 12, 24])
    add_lag_features(df, "relative_humidity_2m", [1, 3, 6])
    add_lag_features(df, "pressure_msl", [1, 3, 6])

    # rolling stats
    add_rolling_features(df, "rain", [3, 6, 12, 24])
    add_rolling_features(df, "precipitation", [6, 12, 24])

    # last row = “current hour feature vector”
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Not enough history to build lag/rolling features (need >= 24h).")

    return df.tail(1)   # one-row DataFrame
