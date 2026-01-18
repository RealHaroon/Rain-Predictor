from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/karachi_hourly.csv")
OUT_PATH = Path("data/processed/train_table.csv")

HORIZON_HOURS = 24
RAIN_THRESHOLD_MM = 0.1  # label = 1 if next-24h rain >= this


BASE_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "rain",
]


def future_sum(series: pd.Series, horizon: int) -> pd.Series:
    # Sum of next horizon values: s(t+1)+...+s(t+horizon)
    out = None
    for k in range(1, horizon + 1):
        shifted = series.shift(-k)
        out = shifted if out is None else (out + shifted)
    return out


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> None:
    for l in lags:
        df[f"{col}_lag_{l}h"] = df[col].shift(l)


def add_rolling_features(df: pd.DataFrame, col: str, windows: list[int]) -> None:
    # Rolling windows are backward-looking; shift(1) makes them strictly “past-only”
    # pandas rolling provides rolling window calculations [web:110]
    for w in windows:
        df[f"{col}_rollsum_{w}h"] = df[col].rolling(window=w, min_periods=w).sum().shift(1)
        df[f"{col}_rollmean_{w}h"] = df[col].rolling(window=w, min_periods=w).mean().shift(1)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run Step 2 first.")

    df = pd.read_csv(RAW_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Keep only columns we need (ignore extra metadata columns safely)
    keep = ["time"] + [c for c in BASE_FEATURES if c in df.columns]
    df = df[keep].copy()

    # ----- Target label: will_rain_24h -----
    # Open-Meteo: hourly rain is "preceding hour sum" [page:1]
    df["rain_next_24h_mm"] = future_sum(df["rain"], HORIZON_HOURS)
    df["will_rain_24h"] = (df["rain_next_24h_mm"] >= RAIN_THRESHOLD_MM).astype("int64")

    # ----- Feature engineering (past-only) -----
    # Time features
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month
    df["dayofyear"] = df["time"].dt.dayofyear

    # Lags
    add_lag_features(df, "rain", [1, 3, 6, 12, 24])
    add_lag_features(df, "precipitation", [1, 3, 6, 12, 24])
    add_lag_features(df, "relative_humidity_2m", [1, 3, 6])
    add_lag_features(df, "pressure_msl", [1, 3, 6])

    # Rolling stats
    add_rolling_features(df, "rain", [3, 6, 12, 24])
    add_rolling_features(df, "precipitation", [6, 12, 24])

    # Drop rows that have NaNs created by shifts/rolling/future label
    df = df.dropna().reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # quick sanity prints
    pos_rate = df["will_rain_24h"].mean()
    print(f"Saved: {OUT_PATH} rows={len(df)} cols={len(df.columns)} pos_rate={pos_rate:.4f}")


if __name__ == "__main__":
    main()
