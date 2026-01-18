from __future__ import annotations
from pathlib import Path
import requests
import pandas as pd

GEOCODE_URL="https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL="https://archive-api.open-meteo.com/v1/archive"

CITY="Karachi"
COUNTRY_CODE="PK"

START_DATE="2020-01-01"
END_DATE="2025-12-31"

HOURLY_VARS=[
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "rain"
]

OUT_PATH=Path("data/raw/karachi_hourly.csv")

def geocode_city(name:str,country_code:str | None = None) -> dict :
    params={"name":name,"count":1, "language":"en"}
    if country_code:
        params["countryCode"]=country_code
    
    r=requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data=r.json()

    results=data.get("results") or []
    if not results:
        raise ValueError(f"No geocoding results for '{name}'")
    return results[0]

def fetch_archive_hourly(lat:float, lon:float, start_date:str, end_date:str)-> pd.DataFrame:
    params={
        "latitude":lat,
        "longitude":lon,
        "start_date":start_date,
        "end_date":end_date,
        "hourly":",".join(HOURLY_VARS),
        "timezone":"auto"
    }

    r=requests.get(ARCHIVE_URL, params=params, timeout=60)
    r.raise_for_status()
    data= r.json()

    hourly=data.get('hourly')
    if not hourly or "time" not in hourly:
        raise ValueError("Unexpected archive response: missing hourly/time")
    
    df=pd.DataFrame(hourly)
    df['time']=pd.to_datetime(df["time"])
    df=df.sort_values("time").reset_index(drop=True)

    # keep metadata columsn too (later will be used)
    df['latitude']=data.get('latitude')
    df['longitude']=data.get('longitude')
    df['timezone']=data.get('timezone')

    return df

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    loc=geocode_city(CITY, COUNTRY_CODE)
    lat, lon=float(loc['latitude']), float(loc['longitude'])

    df=fetch_archive_hourly(lat, lon, START_DATE, END_DATE)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH} rows={len(df)} cols={len(df.columns)}")

if __name__=="__main__":
    main()