from __future__ import annotations
from pydantic import BaseModel

from typing import Dict
from pydantic import BaseModel

class RainFeatures(BaseModel):
    features:Dict[str, float]

class RainPrediction(BaseModel):
    probability:float
    will_rain_24h:bool

class CityRequest(BaseModel):
    city: str
    country_code: str = "PK"