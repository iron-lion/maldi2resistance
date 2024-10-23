from typing import List

from pydantic import BaseModel


class MSData(BaseModel):
    xValues: List[float]
    yValues: List[float]


class Resistance(BaseModel):
    antibioticName: str
    antibioticResistance: float


class Resistances(BaseModel):
    resistances: List[Resistance]
