from pydantic import BaseModel, Field

class EnergyPredictionRequest(BaseModel):
    building_id: str = Field(..., description="Full source building identifier from dataset", json_schema_extra={"example": "Bear_education_Bob"})
    air_temperature: float = Field(..., description="Ambient air temperature in Celsius", json_schema_extra={"example": 22.5})
    dew_temperature: float = Field(..., description="Dew point temperature in Celsius", json_schema_extra={"example": 10.0})
    wind_speed: float = Field(..., description="Wind speed in m/s", json_schema_extra={"example": 3.5})
    cloud_coverage: float = Field(..., description="Cloud coverage metric", json_schema_extra={"example": 0.0})
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)", json_schema_extra={"example": 14})
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)", json_schema_extra={"example": 2})

class EnergyPredictionResponse(BaseModel):
    building_id: str
    predicted_load_kwh: float
    model_version: str