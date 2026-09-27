import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from app.schemas import EnergyPredictionRequest, EnergyPredictionResponse

app = FastAPI(title="Energy Load Forecasting API", version="1.0")

# Load model artifact bundle on startup
MODEL_PATH = Path("app/models/energy_model.pkl")
if MODEL_PATH.exists():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    building_mapping = artifact["building_mapping"]
else:
    model = None
    building_mapping = {}

@app.get("/")
def read_root():
    return {"status": "online", "message": "Energy Forecaster API is running!"}

@app.post("/predict", response_model=EnergyPredictionResponse)
def predict_energy(payload: EnergyPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Trained model not found. Run train.py first.")
    
    # Look up the building name in our training mapping (default to 0 if unknown)
    building_code = building_mapping.get(payload.building_id, 0)
    
    # Construct input dataframe matching training features exactly
    input_data = pd.DataFrame([{
        "building_encoded": building_code,
        "air_temperature": payload.air_temperature,
        "dew_temperature": payload.dew_temperature,
        "wind_speed": payload.wind_speed,
        "cloud_coverage": payload.cloud_coverage,
        "hour_of_day": payload.hour_of_day,
        "day_of_week": payload.day_of_week
    }])
    
    # Run model prediction
    predicted_load = float(model.predict(input_data)[0])
    
    return {
        "building_id": payload.building_id,
        "predicted_load_kwh": predicted_load,
        "model_version": "v1.0-random-forest"
    }