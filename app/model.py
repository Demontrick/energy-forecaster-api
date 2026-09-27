import joblib
import pandas as pd
from pathlib import Path
from app.schemas import EnergyPredictionRequest

# Define path to the saved model artifact
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "energy_model.pkl"

# Global variables for model and building mapping
model = None
building_mapping = {}

# Load the trained model artifact bundle into memory when the API starts up
if MODEL_PATH.exists():
    try:
        artifact = joblib.load(MODEL_PATH)
        # Check if loaded artifact is our new dictionary bundle or legacy raw model
        if isinstance(artifact, dict):
            model = artifact.get("model")
            building_mapping = artifact.get("building_mapping", {})
        else:
            # Fallback support for older single-model files
            model = artifact
            building_mapping = {}
            
        print("✅ Successfully loaded building-aware Random Forest model and mapping into memory!")
    except Exception as e:
        model = None
        building_mapping = {}
        print(f"❌ Error loading model file: {e}")
else:
    print("⚠️ Warning: Trained model not found! Falling back to baseline logic.")

def predict_energy_load(payload: EnergyPredictionRequest) -> float:
    """
    Takes a validated request payload, maps the building ID string to its encoded integer,
    and passes the full feature vector into our trained building-aware model.
    """
    if model is None:
        # Fallback heuristic if model file is missing
        return round(50.0 + (payload.air_temperature * 1.5), 2)
    
    try:
        # Map incoming building string to encoded integer (default to 0 if unknown building)
        building_code = building_mapping.get(payload.building_id, 0)
        
        # Wrap features in a DataFrame with explicit column names matching training exactly
        features_df = pd.DataFrame([[
            building_code,
            payload.air_temperature,
            payload.dew_temperature,
            payload.wind_speed,
            payload.cloud_coverage,
            payload.hour_of_day,
            payload.day_of_week
        ]], columns=[
            'building_encoded',
            'air_temperature', 
            'dew_temperature', 
            'wind_speed', 
            'cloud_coverage', 
            'hour_of_day', 
            'day_of_week'
        ])
        
        # Predict using the building-aware Random Forest model
        prediction = model.predict(features_df)[0]
        
        return round(float(prediction), 2)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return 100.0  # Safe default fallback