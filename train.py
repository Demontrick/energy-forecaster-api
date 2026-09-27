import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "app" / "models"

def main():
    print("🚀 Starting building-aware training pipeline...")
    
    # Point directly to your pre-merged unified training dataset
    training_data_path = DATA_DIR / "energy_training_data.csv"
    
    if not training_data_path.exists():
        print(f"❌ Error: Unified training dataset not found at {training_data_path}. Run prepare_data.py first.")
        return

    print("📂 Loading unified training dataset...")
    df_merged = pd.read_csv(training_data_path)
    
    if df_merged.empty:
        print("❌ Warning: Training dataset is empty!")
        return
        
    print(f"✅ Successfully loaded! Total training rows: {len(df_merged)}")
    
    # Standardize weather column names to snake_case if they are camelCase
    df_merged = df_merged.rename(columns={
        'airTemperature': 'air_temperature',
        'dewTemperature': 'dew_temperature',
        'windSpeed': 'wind_speed',
        'cloudCoverage': 'cloud_coverage'
    })
    
    # Ensure timestamp is parsed for feature extraction
    df_merged['timestamp_local'] = pd.to_datetime(df_merged['timestamp_local'])
    
    # Extract temporal features if not already in the CSV
    if 'hour_of_day' not in df_merged.columns:
        df_merged['hour_of_day'] = df_merged['timestamp_local'].dt.hour
    if 'day_of_week' not in df_merged.columns:
        df_merged['day_of_week'] = df_merged['timestamp_local'].dt.dayofweek
        
    # Encode building IDs so the model learns individual building baselines
    df_merged['building_encoded'] = df_merged['source_building_id'].astype('category').cat.codes
    
    # Create a mapping dictionary for inference lookup
    categories = df_merged['source_building_id'].astype('category').cat.categories
    building_mapping = {name: code for code, name in enumerate(categories)}

    feature_cols = [
        'building_encoded',
        'air_temperature', 
        'dew_temperature', 
        'wind_speed', 
        'cloud_coverage', 
        'hour_of_day', 
        'day_of_week'
    ]
    
    df_merged = df_merged.dropna(subset=feature_cols + ['energy_kwh'])
    
    X = df_merged[feature_cols]
    y = df_merged['energy_kwh']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🤖 Training Lightweight Optimized Random Forest Regressor...")
    # Constraining max_depth and estimators keeps the output .pkl file tiny (< 40MB)
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=15,
        min_samples_split=10,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"📊 Training complete! Test Mean Squared Error (MSE): {mse:.4f}")
    
    # Save model artifact bundle (Model + Building Mapping)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "energy_model.pkl"
    
    artifact = {
        "model": model,
        "building_mapping": building_mapping
    }
    
    joblib.dump(artifact, model_path)
    print(f"✨ Building-aware model artifact successfully saved to {model_path}!")

if __name__ == "__main__":
    main()