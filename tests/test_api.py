from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    """Test the health check root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Energy Forecaster API is running!" in data["message"]

def test_predict_endpoint():
    """Test the /predict endpoint with a realistic BDG2 payload."""
    payload = {
        "building_id": "Bear_education_Bob",
        "air_temperature": 22.5,
        "dew_temperature": 12.0,
        "wind_speed": 3.5,
        "cloud_coverage": 4.0,
        "hour_of_day": 10,
        "day_of_week": 3
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["building_id"] == "Bear_education_Bob"
    assert "predicted_load_kwh" in data
    assert isinstance(data["predicted_load_kwh"], float)
    assert data["model_version"] == "v1.0-random-forest"