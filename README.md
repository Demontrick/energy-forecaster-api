# ⚡ Building-Aware Energy Load Forecasting API

> A production-grade, asynchronous machine learning API engineered to forecast electrical energy consumption (kWh) using real-world multi-building constraints.
![CI Status](https://github.com/Demontrick/energy-forecaster-api/actions/workflows/ci.yml/badge.svg)
---

## 🚀 Overview

Energy grids and smart campus infrastructures require precise, building-specific load forecasting to optimize power distribution and reduce waste. This project implements a **Building-Aware Machine Learning Pipeline** trained on optimized telemetry consumption and weather data. 

Unlike generic models that treat all buildings uniformly, this system utilizes an asset-encoding mapping architecture. When an inference request is made for a specific physical asset (e.g., `Bear_education_Bob`), the API dynamically translates the asset string into its unique historical baseline while synchronizing hourly meteorological conditions.

---

## 🏗️ Core Architecture & Tech Stack

* **Web Framework:** FastAPI (Asynchronous request handling, automated OpenAPI/Swagger documentation)
* **Machine Learning:** Scikit-learn (Constrained Random Forest Regressor with categorical asset encoding)
* **Data Validation:** Pydantic v2 (Strict schema validation with custom JSON schema examples)
* **Artifact Management:** Joblib (Bundled multi-artifact serialization: Model + Asset Mapping Dictionary)
* **Testing & Quality Assurance:** `pytest` paired with `httpx` for automated endpoint integration testing
* **Containerization & CI/CD:** Docker and GitHub Actions automated pipeline

---

## 📁 Project Structure

energy-forecaster-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application & endpoints
│   ├── models.py          # Artifact loader & inference wrapper
│   ├── schemas.py         # Strict Pydantic v2 data contracts
│   └── models/
│       └── energy_model.pkl # Serialized multi-artifact bundle (Model + Mapping)
│
├── data/
│   └── energy_training_data.csv # Optimized and unified training dataset (~42MB)
│
├── tests/
│   └── test_api.py        # Automated pytest test suite
│
├── .github/
│   └── workflows/
│       └── docker-build.yml # CI/CD pipeline automation
│
├── Dockerfile             # Production container definition
├── requirements.txt       # Pinned project dependencies
└── train.py               # End-to-end data ingestion, merging, and training pipeline

🛠️ Getting Started & Installation
1. Clone the Repository
Bash
git clone [https://github.com/Demontrick/energy-forecaster-api.git](https://github.com/Demontrick/energy-forecaster-api.git)
cd energy-forecaster-api
2. Create and Activate Virtual Environment
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
📊 Training the Pipeline
To execute the data ingestion, asset encoding, and model artifact generation:

Bash
python train.py
This will output the trained Random Forest model and asset mapping bundle directly into app/models/energy_model.pkl.

🧪 Running Automated Tests
Verify system integrity using the automated test suite:

Bash
python -m pytest -v
🔌 Running the API Locally
Start the local development server with Uvicorn:

Bash
uvicorn app.main:app --reload
Once running, you can access:

Interactive Swagger Documentation: http://127.0.0.1:8000/docs

Alternative Redoc Documentation: http://127.0.0.1:8000/redoc

Example Request (POST /predict)
JSON
{
  "building_id": "Bear_education_Bob",
  "air_temperature": 22.5,
  "dew_temperature": 12.0,
  "wind_speed": 3.5,
  "cloud_coverage": 4.0,
  "hour_of_day": 10,
  "day_of_week": 3
}
🐳 Docker Containerization
Build the production container image locally:

Bash
docker build -t energy-forecaster-api:latest .
Run the container:

Bash
docker run -p 8000:8000 energy-forecaster-api:latest
