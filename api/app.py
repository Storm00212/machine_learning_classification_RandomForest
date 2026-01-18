from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Animal Disease Prediction API", version="1.0.0")

# Define input models
class DairyInput(BaseModel):
    body_temperature_c: float = Field(..., ge=35.0, le=42.0)
    milk_yield_l: float = Field(..., ge=0.0)
    milk_conductivity: float = Field(..., ge=0.0)
    feed_intake_kg: float = Field(..., ge=0.0)

class BeefInput(BaseModel):
    body_temperature_c: float = Field(..., ge=35.0, le=42.0)
    feed_intake_kg: float = Field(..., ge=0.0)
    water_intake_l: float = Field(..., ge=0.0)
    respiratory_rate_bpm: float = Field(..., ge=0.0)
    tick_load: str = Field(..., pattern="^(low|moderate)$")

class PoultryInput(BaseModel):
    body_temperature_c: float = Field(..., ge=35.0, le=45.0)
    feed_intake_kg: float = Field(..., ge=0.0)
    water_intake_l: float = Field(..., ge=0.0)
    daily_mortality_percent: float = Field(..., ge=0.0, le=100.0)

# Load scalers and models
models = {}
scalers = {}

def load_resources():
    datasets = ['dairy', 'beef', 'poultry']
    for dataset in datasets:
        try:
            # Load model
            model_path = f"trained_models/{dataset}_model.pkl"
            models[dataset] = joblib.load(model_path)
            logger.info(f"Loaded model for {dataset}")

            # Load original data to fit scaler
            data_path = f"data/{dataset}_realistic_health_dataset.csv"
            data = pd.read_csv(data_path)

            # Fit scaler on numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            scaler.fit(data[numerical_cols])
            scalers[dataset] = {'scaler': scaler, 'numerical_cols': numerical_cols}
            logger.info(f"Loaded scaler for {dataset}")

        except Exception as e:
            logger.error(f"Failed to load resources for {dataset}: {e}")

load_resources()

def preprocess_dairy(input_data: DairyInput) -> np.ndarray:
    data = {
        'body_temperature_c': input_data.body_temperature_c,
        'milk_yield_l': input_data.milk_yield_l,
        'milk_conductivity': input_data.milk_conductivity,
        'feed_intake_kg': input_data.feed_intake_kg
    }
    df = pd.DataFrame([data])
    scaler_info = scalers['dairy']
    df[scaler_info['numerical_cols']] = scaler_info['scaler'].transform(df[scaler_info['numerical_cols']])
    return df.values

def preprocess_beef(input_data: BeefInput) -> np.ndarray:
    data = {
        'body_temperature_c': input_data.body_temperature_c,
        'feed_intake_kg': input_data.feed_intake_kg,
        'water_intake_l': input_data.water_intake_l,
        'respiratory_rate_bpm': input_data.respiratory_rate_bpm,
        'tick_load_low': 1 if input_data.tick_load == 'low' else 0,
        'tick_load_moderate': 1 if input_data.tick_load == 'moderate' else 0
    }
    df = pd.DataFrame([data])
    scaler_info = scalers['beef']
    numerical_cols = [col for col in scaler_info['numerical_cols'] if col in df.columns]
    df[numerical_cols] = scaler_info['scaler'].transform(df[numerical_cols])
    return df.values

def preprocess_poultry(input_data: PoultryInput) -> np.ndarray:
    data = {
        'body_temperature_c': input_data.body_temperature_c,
        'feed_intake_kg': input_data.feed_intake_kg,
        'water_intake_l': input_data.water_intake_l,
        'daily_mortality_percent': input_data.daily_mortality_percent
    }
    df = pd.DataFrame([data])
    scaler_info = scalers['poultry']
    df[scaler_info['numerical_cols']] = scaler_info['scaler'].transform(df[scaler_info['numerical_cols']])
    return df.values

@app.post("/predict/{dataset_name}")
async def predict(dataset_name: str, input_data: Dict[str, Any]):
    logger.info(f"Received prediction request for {dataset_name}")

    if dataset_name not in models:
        raise HTTPException(status_code=404, detail=f"Model for {dataset_name} not found")

    try:
        if dataset_name == 'dairy':
            validated_input = DairyInput(**input_data)
            processed_data = preprocess_dairy(validated_input)
        elif dataset_name == 'beef':
            validated_input = BeefInput(**input_data)
            processed_data = preprocess_beef(validated_input)
        elif dataset_name == 'poultry':
            validated_input = PoultryInput(**input_data)
            processed_data = preprocess_poultry(validated_input)
        else:
            raise HTTPException(status_code=400, detail="Invalid dataset name")

        model = models[dataset_name]
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]

        # Get class names
        class_names = model.classes_

        result = {
            "predicted_disease": prediction,
            "probabilities": {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
        }

        logger.info(f"Prediction completed for {dataset_name}: {prediction}")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    return {"message": "Animal Disease Prediction API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)