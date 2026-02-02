"""
FastAPI for House Sales Predictions
Separate endpoints for price prediction and sold prediction
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from database import PredictionDatabase
import os

app = FastAPI(title="House Sales Prediction API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = PredictionDatabase()

# Load models at startup
price_model_data = None
sold_model_data = None
location_encoder = None


@app.on_event("startup")
async def load_models():
    """Load ML models on startup"""
    global price_model_data, sold_model_data, location_encoder

    try:
        with open('price_model.pkl', 'rb') as f:
            price_model_data = pickle.load(f)
        print("✓ Price model loaded")

        with open('sold_model.pkl', 'rb') as f:
            sold_model_data = pickle.load(f)
        print("✓ Sold model loaded")

        with open('location_encoder.pkl', 'rb') as f:
            location_encoder = pickle.load(f)
        print("✓ Location encoder loaded")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train_models.py first to create the model files")

# Request models


class HouseFeatures(BaseModel):
    Square_Footage: float
    Bedrooms: int
    Bathrooms: float
    Age: int
    Garage_Spaces: float
    Lot_Size: float
    Floors: int
    Neighborhood_Rating: int
    Condition: int
    School_Rating: float
    Has_Pool: int
    Renovated: int
    Location_Type: str
    Distance_To_Center_KM: float
    Days_On_Market: Optional[float] = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "Square_Footage": 2500,
                "Bedrooms": 4,
                "Bathrooms": 2.5,
                "Age": 10,
                "Garage_Spaces": 2,
                "Lot_Size": 8000,
                "Floors": 2,
                "Neighborhood_Rating": 7,
                "Condition": 8,
                "School_Rating": 8,
                "Has_Pool": 1,
                "Renovated": 0,
                "Location_Type": "Suburban",
                "Distance_To_Center_KM": 15.5,
                "Days_On_Market": 0
            }
        }


def prepare_features(features: HouseFeatures) -> np.ndarray:
    """Prepare features for prediction"""
    # Encode location type
    location_encoded = location_encoder.transform([features.Location_Type])[0]

    # Create feature array in the correct order
    feature_dict = features.dict()
    del feature_dict['Location_Type']
    feature_dict['Location_Type_Encoded'] = location_encoded

    # Get features in the correct order
    feature_order = price_model_data['features']  # Same order for both models
    feature_array = np.array([[feature_dict.get(feat, 0)
                             for feat in feature_order]])

    return feature_array


@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "price_model": price_model_data is not None,
            "sold_model": sold_model_data is not None,
            "location_encoder": location_encoder is not None
        }
    }


@app.post("/predict-price")
async def predict_price(features: HouseFeatures):
    """
    Predict house price using Linear Regression
    """
    if price_model_data is None:
        raise HTTPException(
            status_code=503, detail="Price model not loaded. Run train_models.py first.")

    try:
        # Prepare features
        feature_array = prepare_features(features)

        # Scale features
        feature_scaled = price_model_data['scaler'].transform(feature_array)

        # Make prediction
        predicted_price = float(
            price_model_data['model'].predict(feature_scaled)[0])

        # Save to database
        prediction_id = db.save_price_prediction(
            features.dict(), predicted_price)

        return {
            "prediction_id": prediction_id,
            "predicted_price": round(predicted_price, 2),
            "formatted_price": f"${predicted_price:,.2f}",
            "model": "Linear Regression"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-sold")
async def predict_sold(features: HouseFeatures):
    """
    Predict if house will be sold within a week using Logistic Regression
    """
    if sold_model_data is None:
        raise HTTPException(
            status_code=503, detail="Sold model not loaded. Run train_models.py first.")

    try:
        # Prepare features
        feature_array = prepare_features(features)

        # Scale features
        feature_scaled = sold_model_data['scaler'].transform(feature_array)

        # Make prediction
        predicted_sold = int(
            sold_model_data['model'].predict(feature_scaled)[0])
        probability = float(
            sold_model_data['model'].predict_proba(feature_scaled)[0][1])

        # Save to database
        prediction_id = db.save_sold_prediction(
            features.dict(), predicted_sold, probability)

        return {
            "prediction_id": prediction_id,
            "will_sell_within_week": bool(predicted_sold),
            "probability": round(probability * 100, 2),
            "confidence": "High" if probability > 0.7 or probability < 0.3 else "Medium",
            "model": "Logistic Regression"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-both")
async def predict_both(features: HouseFeatures):
    """
    Get both price and sold predictions in one call
    """
    if price_model_data is None or sold_model_data is None:
        raise HTTPException(
            status_code=503, detail="Models not loaded. Run train_models.py first.")

    try:
        # Prepare features
        feature_array = prepare_features(features)

        # Price prediction
        feature_scaled_price = price_model_data['scaler'].transform(
            feature_array)
        predicted_price = float(
            price_model_data['model'].predict(feature_scaled_price)[0])
        price_id = db.save_price_prediction(features.dict(), predicted_price)

        # Sold prediction
        feature_scaled_sold = sold_model_data['scaler'].transform(
            feature_array)
        predicted_sold = int(
            sold_model_data['model'].predict(feature_scaled_sold)[0])
        probability = float(sold_model_data['model'].predict_proba(
            feature_scaled_sold)[0][1])
        sold_id = db.save_sold_prediction(
            features.dict(), predicted_sold, probability)

        return {
            "price_prediction": {
                "prediction_id": price_id,
                "predicted_price": round(predicted_price, 2),
                "formatted_price": f"${predicted_price:,.2f}"
            },
            "sold_prediction": {
                "prediction_id": sold_id,
                "will_sell_within_week": bool(predicted_sold),
                "probability": round(probability * 100, 2),
                "confidence": "High" if probability > 0.7 or probability < 0.3 else "Medium"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions"""
    try:
        price_preds = db.get_recent_predictions('price_predictions', limit)
        sold_preds = db.get_recent_predictions('sold_predictions', limit)

        return {
            "price_predictions": price_preds,
            "sold_predictions": sold_preds
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching predictions: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """Get prediction statistics"""
    try:
        stats = db.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching statistics: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting House Sales Prediction API Server")
    print("="*60)
    print("API Documentation: http://localhost:8000/docs")
    print("Main Interface: http://localhost:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
