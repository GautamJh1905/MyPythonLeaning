"""
FastAPI Application for Customer Segmentation
Loads trained KMeans model and provides prediction endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="API for predicting customer segments using K-Means clustering",
    version="1.0.0"
)

# Global variables to store loaded models
kmeans_model = None
scaler = None
label_encodings = None

# Cluster to segment mapping with business recommendations
CLUSTER_SEGMENTS = {
    0: {
        "name": "Price-Sensitive Occasional",
        "description": "Low-income, low-spending customers with minimal engagement",
        "characteristics": [
            "Lowest annual income (~$262K)",
            "Lowest total spending (~$54K)",
            "Minimal monthly purchases (2.2)",
            "Low app engagement (~19 mins)"
        ],
        "offers": [
            "Discount coupons and promotional codes",
            "Free shipping on first order",
            "Loyalty rewards program signup bonus",
            "Budget-friendly product recommendations"
        ]
    },
    1: {
        "name": "High-Value Loyal",
        "description": "Premium customers with highest income and engagement",
        "characteristics": [
            "Highest annual income (~$1.03M)",
            "Highest total spending (~$663K)",
            "Most frequent purchases (16.1/month)",
            "Highest app engagement (~116 mins)"
        ],
        "offers": [
            "Exclusive VIP member benefits",
            "Early access to new products",
            "Premium customer service",
            "Personalized luxury product recommendations"
        ]
    },
    2: {
        "name": "Value-Seeking Regular",
        "description": "Mid-tier customers with moderate income and regular engagement",
        "characteristics": [
            "Mid-range annual income (~$585K)",
            "Moderate total spending (~$260K)",
            "Regular monthly purchases (8.8)",
            "Moderate app engagement (~68 mins)"
        ],
        "offers": [
            "Seasonal sales and limited-time offers",
            "Bundle deals and combo packages",
            "Referral bonuses for inviting friends",
            "Mid-range product recommendations with best value"
        ]
    }
}


@app.on_event("startup")
async def load_models():
    """Load trained models and preprocessing objects on startup"""
    global kmeans_model, scaler, label_encodings

    try:
        # Load KMeans model
        with open('models/kmeans_customer_segmentation.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)

        # Load StandardScaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load label encodings
        with open('models/label_encodings.pkl', 'rb') as f:
            label_encodings = pickle.load(f)

        print("✓ All models loaded successfully!")
        print(f"  - KMeans clusters: {kmeans_model.n_clusters}")
        print(f"  - Features: {len(scaler.feature_names_in_)}")
        print(f"  - Categorical columns: {list(label_encodings.keys())}")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise


class CustomerInput(BaseModel):
    """Input model for customer data"""
    CustomerID: str = Field(..., description="Unique customer identifier")
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Gender: str = Field(..., description="Customer gender (M/F)")
    City: str = Field(..., description="Customer city")
    AnnualIncome: float = Field(..., gt=0,
                                description="Annual income in dollars")
    TotalSpent: float = Field(..., ge=0, description="Total amount spent")
    MonthlyPurchases: int = Field(..., ge=0,
                                  description="Number of monthly purchases")
    AvgOrderValue: float = Field(..., ge=0, description="Average order value")
    AppTimeMinutes: float = Field(..., ge=0,
                                  description="Time spent on app in minutes")
    DiscountUsage: str = Field(..., description="Discount usage level")
    PreferredShoppingTime: str = Field(...,
                                       description="Preferred shopping time (Day/Night)")

    class Config:
        json_schema_extra = {
            "example": {
                "CustomerID": "C001",
                "Age": 28,
                "Gender": "F",
                "City": "Mumbai",
                "AnnualIncome": 450000.0,
                "TotalSpent": 120000.0,
                "MonthlyPurchases": 6,
                "AvgOrderValue": 2000.0,
                "AppTimeMinutes": 45.0,
                "DiscountUsage": "Medium",
                "PreferredShoppingTime": "Night"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    cluster: int
    segment_name: str
    segment_description: str
    characteristics: List[str]
    recommended_offers: List[str]
    confidence_score: float


def preprocess_customer_data(customer: CustomerInput) -> np.ndarray:
    """
    Preprocess customer data: encode categorical variables and scale features

    Args:
        customer: CustomerInput object with raw customer data

    Returns:
        Scaled feature array ready for prediction
    """
    # Create a dictionary from customer data
    data = customer.model_dump()

    # Create DataFrame
    df = pd.DataFrame([data])

    # Encode categorical variables using saved mappings
    for col, encoding in label_encodings.items():
        if col in df.columns:
            # Get the mapping
            mapping = encoding['mapping']
            # Encode the value
            if df[col].iloc[0] in mapping:
                df[col] = mapping[df[col].iloc[0]]
            else:
                # If value not in mapping, raise error
                raise ValueError(
                    f"Unknown value '{df[col].iloc[0]}' for column '{col}'")

    # Ensure correct column order
    feature_columns = list(scaler.feature_names_in_)
    df_ordered = df[feature_columns]

    # Scale features
    scaled_data = scaler.transform(df_ordered)
    return scaled_data
    return scaled_data


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict customer segment",
            "GET /segments": "Get all segment information",
            "GET /health": "Check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([kmeans_model is not None, scaler is not None, label_encodings is not None])
    }


@app.get("/segments")
async def get_segments():
    """Get information about all customer segments"""
    return {
        "total_clusters": kmeans_model.n_clusters if kmeans_model else 0,
        "segments": CLUSTER_SEGMENTS
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_segment(customer: CustomerInput):
    """
    Predict customer segment based on provided data

    Args:
        customer: CustomerInput with customer details

    Returns:
        PredictionResponse with cluster, segment info, and recommendations
    """
    try:
        # Preprocess the customer data
        scaled_data = preprocess_customer_data(customer)

        # Predict cluster
        cluster = int(kmeans_model.predict(scaled_data)[0])

        # Calculate confidence (distance to cluster center)
        distances = kmeans_model.transform(scaled_data)[0]
        closest_distance = distances[cluster]
        # Convert distance to confidence score (closer = higher confidence)
        max_distance = np.max(distances)
        confidence = float(1 - (closest_distance / max_distance)
                           ) if max_distance > 0 else 1.0

        # Get segment information
        segment_info = CLUSTER_SEGMENTS.get(cluster, {
            "name": f"Cluster {cluster}",
            "description": "Unknown segment",
            "characteristics": [],
            "offers": []
        })

        # Create response
        response = PredictionResponse(
            cluster=cluster,
            segment_name=segment_info["name"],
            segment_description=segment_info["description"],
            characteristics=segment_info["characteristics"],
            recommended_offers=segment_info["offers"],
            confidence_score=round(confidence, 4)
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
