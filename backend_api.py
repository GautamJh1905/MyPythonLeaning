# =============================================================================
# BACKEND API LAYER - FastAPI
# =============================================================================
"""
This module provides the REST API endpoints for the loan prediction system.
It handles model training, predictions, and database operations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Import database layer
from db_layer import LoanDatabase

# Initialize FastAPI app
app = FastAPI(
    title="Loan Prediction API",
    description="API for predicting loan approval/rejection",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = LoanDatabase()

# Global variable to store the trained model
model_pipeline = None
model_accuracy = None


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class LoanApplication(BaseModel):
    """Model for loan application input"""
    married: str
    dependents: str
    education: str
    self_employed: str
    property_area: str
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_term: int
    credit_history: int


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    prediction: str
    prediction_value: int
    confidence: float
    probability_approved: float
    probability_rejected: float
    record_id: int
    message: str


class ModelInfo(BaseModel):
    """Model information response"""
    model_trained: bool
    accuracy: Optional[float]
    total_records: int
    message: str


class StatisticsResponse(BaseModel):
    """Statistics response"""
    total_predictions: int
    total_approved: int
    total_rejected: int
    approval_rate: float
    average_confidence: float


# =============================================================================
# Helper Functions
# =============================================================================

def train_model():
    """Train the machine learning model"""
    global model_pipeline, model_accuracy

    try:
        # Load data
        df = pd.read_csv("Loan_dataset.csv")
        df = df.dropna(subset=["Loan_Status"])

        # Prepare features and target
        X = df.drop(columns=["Loan_Status", "Loan_ID", "Gender"])
        y = df["Loan_Status"].map({"Y": 1, "N": 0})

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define column types
        log_cols = ["ApplicantIncome", "CoapplicantIncome"]
        num_cols = ["LoanAmount", "Credit_History", "Loan_Amount_Term"]
        cat_cols = ["Married", "Self_Employed",
                    "Education", "Property_Area", "Dependents"]

        # Create pipelines
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        log_numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine pipelines
        preprocessor = ColumnTransformer([
            ("log_num", log_numeric_pipeline, log_cols),
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ])

        # Create and train model pipeline
        model_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
        ])

        model_pipeline.fit(X_train, y_train)

        # Calculate accuracy
        y_pred = model_pipeline.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)

        # Save model to disk
        with open('loan_model.pkl', 'wb') as f:
            pickle.dump(model_pipeline, f)

        return True, model_accuracy, len(df)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error training model: {str(e)}")


def load_model():
    """Load the trained model from disk"""
    global model_pipeline, model_accuracy

    if os.path.exists('loan_model.pkl'):
        with open('loan_model.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)
        return True
    return False


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_pipeline, model_accuracy

    # Try to load existing model
    if not load_model():
        # Train model if not found
        print("No saved model found. Training new model...")
        train_model()
    else:
        print("✅ Model loaded successfully")


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Loan Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "train": "/train",
            "predict": "/predict",
            "model_info": "/model-info",
            "predictions": "/predictions",
            "statistics": "/statistics"
        }
    }


@app.post("/train", response_model=ModelInfo)
def train_model_endpoint():
    """Train or retrain the model"""
    try:
        success, accuracy, total_records = train_model()
        return ModelInfo(
            model_trained=success,
            accuracy=accuracy,
            total_records=total_records,
            message=f"Model trained successfully with {accuracy*100:.2f}% accuracy"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfo)
def get_model_info():
    """Get information about the current model"""
    global model_pipeline, model_accuracy

    if model_pipeline is None:
        return ModelInfo(
            model_trained=False,
            accuracy=None,
            total_records=0,
            message="Model not trained yet"
        )

    df = pd.read_csv("Loan_dataset.csv")

    return ModelInfo(
        model_trained=True,
        accuracy=model_accuracy,
        total_records=len(df),
        message="Model is ready for predictions"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplication):
    """Make a loan prediction"""
    global model_pipeline

    if model_pipeline is None:
        raise HTTPException(
            status_code=400, detail="Model not trained. Please train the model first.")

    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'Married': [application.married],
            'Self_Employed': [application.self_employed],
            'Education': [application.education],
            'Property_Area': [application.property_area],
            'ApplicantIncome': [application.applicant_income],
            'CoapplicantIncome': [application.coapplicant_income],
            'LoanAmount': [application.loan_amount],
            'Credit_History': [application.credit_history],
            'Loan_Amount_Term': [application.loan_term],
            'Dependents': [application.dependents]
        })

        # Make prediction
        prediction = model_pipeline.predict(input_data)[0]
        prediction_proba = model_pipeline.predict_proba(input_data)[0]

        prediction_label = 'Approved' if prediction == 1 else 'Rejected'
        confidence = prediction_proba[prediction] * 100

        # Save to database
        record_id = db.save_prediction(
            married=application.married,
            dependents=application.dependents,
            education=application.education,
            self_employed=application.self_employed,
            property_area=application.property_area,
            applicant_income=application.applicant_income,
            coapplicant_income=application.coapplicant_income,
            loan_amount=application.loan_amount,
            loan_term=application.loan_term,
            credit_history=application.credit_history,
            prediction=prediction_label,
            confidence=confidence
        )

        return PredictionResponse(
            prediction=prediction_label,
            prediction_value=int(prediction),
            confidence=confidence,
            probability_approved=prediction_proba[1] * 100,
            probability_rejected=prediction_proba[0] * 100,
            record_id=record_id,
            message=f"Loan {prediction_label.lower()} with {confidence:.2f}% confidence"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/predictions")
def get_all_predictions():
    """Get all predictions from database"""
    try:
        predictions = db.get_all_predictions()
        return {
            "total": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving predictions: {str(e)}")


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    """Get a specific prediction by ID"""
    try:
        prediction = db.get_prediction_by_id(prediction_id)
        if prediction is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving prediction: {str(e)}")


@app.get("/statistics", response_model=StatisticsResponse)
def get_statistics():
    """Get statistics about predictions"""
    try:
        stats = db.get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving statistics: {str(e)}")


@app.delete("/predictions/{prediction_id}")
def delete_prediction(prediction_id: int):
    """Delete a prediction by ID"""
    try:
        deleted = db.delete_prediction(prediction_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return {"message": f"Prediction {prediction_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting prediction: {str(e)}")


# =============================================================================
# Run the API
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Loan Prediction API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
