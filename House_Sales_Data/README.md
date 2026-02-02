# House Sales Prediction System 🏠

Complete ML-powered system for predicting house prices and sale probability using Linear and Logistic Regression.

## Features

- **Linear Regression**: Predict house prices based on property features
- **Logistic Regression**: Predict if a house will sell within a week
- **Cross-Validation**: Both models trained with 5-fold cross-validation
- **Separate APIs**: Individual endpoints for each prediction type
- **Database Storage**: All predictions saved to SQLite database
- **Beautiful Web Interface**: User-friendly frontend for making predictions
- **Real-time Statistics**: View prediction statistics and history

## Project Structure

```
House_Sales_Data/
├── house_sales_data.csv        # Training dataset
├── train_models.py             # Model training script with cross-validation
├── database.py                 # Database layer for storing predictions
├── house_sales_api.py          # FastAPI backend with separate endpoints
├── index.html                  # Web interface
├── start_app.bat               # Windows startup script
├── price_model.pkl             # Trained Linear Regression model (generated)
├── sold_model.pkl              # Trained Logistic Regression model (generated)
├── location_encoder.pkl        # Label encoder (generated)
└── house_predictions.db        # SQLite database (generated)
```

## Installation

1. **Install required packages:**
```bash
pip install pandas numpy scikit-learn fastapi uvicorn
```

2. **Navigate to the project directory:**
```bash
cd House_Sales_Data
```

## Usage

### Option 1: Quick Start (Windows)
Double-click `start_app.bat` or run:
```bash
start_app.bat
```

This will:
1. Train both models with cross-validation
2. Start the API server
3. Open the web interface

### Option 2: Manual Start

1. **Train the models:**
```bash
python train_models.py
```

2. **Start the API server:**
```bash
python house_sales_api.py
```

3. **Access the web interface:**
Open your browser and go to: http://localhost:8000

## API Endpoints

### 1. Predict Price (Linear Regression)
```
POST /predict-price
```
Returns predicted house price

### 2. Predict Sold (Logistic Regression)
```
POST /predict-sold
```
Returns probability of selling within a week

### 3. Predict Both
```
POST /predict-both
```
Returns both predictions in one call

### 4. Get Recent Predictions
```
GET /predictions/recent?limit=10
```
Returns recent predictions from database

### 5. Get Statistics
```
GET /statistics
```
Returns prediction statistics

### 6. Health Check
```
GET /health
```
Check if models are loaded

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Input Features

The system uses the following features for predictions:

- **Square_Footage**: Property size (500-10,000 sq ft)
- **Bedrooms**: Number of bedrooms (1-10)
- **Bathrooms**: Number of bathrooms (1-10)
- **Age**: Property age in years (0-100)
- **Garage_Spaces**: Number of garage spaces (0-5)
- **Lot_Size**: Lot size in square feet (1,000-50,000)
- **Floors**: Number of floors (1-4)
- **Neighborhood_Rating**: Rating 1-10
- **Condition**: Property condition 1-10
- **School_Rating**: Nearby school rating 1-10
- **Has_Pool**: 0 (No) or 1 (Yes)
- **Renovated**: 0 (No) or 1 (Yes)
- **Location_Type**: Suburban, Urban, or Downtown
- **Distance_To_Center_KM**: Distance to city center (0-100 km)
- **Days_On_Market**: Days listed (0-365)

## Model Performance

### Linear Regression (Price Prediction)
- Uses 5-fold cross-validation
- Reports R² score and RMSE
- Saves model with StandardScaler

### Logistic Regression (Sold Prediction)
- Uses 5-fold cross-validation
- Reports accuracy and classification metrics
- Includes confusion matrix (crosstab)
- Returns probability scores

## Database

All predictions are automatically saved to `house_predictions.db` with:
- Timestamp
- All input features
- Predicted values
- Probability scores (for sold predictions)

## Example Usage

### Using the Web Interface:
1. Fill in property details
2. Click "Predict Price" for price prediction
3. Click "Predict Sale" for sold prediction
4. Click "Predict Both" for both predictions

### Using Python:
```python
import requests

data = {
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

# Predict price
response = requests.post("http://localhost:8000/predict-price", json=data)
print(response.json())

# Predict sold
response = requests.post("http://localhost:8000/predict-sold", json=data)
print(response.json())

# Get both predictions
response = requests.post("http://localhost:8000/predict-both", json=data)
print(response.json())
```

## Technologies Used

- **Python**: Core programming language
- **scikit-learn**: Machine learning models
- **FastAPI**: Modern web framework
- **SQLite**: Database for storing predictions
- **HTML/CSS/JavaScript**: Frontend interface
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

## Model Files

After training, three pickle files are created:
1. `price_model.pkl`: Contains Linear Regression model, scaler, and feature names
2. `sold_model.pkl`: Contains Logistic Regression model, scaler, and feature names
3. `location_encoder.pkl`: LabelEncoder for Location_Type feature

## Notes

- Models use StandardScaler for feature normalization
- Cross-validation ensures robust model performance
- All predictions are logged with timestamps
- Web interface includes real-time statistics
- API supports CORS for frontend integration

## Troubleshooting

If you encounter errors:

1. **Models not found**: Run `train_models.py` first
2. **Port 8000 in use**: Change port in `house_sales_api.py`
3. **Import errors**: Install required packages
4. **Database errors**: Delete `house_predictions.db` and restart

## Future Enhancements

- Add more regression models (Ridge, Lasso, Random Forest)
- Implement feature importance visualization
- Add model comparison dashboard
- Export predictions to CSV
- Add authentication for API
- Deploy to cloud platform

---

**Created for Machine Learning House Sales Prediction Project**
