# Customer Segmentation API

## Project Overview
FastAPI-based REST API for customer segmentation using K-Means clustering. The API predicts which customer segment a new customer belongs to and provides personalized offers.

## Customer Segments

### Cluster 0: Price-Sensitive Occasional
- **Characteristics:** Low income (~$262K), minimal spending (~$54K), infrequent purchases
- **Offers:** Discounts, free shipping, budget-friendly recommendations

### Cluster 1: High-Value Loyal  
- **Characteristics:** Highest income (~$1.03M), highest spending (~$663K), most engaged
- **Offers:** VIP benefits, exclusive access, premium service

### Cluster 2: Value-Seeking Regular
- **Characteristics:** Mid-range income (~$585K), moderate spending (~$260K), regular purchases
- **Offers:** Seasonal sales, bundle deals, referral bonuses

## Project Structure
```
ML_assessment/
├── app.py                      # FastAPI application
├── test_api.py                 # API testing script
├── requirements.txt            # Python dependencies
├── models/                     # Saved model artifacts
│   ├── kmeans_customer_segmentation.pkl
│   ├── scaler.pkl
│   └── label_encodings.pkl
├── CustomerData.csv           # Training data
└── ML_Unsupervised_Learning_Assessment.ipynb  # Training notebook
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python app.py
```
The API will start on `http://localhost:8000`

### 3. Access API Documentation
Open your browser and navigate to:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint to verify API and models are loaded

### `GET /segments`
Get information about all customer segments

### `POST /predict`
Predict customer segment based on provided data

**Request Body:**
```json
{
  "CustomerID": "C001",
  "Age": 28,
  "Gender": "Female",
  "City": "Mumbai",
  "AnnualIncome": 450000.0,
  "TotalSpent": 120000.0,
  "MonthlyPurchases": 6,
  "AvgOrderValue": 2000.0,
  "AppTimeMinutes": 45.0,
  "DiscountUsage": "Medium",
  "PreferredShoppingTime": "Night"
}
```

**Response:**
```json
{
  "cluster": 2,
  "segment_name": "Value-Seeking Regular",
  "segment_description": "Mid-tier customers with moderate income and regular engagement",
  "characteristics": [
    "Mid-range annual income (~$585K)",
    "Moderate total spending (~$260K)",
    "Regular monthly purchases (8.8)",
    "Moderate app engagement (~68 mins)"
  ],
  "recommended_offers": [
    "Seasonal sales and limited-time offers",
    "Bundle deals and combo packages",
    "Referral bonuses for inviting friends",
    "Mid-range product recommendations with best value"
  ],
  "confidence_score": 0.8542
}
```

## Testing the API

Run the test script to validate all endpoints:
```bash
python test_api.py
```

This will test:
- Root endpoint
- Health check
- Segments information
- Predictions for different customer types

## Using cURL

Test the prediction endpoint with cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "CustomerID": "C001",
    "Age": 35,
    "Gender": "Female",
    "City": "Delhi",
    "AnnualIncome": 600000.0,
    "TotalSpent": 250000.0,
    "MonthlyPurchases": 9,
    "AvgOrderValue": 3500.0,
    "AppTimeMinutes": 70.0,
    "DiscountUsage": "Medium",
    "PreferredShoppingTime": "Day"
  }'
```

## Model Details

- **Algorithm:** K-Means Clustering
- **Number of Clusters:** 3
- **Features:** 11 (CustomerID, Age, Gender, City, AnnualIncome, TotalSpent, MonthlyPurchases, AvgOrderValue, AppTimeMinutes, DiscountUsage, PreferredShoppingTime)
- **Preprocessing:** Label encoding for categorical variables, StandardScaler for normalization
- **Silhouette Score:** 0.3036
- **Training Data:** 86 customers (after outlier removal)

## Notes

- Categorical variables (Gender, City, DiscountUsage, PreferredShoppingTime) are automatically encoded
- The API returns a confidence score based on distance to cluster center
- All models are loaded at startup for fast predictions
