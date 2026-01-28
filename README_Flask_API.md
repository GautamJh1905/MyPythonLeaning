# Loan Prediction Flask API

A Flask-based REST API for loan approval prediction using a trained Logistic Regression model.

## 🎯 Features

- ✅ Single loan application prediction
- ✅ Batch predictions for multiple applicants
- ✅ Automatic validation and fixing of invalid Credit_History values
- ✅ Health check endpoint
- ✅ Detailed prediction probabilities
- ✅ Comprehensive error handling

## 📋 Prerequisites

- Python 3.7+
- Virtual environment (recommended)
- Required packages: flask, joblib, pandas, scikit-learn, numpy

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install flask joblib pandas scikit-learn numpy requests
```

### 2. Train and Save the Model

Run the Jupyter notebook `ML_Loan_DataSet_Logistic_Regression.ipynb` to train the model and save it as `loan_model.pkl`.

### 3. Start the Flask Server

```bash
python Flask_Loan_Prediction.py
```

The server will start on `http://127.0.0.1:5001`

### 4. Test the API

```bash
python test_flask_api.py
```

## 📡 API Endpoints

### 1. Home / API Information
- **URL**: `/`
- **Method**: GET
- **Response**: API information and available endpoints

```bash
curl http://127.0.0.1:5001/
```

### 2. Health Check
- **URL**: `/health`
- **Method**: GET
- **Response**: Server health status and model loading status

```bash
curl http://127.0.0.1:5001/health
```

### 3. Single Prediction
- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: application/json

#### Request Body:
```json
{
    "Married": "Yes",
    "Self_Employed": "No",
    "Education": "Graduate",
    "Property_Area": "Urban",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Credit_History": 1,
    "Loan_Amount_Term": 360,
    "Dependents": "2"
}
```

#### Response:
```json
{
    "prediction": "Approved",
    "prediction_value": 1,
    "approval_probability": 0.7842,
    "rejection_probability": 0.2158,
    "input_data": { ... },
    "validation_report": []
}
```

#### Example using curl:
```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d "{\"Married\": \"Yes\", \"Self_Employed\": \"No\", \"Education\": \"Graduate\", \"Property_Area\": \"Urban\", \"ApplicantIncome\": 5000, \"CoapplicantIncome\": 2000, \"LoanAmount\": 150, \"Credit_History\": 1, \"Loan_Amount_Term\": 360, \"Dependents\": \"2\"}"
```

#### Example using Python:
```python
import requests

data = {
    "Married": "Yes",
    "Self_Employed": "No",
    "Education": "Graduate",
    "Property_Area": "Urban",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Credit_History": 1,
    "Loan_Amount_Term": 360,
    "Dependents": "2"
}

response = requests.post('http://127.0.0.1:5001/predict', json=data)
print(response.json())
```

### 4. Batch Prediction
- **URL**: `/predict_batch`
- **Method**: POST
- **Content-Type**: application/json

#### Request Body:
```json
{
    "applicants": [
        {
            "Married": "Yes",
            "Self_Employed": "No",
            "Education": "Graduate",
            "Property_Area": "Urban",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 150,
            "Credit_History": 1,
            "Loan_Amount_Term": 360,
            "Dependents": "2"
        },
        {
            "Married": "No",
            "Self_Employed": "Yes",
            "Education": "Not Graduate",
            "Property_Area": "Rural",
            "ApplicantIncome": 3000,
            "CoapplicantIncome": 0,
            "LoanAmount": 100,
            "Credit_History": 0,
            "Loan_Amount_Term": 360,
            "Dependents": "0"
        }
    ]
}
```

#### Response:
```json
{
    "total_applicants": 2,
    "approved_count": 1,
    "rejected_count": 1,
    "validation_report": [],
    "results": [
        {
            "applicant_index": 0,
            "prediction": "Approved",
            "prediction_value": 1,
            "approval_probability": 0.7842,
            "rejection_probability": 0.2158
        },
        {
            "applicant_index": 1,
            "prediction": "Rejected",
            "prediction_value": 0,
            "approval_probability": 0.0206,
            "rejection_probability": 0.9794
        }
    ]
}
```

## 🛡️ Data Validation

The API automatically validates and fixes invalid `Credit_History` values:
- Valid values: 0 (no credit history) or 1 (good credit history)
- Invalid values (negative numbers) are automatically replaced with 0
- Validation report is included in the response

### Example with Invalid Data:
```json
{
    "Credit_History": -5,
    ...other fields...
}
```

Response will include:
```json
{
    "validation_report": [
        "WARNING: Found 1 invalid Credit_History value(s): [-5]",
        "FIXED: Replaced negative Credit_History values with 0"
    ],
    ...
}
```

## 📊 Input Fields

| Field | Type | Values | Required |
|-------|------|--------|----------|
| Married | String | "Yes", "No" | Yes |
| Self_Employed | String | "Yes", "No" | Yes |
| Education | String | "Graduate", "Not Graduate" | Yes |
| Property_Area | String | "Urban", "Semiurban", "Rural" | Yes |
| ApplicantIncome | Number | Positive integer | Yes |
| CoapplicantIncome | Number | Non-negative integer | Yes |
| LoanAmount | Number | Positive integer | Yes |
| Credit_History | Number | 0 or 1 (negative values auto-fixed) | Yes |
| Loan_Amount_Term | Number | Positive integer (e.g., 360, 180) | Yes |
| Dependents | String | "0", "1", "2", "3+" | Yes |

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_flask_api.py
```

The test suite includes:
- Health check test
- Single prediction test
- Negative credit history handling test
- Batch prediction test

## 📁 Project Structure

```
Machine_Learning/
│
├── Flask_Loan_Prediction.py          # Main Flask API application
├── test_flask_api.py                  # API test suite
├── loan_model.pkl                     # Trained model (generated from notebook)
├── ML_Loan_DataSet_Logistic_Regression.ipynb  # Model training notebook
├── Loan_dataset.csv                   # Training data
└── README_Flask_API.md               # This file
```

## 🔧 Configuration

To change the port or other settings, modify the Flask app configuration in `Flask_Loan_Prediction.py`:

```python
if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
```

## ⚠️ Important Notes

1. **Development Server**: The Flask development server is suitable for testing only. For production, use a WSGI server like Gunicorn or uWSGI.

2. **Model File**: Ensure `loan_model.pkl` exists in the same directory as the Flask app.

3. **Debug Mode**: Debug mode is enabled for development. Disable it in production.

4. **Port Conflict**: If port 5001 is in use, change it in the code.

## 🐛 Troubleshooting

### Error: "Model not loaded"
- Ensure `loan_model.pkl` exists in the same directory
- Retrain the model using the Jupyter notebook

### Error: "Cannot connect to Flask server"
- Check if the server is running
- Verify the correct port (5001)
- Check firewall settings

### Error: "Missing required columns"
- Ensure all required fields are included in the request
- Check field names for typos (case-sensitive)

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Created for Machine Learning practice - January 2026
