# 🏦 Loan Prediction System - 3-Tier Architecture

A complete loan approval prediction system built with a modern 3-tier architecture: Frontend, Backend API, and Database.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                           │
│                 (Streamlit Web UI)                          │
│              ML_Streamlit_Loan_API.py                       │
│                  Port: 8501                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST API
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND LAYER                            │
│                  (FastAPI Server)                           │
│                  backend_api.py                             │
│                  Port: 8000                                 │
│  - ML Model Training & Inference                           │
│  - Business Logic                                          │
│  - API Endpoints                                           │
└────────────────────┬────────────────────────────────────────┘
                     │ Database Operations
                     ↓
┌─────────────────────────────────────────────────────────────┐
│                    DATABASE LAYER                           │
│                  (SQLite + ORM)                            │
│                   db_layer.py                              │
│              loan_predictions.db                           │
│  - Data Persistence                                        │
│  - CRUD Operations                                         │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Machine_Learning/
├── ML_Streamlit_Loan_API.py    # Frontend - Streamlit UI
├── backend_api.py               # Backend - FastAPI Server
├── db_layer.py                  # Database - SQLite Operations
├── Loan_dataset.csv             # Training Data
├── loan_predictions.db          # SQLite Database (created on first run)
├── loan_model.pkl               # Trained ML Model (created on first run)
├── start_app.bat                # Windows Startup Script
└── README_3Tier.md              # This file
```

## 🚀 Quick Start

### Option 1: Using Startup Script (Windows)

Simply double-click `start_app.bat` or run:

```bash
start_app.bat
```

This will:
1. Start the Backend API on port 8000
2. Start the Frontend UI on port 8501
3. Open two terminal windows

### Option 2: Manual Start

**Step 1: Start Backend API**

Open a terminal and run:

```bash
python backend_api.py
```

Backend will start on: http://localhost:8000
API Documentation: http://localhost:8000/docs

**Step 2: Start Frontend**

Open another terminal and run:

```bash
streamlit run ML_Streamlit_Loan_API.py
```

Frontend will start on: http://localhost:8501

## 📚 Layer Details

### 1. Frontend Layer (Streamlit)

**File:** `ML_Streamlit_Loan_API.py`

**Responsibilities:**
- User interface for loan applications
- Input forms and validation
- Data visualization (charts, gauges)
- API communication
- Visual feedback (balloons, animations)

**Key Features:**
- Real-time prediction interface
- Prediction history viewer
- Statistics dashboard
- API connection status

### 2. Backend Layer (FastAPI)

**File:** `backend_api.py`

**Responsibilities:**
- REST API endpoints
- Machine learning model training
- Prediction logic
- Business rules
- Request/Response handling

**API Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| POST | `/train` | Train/retrain the model |
| GET | `/model-info` | Get model information |
| POST | `/predict` | Make a prediction |
| GET | `/predictions` | Get all predictions |
| GET | `/predictions/{id}` | Get specific prediction |
| GET | `/statistics` | Get prediction statistics |
| DELETE | `/predictions/{id}` | Delete a prediction |

**Interactive API Docs:** http://localhost:8000/docs

### 3. Database Layer (SQLite)

**File:** `db_layer.py`

**Responsibilities:**
- Database connection management
- CRUD operations
- Data persistence
- Query execution

**Database Schema:**

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    married TEXT,
    dependents TEXT,
    education TEXT,
    self_employed TEXT,
    property_area TEXT,
    applicant_income REAL,
    coapplicant_income REAL,
    loan_amount REAL,
    loan_term INTEGER,
    credit_history INTEGER,
    prediction TEXT,
    confidence REAL
)
```

## 🎯 Features

✅ **Separation of Concerns** - Each layer has distinct responsibilities  
✅ **Scalability** - Layers can be scaled independently  
✅ **Maintainability** - Easy to update individual components  
✅ **RESTful API** - Standard HTTP/REST communication  
✅ **Interactive Documentation** - Auto-generated API docs  
✅ **Data Persistence** - All predictions saved to database  
✅ **Visual Feedback** - Balloons for approvals, heartbreak for rejections  
✅ **Statistics Dashboard** - Track approval rates and trends  

## 🔧 Technical Stack

- **Frontend:** Streamlit, Plotly, Pandas
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Database:** SQLite3, Python DB-API
- **ML:** scikit-learn, NumPy, Pandas
- **HTTP Client:** Requests

## 📊 Machine Learning Model

- **Algorithm:** Logistic Regression
- **Features:** 
  - Applicant Income
  - Coapplicant Income
  - Loan Amount
  - Loan Term
  - Credit History
  - Education Level
  - Employment Status
  - Property Area
  - Number of Dependents

- **Preprocessing:**
  - Log transformation for income features
  - Standard scaling for numerical features
  - One-hot encoding for categorical features
  - Missing value imputation

## 🧪 Testing the API

You can test the API using curl, Postman, or the interactive docs:

**Example: Make a Prediction**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "married": "Yes",
    "dependents": "2",
    "education": "Graduate",
    "self_employed": "No",
    "property_area": "Urban",
    "applicant_income": 5000,
    "coapplicant_income": 2000,
    "loan_amount": 150,
    "loan_term": 360,
    "credit_history": 1
  }'
```

**Example: Get Statistics**

```bash
curl http://localhost:8000/statistics
```

## 📈 Benefits of 3-Tier Architecture

1. **Modularity:** Each layer can be developed and tested independently
2. **Scalability:** Backend API can handle multiple frontend clients
3. **Security:** Business logic and data access are separated from UI
4. **Flexibility:** Easy to swap out components (e.g., change database)
5. **Reusability:** API can be used by web, mobile, or other applications
6. **Testability:** Each layer can be unit tested separately

## 🔒 Future Enhancements

- Add authentication and authorization
- Implement rate limiting
- Add caching layer (Redis)
- Deploy to cloud (AWS, Azure, GCP)
- Add model versioning
- Implement A/B testing
- Add monitoring and logging
- Create mobile app using same API

## 🐛 Troubleshooting

**Backend not starting?**
- Check if port 8000 is already in use
- Ensure all dependencies are installed: `pip install fastapi uvicorn pydantic`

**Frontend shows "Backend API not running"?**
- Make sure backend is started first
- Check if backend is accessible at http://localhost:8000

**Database errors?**
- Check file permissions for `loan_predictions.db`
- Delete the database file to recreate: `del loan_predictions.db`

**Model not trained?**
- Ensure `Loan_dataset.csv` exists in the same directory
- Call the `/train` endpoint manually

## 📞 Support

For issues or questions:
1. Check the API documentation at http://localhost:8000/docs
2. Review the terminal logs for error messages
3. Check that all three files are in the same directory

## 📄 License

This project is for educational purposes.

---

**Created with ❤️ using Streamlit, FastAPI, and scikit-learn**
