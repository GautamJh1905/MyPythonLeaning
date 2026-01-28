@echo off
REM =============================================================================
REM Startup Script for 3-Tier Loan Prediction Application
REM =============================================================================

echo ========================================
echo Starting 3-Tier Loan Prediction System
echo ========================================
echo.

echo [1/2] Starting Backend API (FastAPI)...
echo Backend will run on: http://localhost:8000
echo API Docs available at: http://localhost:8000/docs
echo.

start "Backend API" cmd /k "c:/Users/JHAGAUT/Documents/LearnPythonJuly2025/Machine_Learning/.venv/Scripts/python.exe backend_api.py"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo [2/2] Starting Frontend (Streamlit)...
echo Frontend will run on: http://localhost:8501
echo.

start "Frontend" cmd /k "c:/Users/JHAGAUT/Documents/LearnPythonJuly2025/Machine_Learning/.venv/Scripts/python.exe -m streamlit run ML_Streamlit_Loan_API.py"

echo.
echo ========================================
echo System Started Successfully!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend UI: http://localhost:8501
echo.
echo Press any key to exit this window...
pause >nul
