@echo off
echo ====================================
echo House Sales Prediction System Setup
echo ====================================
echo.

cd /d "%~dp0"

echo Step 1: Training ML Models...
echo.
python train_models.py
if errorlevel 1 (
    echo Error training models!
    pause
    exit /b 1
)

echo.
echo ====================================
echo Models trained successfully!
echo ====================================
echo.

echo Step 2: Starting API Server...
echo.
echo API will be available at: http://localhost:8000
echo Web Interface: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python house_sales_api.py
