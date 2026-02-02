@echo off
echo Starting Spotify Cluster Prediction API...
echo.
echo Make sure you have installed the required packages:
echo   pip install flask flask-cors scikit-learn pandas numpy
echo.
cd /d "%~dp0"
python spotify_api.py
pause
