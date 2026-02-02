@echo off
echo Starting Spotify Cluster Prediction Streamlit App...
echo.
echo Make sure you have installed streamlit:
echo   pip install streamlit
echo.
cd /d "%~dp0"
streamlit run spotify_streamlit.py
pause
