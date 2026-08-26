@echo off
title Titanic Survival Prediction Web App
color 0A
echo ============================================================
echo   🚢 Titanic Survival Prediction AI - 1-Click Startup
echo ============================================================
echo.

if not exist .venv (
    echo [!] Virtual environment not found. Creating .venv...
    py -m venv .venv
    echo [+] Installing requirements...
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
)

if not exist models\titanic_pipeline.pkl (
    echo [!] Trained model pipeline not found. Training model...
    .\.venv\Scripts\python.exe src/train.py
)

echo.
echo [+] Launching Flask Web Application...
echo [+] Web app will automatically open in your default browser at http://127.0.0.1:5000/
echo.
.\.venv\Scripts\python.exe app/app.py

pause
