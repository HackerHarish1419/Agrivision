@echo off
color 0A
echo ===================================================
echo     AgriVisionAI - Hackathon Setup ^& Launcher
echo ===================================================
echo.

:: 1. Backend Setup
echo [1/4] Checking Python Virtual Environment...
if not exist "venv\Scripts\activate.bat" (
    echo Creating fresh Python virtual environment...
    python -m venv venv
)

echo [2/4] Installing Backend Dependencies...
call venv\Scripts\activate.bat
echo Installing requirements (this might take a minute)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install onnxruntime opencv-python-headless --quiet
echo Backend Ready!
echo.

:: 2. Frontend Setup
echo [3/4] Installing Frontend Node Modules...
cd frontend
call npm install
echo Frontend Ready!
echo.

:: 3. Launching
echo [4/4] Launching Systems...
echo Starting FastAPI Backend in a new window...
start "AgriVisionAI Backend" cmd /k "cd .. & venv\Scripts\activate & python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting React Frontend in a new window...
start "AgriVisionAI Frontend" cmd /k "npm run dev"

echo.
echo ===================================================
echo SUCCESS! System is booting.
echo - Backend will be available at: http://localhost:8000
echo - Frontend will be available at: http://localhost:5173
echo ===================================================
pause
