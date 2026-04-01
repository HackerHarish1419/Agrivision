@echo off
color 0A
echo ===================================================
echo     AgriVisionAI - Hackathon Setup ^& Launcher
echo ===================================================
echo.

:: 0. Resolve dynamic paths
set "AGRIVISION_ROOT=%~dp0"
set "PT_MODEL_PATH=%~dp0training\agrivision_efficientnet_b0.pt"
set "ONNX_MODEL_PATH=%~dp0models\agrivision_efficientnet_b0.onnx"
echo [SYS] Root: %AGRIVISION_ROOT%
echo.

:: 1. First-time .env setup
if exist ".env" goto skip_env
python setup_env.py
if errorlevel 1 (
    echo [ERROR] Setup failed. Please create .env manually.
    pause
    exit /b 1
)
:skip_env

:: 2. Python virtual environment
echo [1/4] Checking Python Virtual Environment...
if exist "venv\Scripts\activate.bat" goto activate_venv
echo Creating fresh Python virtual environment...
python -m venv venv
:activate_venv
call venv\Scripts\activate.bat
echo.

:: 3. Install backend dependencies (only once)
echo [2/4] Initializing Backend Dependencies...
if exist "venv\.installed" goto skip_pip
echo Installing Python packages (first time only, ~2 min)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install onnxruntime opencv-python-headless --quiet
type nul > "venv\.installed"
echo Backend packages installed!
:skip_pip
echo Backend dependencies ready.
echo.

:: 4. Install frontend node modules (only once)
echo [3/4] Initializing Frontend...
if exist "frontend\node_modules\vite" goto skip_npm
echo Installing npm packages (first time only)...
cd frontend
call npm install
cd ..
:skip_npm
echo Frontend ready.
echo.

:: 5. Launch both services
echo [4/4] Launching AgriVisionAI...

set "BACKEND_CMD=cd /d %AGRIVISION_ROOT% & call venv\Scripts\activate & set PT_MODEL_PATH=%PT_MODEL_PATH% & set ONNX_MODEL_PATH=%ONNX_MODEL_PATH% & python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
set "FRONTEND_CMD=cd /d %AGRIVISION_ROOT%frontend & npm run dev -- --host"

start "AgriVisionAI Backend" cmd /k "%BACKEND_CMD%"
timeout /t 2 /nobreak >nul
start "AgriVisionAI Frontend" cmd /k "%FRONTEND_CMD%"

echo.
echo ===================================================
echo   SUCCESS! AgriVisionAI is booting up.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo ===================================================
pause
