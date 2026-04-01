@echo off
setlocal EnableDelayedExpansion
color 0A
echo ===================================================
echo     AgriVisionAI - Hackathon Setup ^& Launcher
echo ===================================================
echo ===================================================
echo.

:: 0. Environment Setup (Resolving dynamic paths)
set "AGRIVISION_ROOT=%~dp0"
set "PT_MODEL_PATH=%~dp0training\agrivision_efficientnet_b0.pt"
set "ONNX_MODEL_PATH=%~dp0models\agrivision_efficientnet_b0.onnx"
echo [SYS] Bound architecture dynamically to: %AGRIVISION_ROOT%
echo.

if not exist ".env" (
    echo ===================================================
    echo       [INITIAL SETUP DETECTED]
    echo ===================================================
    echo Providing a Groq API Key enables the AI Recovery Engine.
    echo You can get a free key at: https://console.groq.com/keys
    echo.
    
    set /p "USER_GROQ_KEY=Paste your GROQ_API_KEY: "
    set /p "USER_BLUR_THRESH=Set Blur Threshold (Press Enter for Default 100.0): "
    
    if "!USER_BLUR_THRESH!"=="" set "USER_BLUR_THRESH=100.0"

    echo.
    echo Generating .env configuration...
    (
        echo GROQ_API_KEY=!USER_GROQ_KEY!
        echo GROQ_MODEL=llama-3.3-70b-versatile
        echo MODEL_PATH=models/best_model.pth
        echo CLASS_NAMES_PATH=models/class_names.json
        echo BLUR_THRESHOLD=!USER_BLUR_THRESH!
        echo CONFIDENCE_THRESHOLD=0.60
    ) > .env
    
    echo [OK] .env file fully configured!
    echo ===================================================
    echo.
)

:: 1. Backend Setup
echo [1/4] Checking Python Virtual Environment...
if not exist "venv\Scripts\activate.bat" (
    echo Creating fresh Python virtual environment...
    python -m venv venv
)

echo [2/4] Initializing Backend Environment...
call venv\Scripts\activate.bat

if not exist "venv\.dependencies_installed" (
    echo Installing requirements (this might take a minute)...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
    pip install onnxruntime opencv-python-headless --quiet
    echo. > venv\.dependencies_installed
    echo Backend Installations Complete!
) else (
    echo Backend dependencies already installed! Skipping.
)
echo.

:: 2. Frontend Setup
echo [3/4] Initializing Frontend Node Modules...
if not exist "frontend\node_modules\" (
    echo Installing npm packages...
    cd frontend
    call npm install
    cd ..
    echo Frontend Installations Complete!
) else (
    echo Frontend modules already installed! Skipping.
)
echo.

:: 3. Launching
echo [4/4] Launching Systems...
echo Starting FastAPI Backend in a new window...
start "AgriVisionAI Backend" cmd /k "cd .. & venv\Scripts\activate & set IN_BAT=1 & set PT_MODEL_PATH=%PT_MODEL_PATH% & set ONNX_MODEL_PATH=%ONNX_MODEL_PATH% & python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting React Frontend in a new window...
start "AgriVisionAI Frontend" cmd /k "cd /d %~dp0frontend & npm run dev"

echo.
echo ===================================================
echo SUCCESS! System is booting.
echo - Backend will be available at: http://localhost:8000
echo - Frontend will be available at: http://localhost:5173
echo ===================================================
pause
