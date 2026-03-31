@echo off
REM Digital Detox AI Launcher

echo ====================================
echo Starting Digital Detox AI System
echo ====================================

cd /d %~dp0

REM -------------------------------
REM Check Virtual Environment
REM -------------------------------
IF NOT EXIST venv\Scripts\activate.bat (
  echo Virtual environment not found.
  echo Run these commands first:
  echo python -m venv venv
  echo venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

REM -------------------------------
REM Start Ollama Server
REM -------------------------------
echo.
echo Starting Ollama server...
start "Ollama Server" cmd /k "ollama serve"

timeout /t 3 > nul

echo Loading Ollama model (phi3)...
start "Ollama Model" cmd /k "ollama run phi3"

timeout /t 5 > nul

REM -------------------------------
REM Activate Python Environment
REM -------------------------------
echo.
echo Activating Python virtual environment...
call venv\Scripts\activate.bat

REM -------------------------------
REM Initialize Database
REM -------------------------------
echo.
echo Initializing database...
venv\Scripts\python.exe -c "from database import init_db; init_db()" || echo DB init failed

REM -------------------------------
REM Start Backend API
REM -------------------------------
echo.
echo Starting FastAPI backend...
start "DigitalDetox Backend" cmd /k "venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

timeout /t 3 > nul

REM -------------------------------
REM Start Frontend Server
REM -------------------------------
echo.
echo Starting frontend server...
start "DigitalDetox Frontend" cmd /k "venv\Scripts\python.exe -m http.server 3000"

timeout /t 3 > nul

REM -------------------------------
REM Open UI in browser
REM -------------------------------
start http://localhost:3000

echo.
echo ====================================
echo System Started Successfully
echo ====================================
echo Backend API: http://127.0.0.1:8000
echo API Docs:   http://127.0.0.1:8000/docs
echo Frontend:   http://localhost:3000
echo ====================================

pause
