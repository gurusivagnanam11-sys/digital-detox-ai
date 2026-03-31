@echo off
REM run_project.bat — activate venv, init DB, start backend and frontend

cd /d %~dp0

IF NOT EXIST venv\Scripts\activate.bat (
  echo Virtual environment not found at "%CD%\venv\Scripts\activate.bat".
  echo Create one with: python -m venv venv
  echo Then install dependencies: venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Initializing database (SQLite)...
venv\Scripts\python.exe -c "from database import init_db; init_db()" || echo DB init failed

echo Starting backend (FastAPI + Uvicorn)...
start "DigitalDetox Backend" cmd /k "venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

echo Starting frontend (static server on port 3000)...
start "DigitalDetox Frontend" cmd /k "venv\Scripts\python.exe -m http.server 3000"

echo Launcher started both services. Close this window to stop the launcher (services run in separate windows).
pause
