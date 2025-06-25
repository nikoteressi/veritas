@echo off
echo Starting Veritas Backend...
cd backend
call venv\Scripts\activate.bat
echo Backend virtual environment activated
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
