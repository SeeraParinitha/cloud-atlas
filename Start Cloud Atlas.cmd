@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The project Python environment is missing.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\run_local_web.py"
pause
