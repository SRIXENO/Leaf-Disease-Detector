@echo off
setlocal

REM Start script for LeafDiseaseProject (Windows)
cd /d "%~dp0"

if not exist "dataset\app.py" (
  echo [ERROR] Could not find dataset\app.py
  echo Run this file from the project root folder.
  pause
  exit /b 1
)

REM Prefer Python launcher, fallback to python
set "PY_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY_CMD=python"
  ) else (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
  )
)

echo Starting Leaf Disease Project...
echo App URL: http://127.0.0.1:8080
echo.

cd /d "dataset"
%PY_CMD% app.py

echo.
echo App stopped.
pause

