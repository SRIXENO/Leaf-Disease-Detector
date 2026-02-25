@echo off
setlocal

REM Dependency installer for LeafDiseaseProject (Windows)
cd /d "%~dp0"

echo Checking Python...
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

echo.
echo Upgrading pip...
%PY_CMD% -m pip install --upgrade pip
if %errorlevel% neq 0 (
  echo [ERROR] Failed to upgrade pip.
  pause
  exit /b 1
)

echo.
echo Installing project dependencies...
%PY_CMD% -m pip install fastapi uvicorn tensorflow keras numpy opencv-python pillow jinja2 python-multipart python-dotenv openai deep-translator reportlab requests pandas
if %errorlevel% neq 0 (
  echo [ERROR] Dependency installation failed.
  pause
  exit /b 1
)

echo.
echo [OK] All dependencies installed successfully.
echo Next step: run start_project.bat
pause

