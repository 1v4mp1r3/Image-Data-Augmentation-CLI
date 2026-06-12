@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PY_CMD="

where py >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=py -3"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo Python 3.10+ was not found. Install Python and add it to PATH.
    exit /b 1
)

%PY_CMD% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo Python 3.10+ is required.
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR%...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 exit /b 1

python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

python -m pip install -e .
if errorlevel 1 exit /b 1

echo.
echo Dependencies installed successfully.
echo Run the tool with run_imgaug.bat

endlocal
