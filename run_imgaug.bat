@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment was not found.
    echo Run install_requirements.bat first.
    exit /b 1
)

if "%~1"=="" (
    echo Usage:
    echo   run_imgaug.bat --input C:\datasets\raw --output C:\datasets\augmented --config configs\example.yaml
    echo.
    echo Example:
    echo   run_imgaug.bat --input samples\raw --output samples\augmented --config configs\example.yaml
    exit /b 0
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1

call "%~dp0imgaug.bat" %*

endlocal
