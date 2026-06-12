@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment was not found.
    echo Run install_requirements.bat first.
    exit /b 1
)

".venv\Scripts\python.exe" -m image_aug_cli.cli %*

endlocal
