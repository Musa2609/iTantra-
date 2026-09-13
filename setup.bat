@echo off
setlocal enabledelayedexpansion
title iTantra - Automated One-Time Setup
echo ===================================================
echo             iTantra One-Time Setup
echo ===================================================
echo.

:: 1. Check Python
set "PYTHON_EXE="

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.12"
    echo [OK] Found Python 3.12 via Python Launcher (py -3.12)
    goto :PYTHON_FOUND
)

python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    echo [INFO] Found default Python in PATH
    goto :PYTHON_FOUND
)

py --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py"
    echo [INFO] Found Python Launcher (py)
    goto :PYTHON_FOUND
)

:: If not found, offer winget install
echo [WARNING] Python 3.12 was not found on this machine.
echo Attempting automatic installation of Python 3.12 via winget...
echo.
winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
if %ERRORLEVEL% EQU 0 (
    echo [OK] Python 3.12 installed successfully!
    echo Please CLOSE this window and run setup.bat again to refresh environment PATH.
    pause
    exit /b 0
) else (
    echo [ERROR] Automatic installation failed.
    echo Please download and install Python 3.12 manually:
    echo   https://www.python.org/downloads/release/python-3128/
    echo IMPORTANT: Make sure to check "Add python.exe to PATH" during installation!
    pause
    exit /b 1
)

:PYTHON_FOUND
echo.

:: 2. Create voice-env virtual environment
if exist "voice-env\Scripts\python.exe" (
    echo [SKIP] Virtual environment 'voice-env' already exists.
) else (
    echo [...] Creating virtual environment 'voice-env'...
    if exist "voice-env" rd /s /q "voice-env"
    %PYTHON_CMD% -m venv voice-env
    if not exist "voice-env\Scripts\python.exe" (
        echo [ERROR] Failed to create virtual environment.
        echo Please ensure Python 3.12 is installed with PATH enabled.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
)
echo.

:: 3. Upgrade pip and install requirements
echo [...] Upgrading pip...
".\voice-env\Scripts\python.exe" -m pip install --upgrade pip --quiet

echo [...] Installing AI, ASR, TTS, and acoustic packages...
echo       (This takes 3-7 minutes on first run, please wait...)
echo.
".\voice-env\Scripts\python.exe" -m pip install -r requirements-voice.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Package installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo.
echo [OK] All packages installed successfully.
echo.

:: 4. Check Hugging Face Login
if exist "%USERPROFILE%\.cache\huggingface\token" (
    echo [OK] Hugging Face authentication token found.
) else (
    echo ===================================================
    echo           HUGGING FACE LOGIN REQUIRED
    echo ===================================================
    echo AI4Bharat models require a free Hugging Face account.
    echo 1. Make sure you clicked "Request access" on:
    echo    https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
    echo    https://huggingface.co/ai4bharat/indic-parler-tts
    echo 2. Get your token at: https://huggingface.co/settings/tokens
    echo.
    ".\voice-env\Scripts\python.exe" -m huggingface_hub.commands.huggingface_cli login
)

echo.
echo ===================================================
echo                SETUP COMPLETE!
echo ===================================================
echo You can now run:
echo   - run_sender.bat    (To record speech and transmit acoustic tones)
echo   - run_receiver.bat  (To listen for tones and speak received message)
echo.
pause
