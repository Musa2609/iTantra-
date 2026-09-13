@echo off
setlocal enabledelayedexpansion
title iTantra - Automated One-Time Setup
echo ===================================================
echo             iTantra One-Time Setup
echo ===================================================
echo.

:: Check specifically for Python 3.12 (PyTorch/ONNX does NOT support Python 3.14 or 3.13)
set "PYTHON_CMD="

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.12"
    echo [OK] Found Python 3.12 via Python Launcher
    goto :SETUP_VENV
)

:: Check if standard 'python' command happens to be 3.12
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    echo [OK] Found Python 3.12 in PATH
    goto :SETUP_VENV
)

:: Python 3.12 was NOT found!
echo ===================================================
echo [!] Python 3.12 is REQUIRED for AI4Bharat/PyTorch.
echo     (Python 3.13 and 3.14 are not supported yet)
echo ===================================================
echo.
echo Downloading official Python 3.12 installer directly...
echo Please wait 10-20 seconds...
echo.

curl -L -o "%TEMP%\python-3.12.8-amd64.exe" "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Could not download Python 3.12.
    echo Please manually download and install Python 3.12 from:
    echo   https://www.python.org/downloads/release/python-3128/
    echo IMPORTANT: Check 'Add python.exe to PATH' when installing!
    pause
    exit /b 1
)

echo Installing Python 3.12...
"%TEMP%\python-3.12.8-amd64.exe" /passive PrependPath=1 Include_pip=1
del "%TEMP%\python-3.12.8-amd64.exe" >nul 2>&1

:: Check if py -3.12 is now available
py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.12"
    echo [OK] Python 3.12 installed and verified!
    goto :SETUP_VENV
)

:: Check common install path if launcher hasn't reloaded PATH
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD="%LocalAppData%\Programs\Python\Python312\python.exe""
    echo [OK] Found Python 3.12 at %LocalAppData%\Programs\Python\Python312
    goto :SETUP_VENV
)

echo [OK] Python 3.12 has been installed.
echo Please CLOSE this window and double-click setup.bat again so Windows loads the new PATH!
pause
exit /b 0

:SETUP_VENV
echo.
:: 2. Create voice-env virtual environment
if exist "voice-env\Scripts\python.exe" (
    echo [SKIP] Virtual environment 'voice-env' already exists.
) else (
    echo [...] Creating virtual environment 'voice-env' using %PYTHON_CMD%...
    if exist "voice-env" rd /s /q "voice-env"
    %PYTHON_CMD% -m venv voice-env
    if not exist "voice-env\Scripts\python.exe" (
        echo [ERROR] Failed to create virtual environment.
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
