@echo off
setlocal enabledelayedexpansion
title iTantra - Acoustic Speech Sender
echo ===================================================
echo             iTantra - SENDER MODE
echo ===================================================
echo.

:: Check environment
if not exist "voice-env\Scripts\python.exe" (
    echo [!] Environment not found. Running setup first...
    call setup.bat
    if not exist "voice-env\Scripts\python.exe" exit /b 1
)

echo Select how you want to send your message:
echo.
echo   [1] Speak any Indian language -^> Translate to English (eng_Latn)  [DEFAULT]
echo   [2] Speak any Indian language -^> Translate to Hindi   (hin_Deva)
echo   [3] Speak any Indian language -^> Translate to Gujarati (guj_Gujr)
echo   [4] Speak any Indian language -^> Translate to Tamil    (tam_Taml)
echo   [5] No translation (Send as spoken)
echo   [6] Custom target language tag
echo.
set /p "CHOICE=Enter choice [1-6, default: 1]: "
if "%CHOICE%"=="" set "CHOICE=1"

set "TARGET_ARG=--target-language eng_Latn"
if "%CHOICE%"=="1" set "TARGET_ARG=--target-language eng_Latn"
if "%CHOICE%"=="2" set "TARGET_ARG=--target-language hin_Deva"
if "%CHOICE%"=="3" set "TARGET_ARG=--target-language guj_Gujr"
if "%CHOICE%"=="4" set "TARGET_ARG=--target-language tam_Taml"
if "%CHOICE%"=="5" set "TARGET_ARG="
if "%CHOICE%"=="6" (
    set /p "CUSTOM_TAG=Enter target tag (e.g. tel_Telu, pan_Guru, mar_Deva): "
    set "TARGET_ARG=--target-language !CUSTOM_TAG!"
)

set /p "SECS=Recording duration in seconds [default: 5]: "
if "%SECS%"=="" set "SECS=5"

echo.
echo ===================================================
echo Ready to record! Please speak clearly into your mic.
echo (Make sure receiver laptop is already listening!)
echo ===================================================
echo.

".\voice-env\Scripts\python.exe" -u test_asr_tts_ggwave.py --physical-send --record %SECS% --auto-detect %TARGET_ARG%

echo.
echo ===================================================
echo Transmission finished.
echo ===================================================
pause
