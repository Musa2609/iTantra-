@echo off
setlocal enabledelayedexpansion
title iTantra - Acoustic Speech Receiver
echo ===================================================
echo            iTantra - RECEIVER MODE
echo ===================================================
echo.

:: Check environment
if not exist "voice-env\Scripts\python.exe" (
    echo [!] Environment not found. Running setup first...
    call setup.bat
    if not exist "voice-env\Scripts\python.exe" exit /b 1
)

echo Place this laptop's microphone close to the Sender's speakers.
echo.
set /p "WINDOW=Listening window in seconds [default: 20]: "
if "%WINDOW%"=="" set "WINDOW=20"

echo.
echo ===================================================
echo Receiver will listen for %WINDOW% seconds.
echo Please START playback on the Sender laptop now!
echo ===================================================
echo.

".\voice-env\Scripts\python.exe" -u test_asr_tts_ggwave.py --physical-receive --listen-duration %WINDOW% --tts-output received_speech.wav

echo.
echo ===================================================
echo Session complete.
echo ===================================================
pause
