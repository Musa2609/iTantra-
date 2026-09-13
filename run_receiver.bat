@echo off
setlocal enabledelayedexpansion
title iTantra - Laptop 2 (Receiver Station)
echo ===================================================
echo       iTantra - LAPTOP 2 RECEIVER STATION
echo  Mic -^> ggwave decode -^> Semantic -^> TTS Speaker
echo ===================================================
echo.
echo Place Laptop 2 microphone close to Laptop 1 speakers.
echo.

set /p "WINDOW=Listening window in seconds [default: 12]: "
if "%WINDOW%"=="" set "WINDOW=12"

echo.
echo ===================================================
echo Receiver listening for %WINDOW% seconds.
echo Please START transmission on Laptop 1 now!
echo ===================================================
echo.

python -u run_receiver.py --listen-duration %WINDOW% --language hi

echo.
pause
