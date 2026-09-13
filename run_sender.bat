@echo off
setlocal enabledelayedexpansion
title iTantra - Laptop 1 (Sender Station)
echo ===================================================
echo        iTantra - LAPTOP 1 SENDER STATION
echo  Speech -^> Semantic -^> ggwave -^> Laptop 1 Speaker
echo ===================================================
echo.

set /p "SECS=Recording duration in seconds [default: 4]: "
if "%SECS%"=="" set "SECS=4"

echo.
echo ===================================================
echo Ready to transmit! Speak clearly into mic.
echo (Ensure Laptop 2 is running run_receiver.bat!)
echo ===================================================
echo.

python -u run_sender.py --record %SECS% --language hi

echo.
pause
