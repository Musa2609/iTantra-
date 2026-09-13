# iTantra Setup Script — Run this once on every new device
# Usage: Right-click in PowerShell -> Run, or:  .\setup.ps1

Write-Host ""
Write-Host "=== iTantra One-Time Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python 3.12
try {
    $pyver = & py -3.12 --version 2>&1
    Write-Host "[OK] Found: $pyver" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python 3.12 not found." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/release/python-3120/"
    exit 1
}

# Create virtual environment
if (Test-Path "voice-env") {
    Write-Host "[SKIP] voice-env already exists." -ForegroundColor Yellow
} else {
    Write-Host "[...] Creating voice-env..." -ForegroundColor Cyan
    py -3.12 -m venv voice-env
    Write-Host "[OK] voice-env created." -ForegroundColor Green
}

# Upgrade pip
Write-Host "[...] Upgrading pip..." -ForegroundColor Cyan
& ".\voice-env\Scripts\python.exe" -m pip install --upgrade pip --quiet

# Install requirements
Write-Host "[...] Installing packages from requirements-voice.txt..." -ForegroundColor Cyan
Write-Host "      (This may take 5-10 minutes on first run)" -ForegroundColor Gray
& ".\voice-env\Scripts\python.exe" -m pip install -r requirements-voice.txt

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next step: Log in to Hugging Face (only needed once):" -ForegroundColor Cyan
Write-Host '  .\voice-env\Scripts\python.exe -m huggingface_hub.commands.huggingface_cli login'
Write-Host ""
Write-Host "Then to run as RECEIVER:" -ForegroundColor Cyan
Write-Host '  .\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-receive --tts-output received_speech.wav'
Write-Host ""
Write-Host "Then to run as SENDER:" -ForegroundColor Cyan
Write-Host '  .\voice-env\Scripts\python.exe -u test_asr_tts_ggwave.py --physical-send --record 5 --auto-detect --target-language eng_Latn'
Write-Host ""
