# iTantra Setup Script — Run this once on every new device
# Usage: .\setup.ps1

Write-Host ""
Write-Host "=== iTantra One-Time Setup ===" -ForegroundColor Cyan
Write-Host ""

# Find a valid Python executable
$pythonCmd = $null

# 1. Try py -3.12
$testPy312 = & py -3.12 -c "import sys; print(sys.version)" 2>
if ($LASTEXITCODE -eq 0) {
    $pythonCmd = "py -3.12"
    Write-Host "[OK] Found Python 3.12 via py launcher" -ForegroundColor Green
}

# 2. Try python command directly
if (-not $pythonCmd) {
    $testPython = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Detected python version: $testPython" -ForegroundColor Yellow
        $pythonCmd = "python"
    }
}

# 3. Try py command generic
if (-not $pythonCmd) {
    $testPy = & py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Detected py launcher version: $testPy" -ForegroundColor Yellow
        $pythonCmd = "py"
    }
}

# If no Python found at all
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not found on this machine." -ForegroundColor Red
    Write-Host ""
    Write-Host "Installing Python 3.12 automatically via winget..." -ForegroundColor Cyan
    try {
        winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
        Write-Host "[OK] Python 3.12 installed! Please RESTART PowerShell and re-run .\setup.ps1" -ForegroundColor Green
    } catch {
        Write-Host "Please download and install Python 3.12 manually from:" -ForegroundColor Yellow
        Write-Host "https://www.python.org/downloads/release/python-3128/" -ForegroundColor White
        Write-Host "(Make sure to check 'Add python.exe to PATH' during installation!)" -ForegroundColor Yellow
    }
    exit 1
}

# Create virtual environment if not exists
if (Test-Path "voice-env\Scripts\python.exe") {
    Write-Host "[SKIP] voice-env already exists." -ForegroundColor Yellow
} else {
    Write-Host "[...] Creating virtual environment 'voice-env' using $pythonCmd..." -ForegroundColor Cyan
    if (Test-Path "voice-env") { Remove-Item -Recurse -Force "voice-env" }
    
    if ($pythonCmd -eq "py -3.12") {
        & py -3.12 -m venv voice-env
    } else {
        Invoke-Expression "$pythonCmd -m venv voice-env"
    }

    if (-not (Test-Path "voice-env\Scripts\python.exe")) {
        Write-Host "[ERROR] Failed to create voice-env." -ForegroundColor Red
        Write-Host "Please install Python 3.12 from: https://www.python.org/downloads/release/python-3128/" -ForegroundColor Yellow
        Write-Host "(Check 'Add python.exe to PATH' when installing!)" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[OK] voice-env created successfully." -ForegroundColor Green
}

# Upgrade pip
Write-Host "[...] Upgrading pip..." -ForegroundColor Cyan
& ".\voice-env\Scripts\python.exe" -m pip install --upgrade pip --quiet

# Install requirements
Write-Host "[...] Installing packages from requirements-voice.txt..." -ForegroundColor Cyan
Write-Host "      (This may take a few minutes on first run)" -ForegroundColor Gray
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
