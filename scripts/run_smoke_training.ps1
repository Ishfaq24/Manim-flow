param(
    [string]$Python = "python",
    [string]$VenvPath = ".venv-train",
    [string]$Config = "configs/smoke_lora.json"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating local training environment..."
& $Python -m venv $VenvPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing training dependencies..."
& $VenvPython -m pip install -r requirements-train.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Running LoRA smoke training..."
& $VenvPython scripts\train_lora_smoke.py --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done. Adapter output: models/manim-smoke-lora"
