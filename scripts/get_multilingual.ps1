# Anvesh: download multilingual speech models and emotion data from Hugging Face.
# Run from the anvesh folder:  powershell -ExecutionPolicy Bypass -File scripts\get_multilingual.ps1
# Before running: sign in on huggingface.co and accept the terms on
#   huggingface.co/ai4bharat/indic-conformer-600m-multilingual
#   huggingface.co/datasets/ai4bharat/Rasa
#   huggingface.co/datasets/amu-cai/CAMEO
$ErrorActionPreference = "Continue"   # progress bars go to stderr; PowerShell 5 must not treat them as errors
$backend = Join-Path $PSScriptRoot "..\backend"
Set-Location $backend
if (-not (Test-Path ".venv\Scripts\python.exe")) { Write-Host "backend\.venv not found: run setup.bat first."; exit 1 }
$py = ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path "..\logs" | Out-Null
Write-Host "Installing download tools ..."
& $py -m pip install -q -U "huggingface_hub>=0.25" "datasets>=2.20" soundfile
Write-Host "Starting downloads (you will be asked for your token once if you are not signed in) ..."
& $py eval\get_multilingual_audio.py @args   # the script writes logs\multilingual.log itself
Write-Host ""
Write-Host "Log saved to logs\multilingual.log. Done."
