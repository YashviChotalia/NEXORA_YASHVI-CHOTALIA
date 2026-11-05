# Anvesh one-time setup for Windows. Writes a full log to logs\setup.log
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
$Log = Join-Path $Root "logs\setup.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null
"=== Anvesh setup $(Get-Date -Format s) ===" | Out-File $Log -Encoding utf8
function Log($msg) { Write-Host $msg; $msg | Out-File $Log -Append -Encoding utf8 }
function Run($label, $exe, [string[]]$argList) {
  Log ">>> $label"
  & $exe @argList 2>&1 | ForEach-Object { $line = "$_"; $line | Out-File $Log -Append -Encoding utf8; Write-Host $line }
  $code = $LASTEXITCODE
  Log "<<< $label exit=$code"
  return $code
}
$env:PYTHONUTF8 = "1"

# 1) Pick a supported Python (3.10-3.12 preferred)
$py = $null; $pyArgs = @()
foreach ($v in @("3.12", "3.11", "3.10")) {
  try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = "py"; $pyArgs = @("-$v"); break } } catch {}
}
if (-not $py) { if (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" } }
if (-not $py) { Log "ERROR: Python not found. Install Python 3.11 from python.org and tick 'Add python.exe to PATH'."; exit 1 }
$ver = & $py @pyArgs -c "import sys; print('%d.%d' % sys.version_info[:2])"
Log "Python: $py $pyArgs -> $ver"
if (Get-Command node -ErrorAction SilentlyContinue) { Log "Node: $(node --version)" } else { Log "ERROR: Node.js not found. Install Node 20 LTS from nodejs.org."; exit 1 }

# 2) Backend venv + core deps (required)
$Backend = Join-Path $Root "backend"
Set-Location $Backend
if (-not (Test-Path ".venv\Scripts\python.exe")) { Run "create venv" $py ($pyArgs + @("-m", "venv", ".venv")) | Out-Null }
$vpy = Join-Path $Backend ".venv\Scripts\python.exe"
Run "upgrade pip" $vpy @("-m", "pip", "install", "--upgrade", "pip") | Out-Null
$core = Run "install core requirements" $vpy @("-m", "pip", "install", "-r", "requirements-dev.txt")
if ($core -ne 0) { Log "ERROR: core install failed. See logs\setup.log for details."; exit 1 }

# 3) Speech models (optional; failure is logged, app still runs)
$ml = Run "install speech-model requirements" $vpy @("-m", "pip", "install", "-r", "requirements-ml.txt")
if ($ml -ne 0) { Log "WARNING: speech-model install failed; the app runs without live ASR. See log above." }

# 4) Self-test + demo data
Run "backend tests" $vpy @("-m", "pytest", "-q") | Out-Null
Run "load training scenarios" $vpy @("-m", "anvesh.seed", "--reset") | Out-Null
Run "check speech model imports" $vpy @("-c", "import faster_whisper; print('faster-whisper OK')") | Out-Null
if ($ml -eq 0) {
  $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
  $dl = Run "download speech models (first run only, several GB)" $vpy @("-m", "anvesh.download_models")
  if ($dl -ne 0) { Log "WARNING: some models did not download; they will retry when the backend starts." }
}

# 5) Frontend
$Frontend = Join-Path $Root "frontend"
Set-Location $Frontend
if (-not (Test-Path ".env.local")) { Copy-Item ".env.example" ".env.local" }
$npm = Run "npm install" "npm.cmd" @("install", "--no-audit", "--no-fund")
if ($npm -ne 0) { Log "ERROR: npm install failed. See logs\setup.log for details."; exit 1 }
Log "SETUP COMPLETE. Now double-click start.bat"
