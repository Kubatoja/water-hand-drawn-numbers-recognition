# Overnight run of all revision experiments (sequential; timing benchmark last, on an idle machine).
# Usage: powershell -ExecutionPolicy Bypass -File revision\run_all.ps1
# All scripts are resumable: re-running this file continues where it stopped.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "results\final\revision\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$status = Join-Path $logs "status.txt"
Set-Location $PSScriptRoot

function Step($name, $script, $argList) {
    Add-Content $status "$(Get-Date -Format s) START $name"
    $log = Join-Path $logs "$name.log"
    cmd /c "`"$py`" -u $script $($argList -join ' ') >> `"$log`" 2>&1"
    Add-Content $status "$(Get-Date -Format s) END   $name (exit $LASTEXITCODE)"
}

# (completed 2026-10-13) Step "ablation" "ablation.py" @("MNIST", "Fashion-MNIST", "USPS")
Step "emnist" "rerun_emnist.py" @()
Step "mnist_lgbm_check" "rerun_emnist.py" @("--mnist-lgbm-check")
Step "timing" "timing_benchmark.py" @()
Add-Content $status "$(Get-Date -Format s) ALL DONE"
