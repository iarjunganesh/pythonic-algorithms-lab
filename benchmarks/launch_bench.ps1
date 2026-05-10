# launch_bench.ps1 — Safe benchmark launcher
# Usage:
#   .\benchmarks\launch_bench.ps1
#   .\benchmarks\launch_bench.ps1 -Sizes "1000 5000 10000 50000 100000" -Repeat 5
#
# - Checks for an existing lock file (run_benchmarks is already running)
# - Kills any stale run_benchmarks Python process before starting
# - Launches a single hidden process with timestamped log files
# - Prints the PID so you can track it

param(
    [string]$Sizes  = "1000 5000 10000 50000 100000",
    [int]   $Repeat = 5,
    [string]$Root   = (Split-Path $PSScriptRoot -Parent)
)

$python   = "$Root\.venv\Scripts\python.exe"
$script   = "benchmarks\run_benchmarks.py"
$lockFile = "$Root\benchmarks\.run_benchmarks.lock"
$ts       = (Get-Date -Format 'yyyyMMdd-HHmmss')
$outCsv   = "benchmarks\results_$ts.csv"
$outLog   = "benchmarks\bench_$ts.log"
$errLog   = "benchmarks\bench_err_$ts.log"

# --- Check lock file ---
if (Test-Path $lockFile) {
    $lockedPid = (Get-Content $lockFile -ErrorAction SilentlyContinue).Trim()
    $alive = $false
    if ($lockedPid -match '^\d+$') {
        $alive = [bool](Get-Process -Id $lockedPid -ErrorAction SilentlyContinue)
    }
    if ($alive) {
        Write-Host "[launch_bench] Benchmark already running (PID $lockedPid, lock: $lockFile). Aborting." -ForegroundColor Yellow
        Write-Host "  To force a fresh start, run: Stop-Process -Id $lockedPid -Force; Remove-Item '$lockFile'"
        exit 1
    }
    Write-Host "[launch_bench] Stale lock found (PID $lockedPid gone). Removing." -ForegroundColor Cyan
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}

# --- Kill any orphaned run_benchmarks processes ---
$orphans = Get-CimInstance Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%run_benchmarks%'" |
           Select-Object -ExpandProperty ProcessId
if ($orphans) {
    Write-Host "[launch_bench] Killing orphaned benchmark PIDs: $orphans" -ForegroundColor Cyan
    $orphans | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Milliseconds 600
}

# --- Launch ---
$sizeArray = $Sizes -split ' '
$argList   = @($script, "--full") + @("--sizes") + $sizeArray + @("--repeat", $Repeat, "--out", $outCsv)

$proc = Start-Process `
    -FilePath   $python `
    -ArgumentList $argList `
    -WorkingDirectory $Root `
    -RedirectStandardOutput "$Root\$outLog" `
    -RedirectStandardError  "$Root\$errLog" `
    -WindowStyle Hidden `
    -PassThru

Write-Host ""
Write-Host "[launch_bench] Benchmark started:" -ForegroundColor Green
Write-Host "  PID    : $($proc.Id)"
Write-Host "  Sizes  : $Sizes"
Write-Host "  Repeat : $Repeat"
Write-Host "  CSV    : $outCsv"
Write-Host "  Log    : $outLog"
Write-Host "  ErrLog : $errLog"
Write-Host ""
Write-Host "Check status:"
Write-Host "  Get-Process -Id $($proc.Id) -ErrorAction SilentlyContinue"
Write-Host "  Get-Content '$Root\$outLog' -Tail 5"
Write-Host "  Test-Path '$Root\$outCsv'"
