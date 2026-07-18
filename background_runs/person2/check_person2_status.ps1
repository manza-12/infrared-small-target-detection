$root = 'C:\Users\bbqsa\Desktop\红外检测\infrared_detection_code (1)'
Set-Location -LiteralPath $root
$bgDir = Join-Path $root 'background_runs\person2'
$pidFile = Join-Path $bgDir 'person2_pid.txt'
$stdout = Join-Path $bgDir 'person2_full_train_stdout.log'
$stderr = Join-Path $bgDir 'person2_full_train_stderr.log'
Write-Host '=== Person2 background status ==='
Write-Host ('Time: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Host ('Project: ' + $root)
if (Test-Path -LiteralPath $pidFile) {
    $pidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    Write-Host ('PID: ' + $pidText)
    $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host ('Process: RUNNING ' + $proc.ProcessName + ' CPU=' + $proc.CPU)
    } else {
        Write-Host 'Process: NOT RUNNING'
    }
} else {
    Write-Host 'PID file not found.'
}
Write-Host "`n=== stdout last 80 lines ==="
if (Test-Path -LiteralPath $stdout) { Get-Content -LiteralPath $stdout -Tail 80 } else { Write-Host 'stdout log not found.' }
Write-Host "`n=== stderr last 80 lines ==="
if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Tail 80 } else { Write-Host 'stderr log not found.' }
Write-Host "`n=== latest 10 result directories ==="
$resultDir = Join-Path $root 'result'
if (Test-Path -LiteralPath $resultDir) {
    Get-ChildItem -LiteralPath $resultDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 10 FullName, LastWriteTime | Format-Table -AutoSize
} else {
    Write-Host 'result directory not found.'
}
Write-Host "`n=== latest checkpoint files ==="
if (Test-Path -LiteralPath $resultDir) {
    Get-ChildItem -LiteralPath $resultDir -Recurse -Filter *.pkl -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 10 FullName, LastWriteTime, Length | Format-Table -AutoSize
}
Write-Host "`n=== nvidia-smi ==="
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) { & nvidia-smi } else { Write-Host 'nvidia-smi not found in PATH.' }
