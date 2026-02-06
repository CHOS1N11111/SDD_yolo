$base = $PSScriptRoot
if (-not $base) {
  $base = (Get-Location).Path
}
$src = Join-Path $base "backup"
$dst = Join-Path $base "backup_history"

New-Item -ItemType Directory -Path $dst -Force | Out-Null

$log = Join-Path $dst "backup_log.txt"
function Write-Log {
  param([string]$Message)
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$ts $Message" | Tee-Object -FilePath $log -Append | Out-Host
}

Write-Log "Backup loop started. Source: $src"

while ($true) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $last = Join-Path $src "yolov4-custom_last.weights"
  $best = Join-Path $src "yolov4-custom_best.weights"

  if (Test-Path $last) {
    $outLast = Join-Path $dst "yolov4-custom_last_$ts.weights"
    Copy-Item $last $outLast -ErrorAction SilentlyContinue
    Write-Log "Copied last -> $outLast"
  } else {
    Write-Log "Missing last weights: $last"
  }

  if (Test-Path $best) {
    $outBest = Join-Path $dst "yolov4-custom_best_$ts.weights"
    Copy-Item $best $outBest -ErrorAction SilentlyContinue
    Write-Log "Copied best -> $outBest"
  } else {
    Write-Log "Missing best weights: $best"
  }

  Start-Sleep -Seconds 600
}
