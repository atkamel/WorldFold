# Milestone V job queue (one Isaac process at a time, only while fewer than 3 run machine-wide). Resumable: a job
# whose .done marker exists is skipped. Run from the worktree root.
param([string[]]$Jobs)
$ErrorActionPreference = "Continue"
. C:\Users\ethan\Documents\Projects\watai\WorldFold\isaac\env_windows.ps1
$py = "C:\Users\ethan\Documents\Projects\watai\WorldFold\.venv-isaac\Scripts\python.exe"
$all = [ordered]@{
  "vec_check" = @{ args = @("-u", "isaac/vec_check.py", "--n", "3", "--out", "outputs/isaac/vec"); env = @{} }
  "bench_B2"  = @{ args = @("-u", "isaac/bench_vec.py", "--B", "2", "--out", "outputs/isaac/vec/bench.jsonl"); env = @{} }
  "bench_B1"  = @{ args = @("-u", "isaac/bench_vec.py", "--B", "1", "--episodes", "3", "--out", "outputs/isaac/vec/bench.jsonl"); env = @{} }
  "bench_B4"  = @{ args = @("-u", "isaac/bench_vec.py", "--B", "4", "--out", "outputs/isaac/vec/bench.jsonl"); env = @{} }
  "bench_B8"  = @{ args = @("-u", "isaac/bench_vec.py", "--B", "8", "--out", "outputs/isaac/vec/bench.jsonl"); env = @{} }
  "parity_B4" = @{ args = @("-u", "-m", "imitation.evaluate", "--backend", "isaac_weld", "--ckpt", "expert", "--sets", "id_easy", "--n", "40", "--workers", "1", "--out", "outputs/isaac/vec/parity_B4.json"); env = @{ WORLDFOLD_ISAAC_ENVS_PER_PROC = "4" } }
  "parity_B1" = @{ args = @("-u", "-m", "imitation.evaluate", "--backend", "isaac_weld", "--ckpt", "expert", "--sets", "id_easy", "--n", "40", "--workers", "1", "--out", "outputs/isaac/vec/parity_B1.json"); env = @{ WORLDFOLD_ISAAC_ENVS_PER_PROC = "1" } }
}
$Jobs = @($Jobs | % { $_ -split "," } | ? { $_ })
foreach ($name in $Jobs) {
  $job = $all[$name]
  if ($null -eq $job) { "unknown job $name"; break }
  $done = "outputs/isaac/vec/$name.done"
  if (Test-Path $done) { "skip $name (done)"; continue }
  while ($true) {
    $n = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match '.venv-isaac' -and $_.CommandLine -notmatch 'pytest' }).Count
    if ($n -lt 3) { break }
    Start-Sleep -Seconds 60
  }
  "start $name at $(Get-Date -Format s) ($n other Isaac procs)"
  foreach ($k in $job.env.Keys) { Set-Item -Path "env:$k" -Value $job.env[$k] }
  & $py @($job.args) *>&1 | Out-File -Encoding utf8 "outputs/isaac/vec/$name.log"
  $code = $LASTEXITCODE
  foreach ($k in $job.env.Keys) { Remove-Item -Path "env:$k" -ErrorAction SilentlyContinue }
  "end $name exit $code at $(Get-Date -Format s)"
  if ($code -eq 0) { New-Item -ItemType File -Force $done | Out-Null } else { "stopping: $name failed"; break }
}
