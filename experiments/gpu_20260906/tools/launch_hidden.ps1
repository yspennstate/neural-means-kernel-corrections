# Launch a python script with NO visible window, BelowNormal priority, pinned to
# the background logical processors 6-9 (mask 0x3C0; 4,5 and 14,15 belong to the movie lanes), stdout+stderr
# to a log file. Writes the pid to <log>.pid so a later turn can find the run.
#   powershell -NoProfile -ExecutionPolicy Bypass -File launch_hidden.ps1 -Script <py> -Log <log> [-Args "<args>"] [-WorkDir <dir>]
param(
  [Parameter(Mandatory=$true)][string]$Script,
  [Parameter(Mandatory=$true)][string]$Log,
  [string]$Args = "",
  [string]$WorkDir = "C:\Users\owner\GOAL20H_20260906\nmkc_gpu_experiments",
  [string]$Python = "C:\Python314\python.exe"
)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:NMKC_DATA = "C:\Users\owner\GOAL20H_20260906\nmkc_gpu_experiments\data\structmech"
$logDir = Split-Path -Parent $Log
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Python
$psi.Arguments = "-u `"$Script`" $Args"
$psi.WorkingDirectory = $WorkDir
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$null = $p.Start()
try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch {}
try { $p.ProcessorAffinity = [IntPtr]0x3C0 } catch {}
Set-Content -Path "$Log.pid" -Value $p.Id -Encoding ascii
# pump both streams into the log (async so neither pipe blocks the child)
$out = $p.StandardOutput
$err = $p.StandardError
$sw = New-Object System.IO.StreamWriter($Log, $true, (New-Object System.Text.UTF8Encoding($false)))
$sw.AutoFlush = $true
$sw.WriteLine("[launch_hidden] pid=$($p.Id) start=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') script=$Script args=$Args")
$errTask = $err.ReadToEndAsync()
while (-not $out.EndOfStream) { $sw.WriteLine($out.ReadLine()) }
$p.WaitForExit()
$sw.WriteLine("[stderr]")
$sw.WriteLine($errTask.Result)
$sw.WriteLine("[launch_hidden] exit=$($p.ExitCode) end=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$sw.Close()
