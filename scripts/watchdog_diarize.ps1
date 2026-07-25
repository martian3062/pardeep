# Watchdog for the diarization run: supervisors catch crashes, this catches
# SILENT HANGS. If the log stops growing for 15 minutes, the python worker is
# tree-killed; the supervisor loop relaunches it and the cache resumes progress.
$log = "E:\me_too\data\processed\diarize_v4.log"
$lastSize = -1
$stale = 0
while ($true) {
    Start-Sleep -Seconds 300
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "diarize supervisor: done" -Quiet)) {
        Write-Output "watchdog: run complete, exiting"
        break
    }
    $size = if (Test-Path $log) { (Get-Item $log).Length } else { 0 }
    if ($size -eq $lastSize) { $stale++ } else { $stale = 0; $lastSize = $size }
    if ($stale -ge 3) {
        Write-Output "watchdog: no log growth for 15 min — restarting worker"
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match "src\.ingest\.run diarize" } |
            ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>&1 | Out-Null }
        $stale = 0
    }
}
