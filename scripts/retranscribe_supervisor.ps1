# Re-transcribes all calls with large-v3 + anti-hallucination settings.
# First attempt clears old transcripts and carries language hints forward;
# later attempts resume (files already done are skipped by sha256).
$env:UV_PROJECT_ENVIRONMENT = "E:\cache\venvs\me_too"
$env:UV_CACHE_DIR = "E:\cache\uv"
$env:HF_HOME = "E:\cache\huggingface"
$env:TORCH_HOME = "E:\cache\torch"
Set-Location E:\me_too
$py = "E:\cache\venvs\me_too\Scripts\python.exe"

for ($i = 1; $i -le 60; $i++) {
    Write-Output "=== retranscribe: attempt $i ==="
    if ($i -eq 1) {
        & $py -u -m src.ingest.run retranscribe data\raw\calls
    } else {
        & $py -u -m src.ingest.run audio data\raw\calls --kind call
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Output "=== retranscribe: transcription complete, relabeling ==="
        & $py -u -m src.ingest.run diarize
        Write-Output "=== retranscribe: done ==="
        break
    }
    Write-Output "=== retranscribe: crashed (exit $LASTEXITCODE), restarting in 5s ==="
    Start-Sleep -Seconds 5
}
