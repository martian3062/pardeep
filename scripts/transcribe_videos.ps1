# Transcribes audio extracted from personal videos, then labels speakers.
# Treated as multi-speaker ("call" kind) on purpose: most videos are forwarded
# clips, so diarization + the voice fingerprint decide what is actually him.
$env:UV_PROJECT_ENVIRONMENT = "E:\cache\venvs\me_too"
$env:UV_CACHE_DIR = "E:\cache\uv"
$env:HF_HOME = "E:\cache\huggingface"
$env:TORCH_HOME = "E:\cache\torch"
Set-Location E:\me_too
$py = "E:\cache\venvs\me_too\Scripts\python.exe"

for ($i = 1; $i -le 60; $i++) {
    Write-Output "=== video transcribe: attempt $i ==="
    & $py -u -m src.ingest.run audio data\raw\video_audio --kind call
    if ($LASTEXITCODE -eq 0) {
        Write-Output "=== transcription complete, diarizing ==="
        & $py -u -m src.ingest.run diarize
        Write-Output "=== video pipeline: done ==="
        break
    }
    Write-Output "=== crashed (exit $LASTEXITCODE), restarting in 5s ==="
    Start-Sleep -Seconds 5
}
