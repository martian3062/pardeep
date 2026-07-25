# Relaunches call diarization until it completes cleanly (idempotent: only
# pending/unlabeled calls are processed on each attempt).
$env:UV_PROJECT_ENVIRONMENT = "E:\cache\venvs\me_too"
$env:UV_CACHE_DIR = "E:\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "E:\cache\uv-python"
$env:HF_HOME = "E:\cache\huggingface"
$env:TORCH_HOME = "E:\cache\torch"
Set-Location E:\me_too

for ($i = 1; $i -le 20; $i++) {
    Write-Output "=== diarize supervisor: attempt $i ==="
    uv run python -u -m src.ingest.run diarize
    if ($LASTEXITCODE -eq 0) {
        Write-Output "=== diarize supervisor: completed cleanly ==="
        break
    }
    Write-Output "=== diarize supervisor: crashed (exit $LASTEXITCODE), restarting in 5s ==="
    Start-Sleep -Seconds 5
}
Write-Output "=== diarize supervisor: done ==="
