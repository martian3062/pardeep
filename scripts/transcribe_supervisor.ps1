# Relaunches call transcription until it completes cleanly.
# Files that hard-crash the decoder are auto-quarantined (claimed with
# item_count=-1 before processing), so each restart makes progress.
$env:UV_PROJECT_ENVIRONMENT = "E:\cache\venvs\me_too"
$env:UV_CACHE_DIR = "E:\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "E:\cache\uv-python"
$env:HF_HOME = "E:\cache\huggingface"
$env:TORCH_HOME = "E:\cache\torch"
Set-Location E:\me_too

for ($i = 1; $i -le 60; $i++) {
    Write-Output "=== supervisor: attempt $i ==="
    uv run python -u -m src.ingest.run audio data\raw\calls --kind call
    if ($LASTEXITCODE -eq 0) {
        Write-Output "=== supervisor: completed cleanly ==="
        break
    }
    Write-Output "=== supervisor: crashed (exit $LASTEXITCODE), restarting in 5s ==="
    Start-Sleep -Seconds 5
}
Write-Output "=== supervisor: done ==="
