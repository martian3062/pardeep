# One-time environment setup for Pardeep_Self.
# Creates E:\cache dirs, points every model/package cache there (user env vars),
# installs Python 3.11 and the project venv at E:\cache\venvs\me_too.
$ErrorActionPreference = "Stop"

$cache = "E:\cache"
foreach ($d in @("venvs", "uv", "uv-python", "pip", "huggingface", "torch", "ollama", "xdg")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $cache $d) | Out-Null
}

$vars = @{
    UV_PROJECT_ENVIRONMENT = "E:\cache\venvs\me_too"
    UV_CACHE_DIR           = "E:\cache\uv"
    UV_PYTHON_INSTALL_DIR  = "E:\cache\uv-python"
    PIP_CACHE_DIR          = "E:\cache\pip"
    HF_HOME                = "E:\cache\huggingface"
    TORCH_HOME             = "E:\cache\torch"
    OLLAMA_MODELS          = "E:\cache\ollama"
    XDG_CACHE_HOME         = "E:\cache\xdg"
}
foreach ($k in $vars.Keys) {
    [Environment]::SetEnvironmentVariable($k, $vars[$k], "User")   # persistent
    Set-Item -Path "env:$k" -Value $vars[$k]                        # this session
}
Write-Host "User env vars set. NOTE: quit + restart Ollama once for OLLAMA_MODELS to apply."

Set-Location (Split-Path $PSScriptRoot -Parent)
uv python install 3.11
uv sync --all-groups
Write-Host "Done. Venv: E:\cache\venvs\me_too  |  Run things with: uv run python -m src.ingest.run --help"
