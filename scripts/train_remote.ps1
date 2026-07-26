# Orchestrates a training run on the GPU VM and brings the artefacts home.
#
#   scripts\train_remote.ps1 -Model qwen4b -ExportGguf
#   scripts\train_remote.ps1 -Model sarvam24b -Epochs 2
#
# The VM is stateless compute: only the training JSONL goes up, the adapter
# comes back, and the VM copy is wiped. See "Portability rule" in README.
param(
    [string]$Model = "qwen4b",
    [double]$Epochs = 3.0,
    [switch]$ExportGguf,
    [switch]$KeepRemoteData
)
$ErrorActionPreference = "Stop"

$envFile = Get-Content "E:\me_too\.env" | Where-Object { $_ -match "^TRAIN_VM=|^TRAIN_VM_KEY=" }
$vm  = ($envFile | Where-Object { $_ -match "^TRAIN_VM=" })     -replace "^TRAIN_VM=", ""
$key = ($envFile | Where-Object { $_ -match "^TRAIN_VM_KEY=" }) -replace "^TRAIN_VM_KEY=", ""
if (-not $vm -or -not $key) { throw "TRAIN_VM / TRAIN_VM_KEY missing from .env" }
$remote = "~/pardeep_self"

Write-Host "==> uploading dataset + training script to $vm"
ssh -i $key $vm "mkdir -p $remote/datasets $remote/adapters"
scp -i $key E:\me_too\data\datasets\sft_train.jsonl E:\me_too\data\datasets\sft_eval.jsonl "${vm}:$remote/datasets/"
scp -i $key E:\me_too\training\train_qlora.py "${vm}:$remote/"

$gguf = if ($ExportGguf) { "--export-gguf" } else { "" }
Write-Host "==> training $Model (this runs for a while; output streams below)"
ssh -i $key $vm "cd $remote && source .venv/bin/activate && python train_qlora.py --model $Model --epochs $Epochs $gguf"

Write-Host "==> fetching artefacts"
$local = "E:\me_too\data\models\$Model"
New-Item -ItemType Directory -Force -Path $local | Out-Null
scp -i $key -r "${vm}:$remote/adapters/$Model/*" $local

if (-not $KeepRemoteData) {
    Write-Host "==> wiping personal data from VM"
    ssh -i $key $vm "rm -rf $remote/datasets/* $remote/adapters/$Model"
}
Write-Host "done -> $local"
