# Trains the laptop-twin candidates on the VM once the 24B finishes.
# Kaggle cannot be told which GPU to use through its API and kept allocating
# Pascal P100s that modern PyTorch has no kernels for; the L4 trains a 2B in
# about twenty minutes, so this is both faster and deterministic.
$ErrorActionPreference = "Continue"
$key = "D:\data\evolet_rsa"
$vm = "pardeep@34.126.112.227"
$remote = "~/pardeep_self"

Write-Output "=== waiting for the 24B run to finish ==="
while ($true) {
    $busy = ssh -i $key $vm "pgrep -f 'train_qlora.py --model sarvam24b' >/dev/null && echo yes || echo no"
    if ($busy -match "no") { break }
    Start-Sleep -Seconds 120
}

foreach ($m in @("sarvam1", "sarvam1v05")) {
    Write-Output "=== training $m ==="
    ssh -i $key $vm "cd $remote && rm -f train_$m.log && source .venv/bin/activate && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_qlora.py --model $m --epochs 2 --rank 32 --out adapters/$m > train_$m.log 2>&1"
    Write-Output "=== $m done, fetching ==="
    New-Item -ItemType Directory -Force -Path "E:\me_too\data\models\$m" | Out-Null
    scp -i $key -r "${vm}:$remote/adapters/$m/*" "E:\me_too\data\models\$m\"
}
Write-Output "=== all small twins trained ==="
