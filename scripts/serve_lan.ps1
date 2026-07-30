# Serve the twin to your phone on the same Wi-Fi.
#
# HTTPS is mandatory, not optional: getUserMedia only works in a secure context,
# so over plain http://<lan-ip> a phone loads the page, shows the record button,
# and silently captures nothing. The cert is self-signed, so the browser warns
# once — accept it and the origin becomes secure.
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve_lan.ps1

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike '127.*' -and
        $_.IPAddress -notlike '169.254.*' -and
        $_.InterfaceAlias -notlike '*WSL*' -and
        $_.InterfaceAlias -notlike '*Virtual*'
    } | Select-Object -First 1).IPAddress

if (-not $ip) { throw 'No LAN address found — is Wi-Fi connected?' }

Write-Host ''
Write-Host '  On your phone, open:' -ForegroundColor Cyan
Write-Host "     https://$ip`:3000/record" -ForegroundColor White
Write-Host ''
Write-Host '  The certificate is self-signed, so the browser will warn.' -ForegroundColor DarkGray
Write-Host '  Tap Advanced then Proceed. Without HTTPS the mic will not work.' -ForegroundColor DarkGray
Write-Host ''

# Windows blocks inbound 3000 by default, and blocks it harder when the network
# is classed Public — which is the default for a new Wi-Fi, including home Wi-Fi.
# The rule must name the profile the network is actually on or it does nothing.
$rule = Get-NetFirewallRule -DisplayName 'Pardeep_Self dev 3000' -ErrorAction SilentlyContinue
if (-not $rule) {
    $category = (Get-NetConnectionProfile |
        Where-Object { $_.IPv4Connectivity -ne 'Disconnected' } |
        Select-Object -First 1).NetworkCategory
    $profileArg = if ($category -eq 'Public') { 'Public' } else { 'Private' }

    Write-Host "  Inbound port 3000 is closed, and this network is classed $category." -ForegroundColor Yellow
    Write-Host '  Run this ONCE in an admin PowerShell, then the phone can connect:' -ForegroundColor Yellow
    Write-Host ''
    Write-Host "     New-NetFirewallRule -DisplayName 'Pardeep_Self dev 3000' -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow -Profile $profileArg" -ForegroundColor White
    Write-Host ''
    if ($category -eq 'Public') {
        Write-Host '  On your own Wi-Fi you may prefer to mark the network Private instead,' -ForegroundColor DarkGray
        Write-Host '  which is a broader change to how this machine is reachable — your call:' -ForegroundColor DarkGray
        Write-Host "     Set-NetConnectionProfile -InterfaceAlias 'Wi-Fi' -NetworkCategory Private" -ForegroundColor DarkGray
        Write-Host ''
    }
}

$env:UV_PROJECT_ENVIRONMENT = 'E:\cache\venvs\me_too'

# The API stays bound to localhost. Vite proxies /api to it server-side, so the
# network only ever talks to Vite and the API is not exposed.
Write-Host '  starting API on 127.0.0.1:8100 ...' -ForegroundColor DarkGray
$api = Start-Process -PassThru -WindowStyle Hidden -WorkingDirectory $repo `
    -FilePath 'uv' -ArgumentList @('run', '--no-sync', 'litestar', '--app', 'app.api.main:app', 'run', '--port', '8100')

try {
    Write-Host '  starting UI on 0.0.0.0:3000 (https) ...' -ForegroundColor DarkGray
    Push-Location (Join-Path $repo 'app\web')
    npm run dev:lan
} finally {
    Pop-Location
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue }
    Write-Host '  stopped.' -ForegroundColor DarkGray
}
