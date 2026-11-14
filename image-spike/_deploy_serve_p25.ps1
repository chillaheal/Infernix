# Sixth deploy file, copied from _deploy_serve_p24.ps1 (the pattern that worked) with only the backup
# name and the pause reason changed, so .bak-p24 stays as the rollback of the previous deploy. Deploys
# the exe of the auto-skip fix (build\apps\Release\infernix-serve.exe, 568907264 bytes 21:16:08).
# Kill under Continue, hold the watchdog, launch the user's own bat, prove health + busy signal.
$ErrorActionPreference = 'Stop'
$src = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$dst = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$bat = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next local.bat'
$dir = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit'
foreach ($p in @($src, $dst, $bat)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Output "MISSING $p"; exit 2 }
}
Write-Output ("src {0} {1} bytes" -f (Get-Item $src).LastWriteTime, (Get-Item $src).Length)
Write-Output ("dst {0} {1} bytes" -f (Get-Item $dst).LastWriteTime, (Get-Item $dst).Length)

$pause = Join-Path $dir '_serve_watch.pause'
$note  = @{ ends_unix_s = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + 2400;
           reason = 'deploy of the auto-skip fix (44dd4af)' } | ConvertTo-Json -Compress
Set-Content -LiteralPath $pause -Value $note -Encoding ascii
Write-Output ("pause " + $note)

$bak = "$dst.bak-p25"
Copy-Item -LiteralPath $dst -Destination $bak -Force
Write-Output ("backup " + $bak + " " + (Get-Item $bak).Length + " bytes")

$copied = $false
for ($try = 1; $try -le 12; $try = $try + 1) {
    $ErrorActionPreference = 'Continue'
    $null = & taskkill.exe /IM infernix-serve.exe /F 2>&1
    $ErrorActionPreference = 'Stop'
    $waited = 0
    while ($waited -lt 40) {
        if (-not (Get-Process -Name infernix-serve -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Seconds 2; $waited = $waited + 2
    }
    try {
        Copy-Item -LiteralPath $src -Destination $dst -Force
        $copied = $true
        Write-Output ("copied on try {0}: dst {1} {2} bytes" -f $try, (Get-Item $dst).LastWriteTime, (Get-Item $dst).Length)
        break
    } catch {
        Write-Output ("try {0}: copy refused ({1})" -f $try, $_.Exception.GetType().Name)
        Start-Sleep -Seconds 5
    }
}
if (-not $copied) { Write-Output 'FAILED: destination stayed locked'; exit 1 }

Start-Process -FilePath $bat -WorkingDirectory (Split-Path $bat)
Write-Output 'bat started'
$deadline = (Get-Date).AddMinutes(18)
$h = ''
do {
    Start-Sleep -Seconds 10
    $ErrorActionPreference = 'Continue'
    try { $h = (Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 10).Content }
    catch { $h = '' }
    $ErrorActionPreference = 'Stop'
} while ((Get-Date) -lt $deadline -and $h -notmatch '"status":\s*"ok"')
Write-Output "health=$h"
if ($h -match 'image_jobs') { Write-Output 'BUSY_SIGNAL_PRESENT' } else { Write-Output 'BUSY_SIGNAL_MISSING' }
Remove-Item -LiteralPath $pause -Force -ErrorAction SilentlyContinue
Write-Output 'DEPLOY_DONE'
