$ErrorActionPreference = 'Continue'

# Direct install of the prefill-fix serve exe (the blackout window's Copy-Item silently failed while
# the running server held the file open). Kill the listener, verify the handle is gone, copy with
# explicit error capture, verify the hash matches the build exe, restart from the start bat, health.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_fixnow_serve.bat'
$serveLog   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_fixnow.log'
$base       = 'http://127.0.0.1:8888'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) { if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break } }
if (-not $serveLine) { Log 'FATAL no serve line in start bat'; exit 2 }

$want = (Get-FileHash -LiteralPath $buildExe -Algorithm SHA256).Hash
Log ('BUILD_HASH ' + $want)

Log 'grace 5s'
Start-Sleep -Seconds 5

$s = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($s) { taskkill /F /T /PID ([int]$s.OwningProcess) 2>&1 | Out-Null; Log ('killed pid=' + $s.OwningProcess) }
taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
$gone = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try { Rename-Item -LiteralPath $desktopExe -NewName 'infernix-serve.exe.locktest' -ErrorAction Stop; Remove-Item -LiteralPath ($desktopExe + '.locktest') -Force; $gone = $true; break } catch { }
}
if (-not $gone) { Log 'COPY_BLOCKED: handle still open after 60s, aborting (server NOT changed)'; exit 3 }

Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force -ErrorAction Stop
$got = (Get-FileHash -LiteralPath $desktopExe -Algorithm SHA256).Hash
Log ('INSTALLED ' + $got + ' match=' + ($got -eq $want))
if ($got -ne $want) { Log 'HASH_MISMATCH'; exit 4 }

if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
$serveCmd = ('"' + $desktopExe + '" ' + $serveLine.Substring('infernix-serve.exe'.Length).Trim() + ' > "' + $serveLog + '" 2>&1')
Set-Content -LiteralPath $serveBat -Value @('@echo off', ('cd /d "' + $desktopDir + '"'), $serveCmd) -Encoding ASCII
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
Log 'serve launched via explorer (visible window)'

$healthy = $false
for ($i = 0; $i -lt 160; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5
        if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { $healthy = $true; break }
    } catch { }
}
Log ('HEALTHY=' + $healthy)
if ($healthy) { Log 'RESULT: FIX_DEPLOYED_AND_HEALTHY'; exit 0 }
if (Test-Path -LiteralPath $serveLog) { Get-Content -LiteralPath $serveLog -Tail 25 | ForEach-Object { Log ('SVRLOG| ' + $_) } }
Log 'RESULT: SERVER_NOT_UP (exe IS the fixed one; check SVRLOG above)'
exit 5
