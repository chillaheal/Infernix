$ErrorActionPreference = 'Continue'

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$buildDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build'
$cmake      = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveLog   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_yieldfix.log'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_yieldfix_serve.bat'
$restoreBat = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_yieldfix_restore.bat'
$genPs1     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\gen_katt.ps1'
$buildLog   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\yieldfix_build.log'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

# ---- serve line from the production bat (desktop exe, nvfp4, image flags) ----
$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) { if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break } }
if (-not $serveLine) { Log 'FATAL no serve line in start bat'; exit 2 }
Log ('SERVE_LINE_OK len=' + $serveLine.Length)

# ---- PHASE 0: grace ----
Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

# ---- PHASE 1: capture live cmd, stop it, drain ----
$c = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$livePid = 0; $liveCmd = ''
if ($c) {
    $livePid = [int]$c.OwningProcess
    try { $liveCmd = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $livePid)).CommandLine } catch { $liveCmd = '' }
}
Log ('LIVE pid=' + $livePid + ' cmdlen=' + $liveCmd.Length)
if ($livePid -gt 0) {
    taskkill /F /T /PID $livePid 2>&1 | Out-Null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        $c2 = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue
        if (-not $c2) { break }
    }
}
for ($i = 0; $i -lt 40; $i++) {
    $f = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if ($f -and [int]$f -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log 'PHASE1 drained'

# ---- PHASE 2: build the serve target ----
Log 'PHASE2 build infernix-serve (Release)'
& $cmake --build $buildDir --target infernix-serve --config Release -j 2>&1 | Tee-Object -FilePath $buildLog | Select-Object -Last 30
$buildRc = $LASTEXITCODE
Log ('PHASE2 build rc=' + $buildRc)

function Restore-Live {
    if (-not $liveCmd) { Log 'RESTORE: no live cmd (nothing was running)'; return }
    Set-Content -LiteralPath $restoreBat -Value (@('@echo off', $liveCmd)) -Encoding ASCII
    Start-Process -FilePath 'cmd.exe' -ArgumentList ('/c', ('"' + $restoreBat + '"')) | Out-Null
    Log 'RESTORE: live command relaunched (visible window)'
}

if ($buildRc -ne 0) {
    Log 'BUILD_FAILED: restoring live server'
    Restore-Live
    exit 3
}

# ---- PHASE 3: swap the desktop exe to the fresh build ----
Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force
Log ('PHASE3 copied exe size=' + (Get-Item -LiteralPath $desktopExe).Length)

# ---- PHASE 4: launch flash-next with a log redirect (visible window) ----
if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
$serveCmd = ('"' + $desktopExe + '" ' + $serveLine.Substring('infernix-serve.exe'.Length).Trim() + ' > "' + $serveLog + '" 2>&1')
Set-Content -LiteralPath $serveBat -Value @('@echo off', $serveCmd) -Encoding ASCII
$serveProc = Start-Process -FilePath 'cmd.exe' -ArgumentList ('/c', ('"' + $serveBat + '"')) -PassThru
Log ('PHASE4 serve launched pid=' + $serveProc.Id)

# wait healthy (flash-next loads 200k ctx + experts: allow ~10 min)
$healthy = $false
for ($i = 0; $i -lt 130; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 5
        if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { $healthy = $true; break }
    } catch { }
}
Log ('PHASE4 healthy=' + $healthy)
if (-not $healthy) {
    Log 'SERVER_NOT_UP: restoring live server'
    taskkill /F /T /PID $serveProc.Id 2>&1 | Out-Null
    Restore-Live
    exit 4
}

# ---- PHASE 5: deliver at the 768x768 default, then probe the 1024x1024 ceiling ----
$png768  = 'C:\Users\Micke\Desktop\Infernix FlashNext\maine-coon-katt.png'
$png1024 = 'C:\Users\Micke\Desktop\Infernix FlashNext\maine-coon-katt-1024.png'
Log 'PHASE5a generating 768x768 (default)'
$g768 = (& $genPs1 -Size '768x768' -Out $png768 2>&1) -join "`n"
Log ('GEN768: ' + $g768)
$ok768 = (Test-Path -LiteralPath $png768) -and ((Get-Item -LiteralPath $png768).Length -gt 10000)
Log 'PHASE5b probing 1024x1024 (ceiling; may OOM and drop the server)'
$g1024 = (& $genPs1 -Size '1024x1024' -Out $png1024 2>&1) -join "`n"
Log ('GEN1024: ' + $g1024)
$ok1024 = (Test-Path -LiteralPath $png1024) -and ((Get-Item -LiteralPath $png1024).Length -gt 10000)

# ---- PHASE 6: capture the serve log tail (freed VRAM / OOM) BEFORE any relaunch truncates it ----
Log 'PHASE6 serve log tail'
if (Test-Path -LiteralPath $serveLog) {
    $tail = Get-Content -LiteralPath $serveLog -Tail 90
    $tail | ForEach-Object { Log ('SVRLOG| ' + $_) }
    $grep = Select-String -LiteralPath $serveLog -Pattern 'freed|MiB|out of memory|CUDA|OOM|interlude|allocat|yield' -ErrorAction SilentlyContinue
    $grep | Select-Object -Last 30 | ForEach-Object { Log ('SVRGREP| ' + $_.Line) }
} else {
    Log 'NO_SERVE_LOG (process may have crashed before logging)'
}

# ---- PHASE 7: decide ----
function Server-Up {
    try { return ([int](Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200) } catch { return $false }
}
$catOk = $ok768 -or $ok1024
if ($catOk) {
    if (-not (Server-Up)) {
        Log 'PHASE7: 1024 probe dropped the server; relaunching flash-next (no gen) so a working image-capable server remains'
        Start-Process -FilePath 'cmd.exe' -ArgumentList ('/c', ('"' + $serveBat + '"')) | Out-Null
        for ($i = 0; $i -lt 130; $i++) { Start-Sleep -Seconds 5; if (Server-Up) { break } }
    }
    Log ('RESULT: SUCCESS ok768=' + $ok768 + ' ok1024=' + $ok1024 + ' - leaving flash-next (image-capable) running')
    exit 0
} else {
    Log 'RESULT: FAILED - killing flash-next and restoring live server'
    taskkill /F /T /PID $serveProc.Id 2>&1 | Out-Null
    Start-Sleep -Seconds 10
    Restore-Live
    exit 5
}
