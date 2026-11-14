$ErrorActionPreference = 'Continue'

# #75 blackout: device-qualify the VAE reference encoder, deploy the rebuilt serve (with the
# `image` field), and certify a reference-image generation plus a text-to-image regression.
# Self-restoring: any failure path relaunches the captured live server command.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$buildDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build'
$cmake      = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$outDir     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out'
$artifact   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v2.infernix'
$deviceTest = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Debug\infernix_image_vae_encode_prod_cuda_test.exe'
$serveLog   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_refimage.log'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_refimage_serve.bat'
$restoreBat = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_refimage_restore.bat'
$genRef     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\gen_refimage.ps1'
$genKatt    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\gen_katt.ps1'
$buildLog   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\refimage_build.log'
$deviceLog  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\refimage_device_test.log'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) { if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break } }
if (-not $serveLine) { Log 'FATAL no serve line in start bat'; exit 2 }
Log ('SERVE_LINE_OK len=' + $serveLine.Length)

Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

# ---- PHASE 1: capture live cmd, stop it, drain VRAM ----
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
        if (-not (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue)) { break }
    }
}
for ($i = 0; $i -lt 40; $i++) {
    $f = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if ($f -and [int]$f -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log 'PHASE1 drained'

function Restore-Live {
    if (-not $liveCmd) { Log 'RESTORE: no live cmd (nothing was running)'; return }
    Set-Content -LiteralPath $restoreBat -Value (@('@echo off', ('cd /d "' + $desktopDir + '"'), $liveCmd)) -Encoding ASCII
    Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $restoreBat + '"') | Out-Null
    Log 'RESTORE: live command relaunched via explorer (survives this script, visible window)'
}

# ---- PHASE 2: build serve (Release) + the device encode test (Debug) ----
Log 'PHASE2 build infernix-serve + vae encode prod test'
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
& $cmake --build $buildDir --target infernix-serve --config Release -j 6 2>&1 | Tee-Object -FilePath $buildLog | Select-Object -Last 15
$buildRc = $LASTEXITCODE
& $cmake --build $buildDir --target infernix_stage_test_runtime_dlls infernix_image_vae_encode_prod_cuda_test -j 6 2>&1 | Tee-Object -FilePath $deviceLog | Select-Object -Last 15
$testBuildRc = $LASTEXITCODE
Log ('PHASE2 build rc=' + $buildRc + ' testbuild rc=' + $testBuildRc)
if ($buildRc -ne 0 -or $testBuildRc -ne 0) {
    Log 'BUILD_FAILED: restoring live server'
    Restore-Live
    exit 3
}

# ---- PHASE 3: device-qualify the reference encoder against the FP32 oracle ----
Log 'PHASE3 device encode test'
$env:INFERNIX_TEST_ARTIFACT   = $artifact
$env:INFERNIX_REF_ENCODE_ORACLE = $outDir
$env:INFERNIX_VAE_STAGE_STATS = '1'
$dt = & $deviceTest 2>&1
$dtRc = $LASTEXITCODE
$dt | ForEach-Object { Log ('DEVTEST| ' + $_) }
Log ('DEVTEST rc=' + $dtRc)

# ---- PHASE 4: deploy + launch ----
Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force
Log ('PHASE4 copied exe size=' + (Get-Item -LiteralPath $desktopExe).Length)
if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
$serveCmd = ('"' + $desktopExe + '" ' + $serveLine.Substring('infernix-serve.exe'.Length).Trim() + ' > "' + $serveLog + '" 2>&1')
Set-Content -LiteralPath $serveBat -Value @('@echo off', ('cd /d "' + $desktopDir + '"'), $serveCmd) -Encoding ASCII
function Kill-Serve {
    $s = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($s) { taskkill /F /T /PID ([int]$s.OwningProcess) 2>&1 | Out-Null; Log ('killed serve pid=' + $s.OwningProcess) }
    taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
    Start-Sleep -Seconds 5
}
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
Log 'PHASE4 serve launched via explorer (survives this script, visible window)'

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
    Kill-Serve
    Restore-Live
    exit 4
}

# ---- PHASE 5: reference-image generation + text-to-image regression ----
$pngRef   = 'C:\Users\Micke\Desktop\Infernix FlashNext\refimage-cert.png'
$pngPlain = 'C:\Users\Micke\Desktop\Infernix FlashNext\refimage-regression.png'
Log 'PHASE5a reference-image generation (oracle source.png as `image`)'
$gr = (& $genRef -Size '768x768' -Out $pngRef 2>&1) -join "`n"
Log ('GENREF: ' + $gr)
$okRef = (Test-Path -LiteralPath $pngRef) -and ((Get-Item -LiteralPath $pngRef).Length -gt 10000)
Log 'PHASE5b text-to-image regression (no image field)'
$gp = (& $genKatt -Size '768x768' -Out $pngPlain 2>&1) -join "`n"
Log ('GENPLAIN: ' + $gp)
$okPlain = (Test-Path -LiteralPath $pngPlain) -and ((Get-Item -LiteralPath $pngPlain).Length -gt 10000)

# ---- PHASE 6: serve log tail ----
Log 'PHASE6 serve log tail'
if (Test-Path -LiteralPath $serveLog) {
    Get-Content -LiteralPath $serveLog -Tail 60 | ForEach-Object { Log ('SVRLOG| ' + $_) }
    Select-String -LiteralPath $serveLog -Pattern 'freed|MiB|out of memory|CUDA|OOM|interlude|image' -ErrorAction SilentlyContinue |
        Select-Object -Last 25 | ForEach-Object { Log ('SVRGREP| ' + $_.Line) }
} else {
    Log 'NO_SERVE_LOG'
}

# ---- PHASE 7: decide ----
function Server-Up {
    try { return ([int](Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200) } catch { return $false }
}
if ($okRef -and $okPlain) {
    if (-not (Server-Up)) {
        Log 'PHASE7: server dropped during gen; relaunching so a working server remains'
        Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
        for ($i = 0; $i -lt 130; $i++) { Start-Sleep -Seconds 5; if (Server-Up) { break } }
    }
    Log ('RESULT: SUCCESS devtest_rc=' + $dtRc + ' ref=' + $okRef + ' plain=' + $okPlain + ' - flash-next left running')
    exit 0
} else {
    Log ('RESULT: FAILED devtest_rc=' + $dtRc + ' ref=' + $okRef + ' plain=' + $okPlain + ' - restoring live server')
    Kill-Serve
    Start-Sleep -Seconds 10
    Restore-Live
    exit 5
}
