param([switch]$SkipCert)
$ErrorActionPreference = 'Continue'

# Prefill-accounting fix (2026-10-26 live crash): blackout window that
#   1) stops the live server and drains the GPU,
#   2) runs infernix_qwen4_exp_prefill_resume_accounting_real_test (needs the free GPU),
#   3) only if the cert passes: installs the fixed serve exe on the Desktop instance and starts it
#      from the start bat's serve line (384k + YaRN 1.5 + nvfp4, the flags the user set),
#   4) health + plain-chat regression on the new server.
# Failure before the copy restores the captured live command with the OLD exe. Self-restoring.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$buildDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build'
$cmake      = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'
$restoreBat = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_prefill_fix_restore.bat'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_prefill_fix_serve.bat'
$serveLog   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_prefill_fix.log'
$certLog    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\prefill_resume_cert_run.log'
$artifact   = 'C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram      = 'C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$base       = 'http://127.0.0.1:8888'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) { if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break } }
if (-not $serveLine) { Log 'FATAL no serve line in start bat'; exit 2 }
foreach ($p in @($desktopExe, $buildExe, $artifact, $ngram, $cmake)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing ' + $p); exit 2 }
}
Log 'PREREQ ok'

Log 'PHASE0 grace 5s (short window)'
Start-Sleep -Seconds 5

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
    Log 'RESTORE: previous live command relaunched via explorer (visible window)'
}

# ---- PHASE 2: device cert on the free GPU (the crash reproducer) ----
$env:INFERNIX_QWEN4_ARTIFACT = $artifact
$env:INFERNIX_QWEN4_NGRAM    = $ngram
if ($SkipCert) { Log 'PHASE2 cert SKIPPED (-SkipCert: deploy-only window)' }
else {
Log 'PHASE2 cert run'
& $cmake --build $buildDir --target infernix_qwen4_exp_prefill_resume_accounting_real_test --config Release -j 6 2>&1 | Out-Null
& ctest --test-dir $buildDir -C Release -R infernix_qwen4_exp_prefill_resume_accounting_real_test --output-on-failure 2>&1 | Tee-Object -FilePath $certLog | Select-Object -Last 40
$certRc = $LASTEXITCODE
Log ('PHASE2 cert rc=' + $certRc)
if ($certRc -ne 0) {
    Log 'CERT_FAILED: no deploy, restoring live server'
    Restore-Live
    exit 3
}
}

# ---- PHASE 3: backup + install the fixed exe ----
$backup = $desktopExe + '.pre-prefill-fix.bak'
Copy-Item -LiteralPath $desktopExe -Destination $backup -Force
Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force
Log ('PHASE3 installed size=' + (Get-Item -LiteralPath $desktopExe).Length + ' backup=' + (Get-Item -LiteralPath $backup).Length)

# ---- PHASE 4: start from the start bat serve line (384k + YaRN + nvfp4) ----
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
for ($i = 0; $i -lt 160; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5
        if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { $healthy = $true; break }
    } catch { }
}
Log ('PHASE4 healthy=' + $healthy)
if (-not $healthy) {
    if (Test-Path -LiteralPath $serveLog) { Get-Content -LiteralPath $serveLog -Tail 30 | ForEach-Object { Log ('SVRLOG| ' + $_) } }
    Log 'SERVER_NOT_UP: restoring previous exe + live command'
    Kill-Serve
    Copy-Item -LiteralPath $backup -Destination $desktopExe -Force
    Restore-Live
    exit 4
}

# ---- PHASE 5: plain chat regression on the new server ----
$okChat = $false
$chatId = ''
try { $chatId = [string](((Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json).data[0].id) } catch {}
$chatBody = @{ model = $chatId; messages = @(@{ role = 'user'; content = 'Reply with exactly: OK' }); max_tokens = 16 } | ConvertTo-Json -Depth 5 -Compress
for ($try = 1; $try -le 6; $try++) {
    try {
        $rc = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body $chatBody -UseBasicParsing -TimeoutSec 300
        $oc = $rc.Content | ConvertFrom-Json
        $okChat = ([int]$rc.StatusCode -eq 200 -and $oc.choices.Count -ge 1)
        Log ('CHAT try=' + $try + ' HTTP=' + [int]$rc.StatusCode + ' finish=' + $oc.choices[0].finish_reason + ' ok=' + $okChat)
        if ($okChat) { break }
    } catch {
        $stc = -1
        try { $stc = [int]$_.Exception.Response.StatusCode } catch {}
        Log ('CHAT try=' + $try + ' HTTP=' + $stc + ' err=' + $_.Exception.Message)
    }
    Start-Sleep -Seconds 20
}

# ---- PHASE 6: serve log tail + decide ----
if (Test-Path -LiteralPath $serveLog) {
    Get-Content -LiteralPath $serveLog -Tail 30 | ForEach-Object { Log ('SVRLOG| ' + $_) }
} else { Log 'NO_SERVE_LOG' }

if ($okChat) {
    Log 'RESULT: PREFILL_FIX DEPLOYED, CERT PASS, SERVER HEALTHY (384k YaRN nvfp4)'
    exit 0
} else {
    Log 'RESULT: CHAT_FAILED: restoring previous exe + live command'
    Kill-Serve
    Copy-Item -LiteralPath $backup -Destination $desktopExe -Force
    Restore-Live
    exit 5
}
