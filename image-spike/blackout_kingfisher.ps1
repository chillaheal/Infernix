$ErrorActionPreference = 'Continue'

# SELF-RESTORING device run for the KV-shrink yield fix + the kingfisher image deliverable.
#   1. Deploy the freshly built (fixed) build-tree infernix-serve.exe onto the Desktop copy.
#   2. Restart to a fresh healthy server, then cross /v1/images/generations (the kingfisher) over an IDLE
#      LLM -- the exact near-idle crossing that previously hit "Paged KV: the pages past the new limit are
#      in use" and Degraded the engine.
#   3. EXPECT: the image GRANTS (PNG, HTTP 200) and the serve log NEVER shows Degraded.
#   4. ALWAYS self-heal to a fresh healthy (fixed) server so the harness reconnects.
# Explorer-detached at every level so the surviving server is not reaped when this script exits.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$serveOut   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_cycle.out.log'
$genLog     = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_kingfisher_gen.log'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_kf_serve.bat'
$genPs1     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\gen_prompt.ps1'
$png        = 'C:\Users\Micke\Desktop\kingfisher_parrot.png'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    return 0
}
function Launch-Bat($bat) { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $bat + '"') | Out-Null }
function Stop-Listener {
    $p = Listener-Pid
    if ($p -gt 0) { taskkill /F /T /PID $p 2>&1 | Out-Null }
    taskkill /F /IM infernix-serve.exe /T 2>&1 | Out-Null
    Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'start flash next.bat' } |
        ForEach-Object { taskkill /F /T /PID $_.ProcessId 2>&1 | Out-Null }
    for ($i = 0; $i -lt 60; $i++) {
        if (Get-Process infernix-serve -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 1; continue }
        try { $fs = [System.IO.File]::Open($desktopExe, 'Open', 'ReadWrite', 'None'); $fs.Close(); break } catch { Start-Sleep -Seconds 1 }
    }
    for ($i = 0; $i -lt 40; $i++) {
        $f = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
        if ($f -and [int]$f -lt 2000) { break }
        Start-Sleep -Seconds 3
    }
}
function Wait-Healthy($iters) {
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 4
            if ([int]$h.StatusCode -eq 200) {
                $m = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 4
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'Qwen3\.8-Flash-Next')) { return $true }
            }
        } catch { }
    }
    return $false
}

if (-not (Test-Path -LiteralPath $buildExe)) { Log ('FATAL build exe missing: ' + $buildExe); exit 2 }
if (-not (Test-Path -LiteralPath $genPs1)) { Log ('FATAL gen ps1 missing: ' + $genPs1); exit 2 }

# ---- serve arg string, verbatim from the production bat minus the leading exe token ----
$serveArgs = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveArgs = $ln.Substring('infernix-serve.exe'.Length).Trim(); break }
}
if (-not $serveArgs) { Log 'FATAL no serve line'; exit 2 }
$serveContent = "@echo off`r`ncd /d `"" + $desktopDir + "`"`r`ninfernix-serve.exe " + $serveArgs + ' > "' + $serveOut + '" 2>&1' + "`r`n"
[System.IO.File]::WriteAllText($serveBat, $serveContent, [System.Text.Encoding]::ASCII)

Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

# ---- PHASE1 deploy fixed exe, stop listener, start fresh fixed server ----
Stop-Listener
Log 'PHASE1 drained'
if (Test-Path -LiteralPath $serveOut) { Remove-Item -LiteralPath $serveOut -Force }
try { Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force; Log 'PHASE1a deployed fixed exe' } catch { Log ('DEPLOY_FAIL ' + $_.Exception.Message); exit 3 }
Launch-Bat $serveBat
$ok = Wait-Healthy 150
Log ('PHASE1b HEALTHY=' + $ok)
if (-not $ok) { Log 'NOT_UP final clean restart'; Stop-Listener; Launch-Bat $serveBat; Log ('RETRY_HEALTHY=' + (Wait-Healthy 150)); exit 5 }

# ---- PHASE2 generate the kingfisher (idle crossing; gen_prompt.ps1 default = kingfisher, 768x768) ----
if (Test-Path -LiteralPath $png) { Remove-Item -LiteralPath $png -Force }
$g = (& $genPs1 -Size '768x768' -Out $png 2>&1) -join "`n"
Log ('GEN| ' + ($g -replace "`n", ' | '))
Log ('GEN_PNG=' + (Test-Path -LiteralPath $png))

# ---- PHASE3 capture serve log (interlude lifecycle) ----
if (Test-Path -LiteralPath $serveOut) { Copy-Item -LiteralPath $serveOut -Destination $genLog -Force }
Log 'PHASE3 serve log'
if (Test-Path -LiteralPath $serveOut) {
    Get-Content -LiteralPath $serveOut -Tail 130 | ForEach-Object { Log ('TAIL| ' + $_) }
    Select-String -LiteralPath $serveOut -Pattern 'interlude|yielded|freed|restoration|Degraded|submit door|pages past|in use|frontier|out of memory|CUDA|OOM|KV shrink|frame resize' -ErrorAction SilentlyContinue |
        Select-Object -Last 60 | ForEach-Object { Log ('GREP| ' + $_.Line) }
} else { Log 'NO_SERVE_LOG' }
$pngOk = Test-Path -LiteralPath $png
$degr  = (Test-Path -LiteralPath $serveOut) -and (Select-String -LiteralPath $serveOut -Pattern 'Degraded' -Quiet -ErrorAction SilentlyContinue)
Log ('KF png=' + $pngOk + ' degraded=' + $degr)
if ($pngOk -and (-not $degr)) { Log 'PASS kingfisher granted, no degradation' } else { Log 'FAIL kingfisher did not grant cleanly' }

# ---- PHASE4 ALWAYS self-heal to a fresh healthy fixed server ----
Log 'PHASE4 final clean restart (self-heal)'
Stop-Listener
Launch-Bat $serveBat
Log ('FINAL_HEALTHY=' + (Wait-Healthy 150))
Log 'DONE'
exit 0
