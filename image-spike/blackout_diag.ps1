$ErrorActionPreference = 'Continue'

# Autonomous, self-restoring DIAGNOSTIC for the image-interlude restoration failure.
# Explorer-detached at every level so the surviving server is not reaped when this script exits.
# Flow: grace -> stop the :8888 owner -> drain -> start fresh fixed flash-next -> run the cat gen
# ONCE -> capture the full serve log (yield line, job outcome, EXACT restoration-failure text) ->
# ALWAYS stop+restart a fresh healthy flash-next so the harness reconnects.
# The gen degrades the server (submit door closes); that is expected -- we read the log, not the HTTP.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$serveOut   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_cycle.out.log'
$genLog     = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_diag_gen.log'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_diag_serve.bat'
$genPs1     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\gen_katt.ps1'
$png        = 'C:\Users\Micke\Desktop\Infernix FlashNext\maine-coon-katt.png'

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
    for ($i = 0; $i -lt 30; $i++) { if ((Listener-Pid) -eq 0) { break }; Start-Sleep -Seconds 2 }
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

# ---- serve arg string, verbatim from the production bat minus the leading exe token ----
$serveArgs = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveArgs = $ln.Substring('infernix-serve.exe'.Length).Trim(); break }
}
if (-not $serveArgs) { Log 'FATAL: no serve line found in start bat'; exit 2 }
Log ('ARGS_OK len=' + $serveArgs.Length)

$serveContent = "@echo off`r`ncd /d `"" + $desktopDir + "`"`r`ninfernix-serve.exe " + $serveArgs + ' > "' + $serveOut + '" 2>&1' + "`r`n"
[System.IO.File]::WriteAllText($serveBat, $serveContent, [System.Text.Encoding]::ASCII)

# ---- PHASE 0 grace: let the caller arm before we take the port ----
Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

# ---- PHASE 1 stop + drain, start fresh fixed flash-next ----
Stop-Listener
Log 'PHASE1 drained'
if (Test-Path -LiteralPath $serveOut) { Remove-Item -LiteralPath $serveOut -Force }
Launch-Bat $serveBat
Log 'PHASE1a launched flash-next'
$ok = Wait-Healthy 130
Log ('PHASE1b HEALTHY=' + $ok)
if (-not $ok) {
    Log 'NOT_UP: fresh clean restart so a server survives for the harness'
    Stop-Listener
    Launch-Bat $serveBat
    Log ('RETRY_HEALTHY=' + (Wait-Healthy 130))
    exit 5
}

# ---- PHASE 2 run the cat gen once (this may degrade the server; expected) ----
Log 'PHASE2 gen 768x768'
if (Test-Path -LiteralPath $png) { Remove-Item -LiteralPath $png -Force }
$g = (& $genPs1 -Size '768x768' -Out $png 2>&1) -join "`n"
Log ('GEN| ' + ($g -replace "`n", ' | '))
Log ('GEN_PNG=' + (Test-Path -LiteralPath $png))

# ---- PHASE 3 capture the FULL serve log BEFORE any restart overwrites it ----
if (Test-Path -LiteralPath $serveOut) { Copy-Item -LiteralPath $serveOut -Destination $genLog -Force }
Log 'PHASE3 serve log tail (interlude lifecycle)'
if (Test-Path -LiteralPath $serveOut) {
    Get-Content -LiteralPath $serveOut -Tail 120 | ForEach-Object { Log ('TAIL| ' + $_) }
    Select-String -LiteralPath $serveOut -Pattern 'interlude|yielded|freed|restoration|Degraded|submit door|out of memory|CUDA|OOM|image' -ErrorAction SilentlyContinue |
        Select-Object -Last 40 | ForEach-Object { Log ('GREP| ' + $_.Line) }
} else { Log 'NO_SERVE_LOG' }

# ---- PHASE 4 ALWAYS end on a fresh healthy server so the harness reconnects ----
Log 'PHASE4 final clean restart (self-heal for the harness)'
Stop-Listener
Launch-Bat $serveBat
$finalOk = Wait-Healthy 150
Log ('FINAL_HEALTHY=' + $finalOk)
Log ('RESULT png=' + (Test-Path -LiteralPath $png) + ' genlog=' + (Test-Path -LiteralPath $genLog))
exit 0
