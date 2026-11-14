$ErrorActionPreference = 'Continue'

# One-shot self-restoring cycle, explorer-detached at every level so the new server is not
# reaped when the script exits (persistent tree: infernix-serve -> cmd -> explorer):
#   stop whatever owns :8888 -> start the FIXED flash-next (via a bat that does cd /d so the
#   relative --request-log-jsonl requests.jsonl resolves, and redirects to a log) -> verify it
#   is the flash-next model -> leave it running. On failure, relaunch the captured live command.
# No build step (the fixed exe is already the desktop binary).

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$serveOut   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_cycle.out.log'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_yield_serve.bat'
$restoreBat = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_yield_restore.bat'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

# ---- serve arg string: verbatim from the production bat, minus the leading exe token ----
$args = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $args = $ln.Substring('infernix-serve.exe'.Length).Trim(); break }
}
if (-not $args) { Log 'FATAL: no serve line found in start bat'; exit 2 }
Log ('ARGS_OK len=' + $args.Length)

# ---- capture the current :8888 listener (for self-restore) ----
$c = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$livePid = 0; $liveCmd = ''
if ($c) {
    $livePid = [int]$c.OwningProcess
    try { $liveCmd = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $livePid)).CommandLine } catch { $liveCmd = '' }
}
Log ('LIVE pid=' + $livePid + ' cmdlen=' + $liveCmd.Length)

# ---- launch a bat detached via explorer.exe (parent = explorer => survives script exit) ----
function Launch-Bat($bat) { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $bat + '"') | Out-Null }

# write the serve bat: cd /d + serve line + log redirect. Single controlled string (CRLF),
# written with WriteAllText to avoid any line-splitting of the cd line.
$serveContent = "@echo off`r`ncd /d `"" + $desktopDir + "`"`r`ninfernix-serve.exe " + $args + ' > "' + $serveOut + '" 2>&1' + "`r`n"
[System.IO.File]::WriteAllText($serveBat, $serveContent, [System.Text.Encoding]::ASCII)

function Restore-Live {
    if (-not $liveCmd) { Log 'RESTORE: no live command captured (nothing was running)'; return }
    $rc = "@echo off`r`ncd /d `"" + $desktopDir + "`"`r`n" + $liveCmd + "`r`n"
    [System.IO.File]::WriteAllText($restoreBat, $rc, [System.Text.Encoding]::ASCII)
    Launch-Bat $restoreBat
    Log 'RESTORE: live command relaunched (explorer-detached)'
}

# ---- grace: give the caller time to arm its wait before we take the port ----
Log 'CYCLE grace 30s'
Start-Sleep -Seconds 30

# ---- stop the current listener and wait for port + GPU to free ----
if ($livePid -gt 0) {
    taskkill /F /T /PID $livePid 2>&1 | Out-Null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        if (-not (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue)) { break }
    }
}
for ($i = 0; $i -lt 40; $i++) {
    $f = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if ($f -and [int]$f -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log 'DRAINED port+gpu'

# ---- start the fixed flash-next, explorer-detached, with a log we can read after ----
if (Test-Path -LiteralPath $serveOut) { Remove-Item -LiteralPath $serveOut -Force }
Launch-Bat $serveBat
Log 'LAUNCHED flash-next (explorer-detached, logging to serve_out)'

# ---- wait healthy: /health 200 AND /v1/models reports the flash-next model ----
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 5
    try {
        $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 4
        if ([int]$h.StatusCode -eq 200) {
            $m = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 4
            if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'Qwen3\.8-Flash-Next')) { $ok = $true; break }
        }
    } catch { }
}
Log ('HEALTHY=' + $ok)

if (-not $ok) {
    Log 'NOT_UP: restoring live server'
    Start-Sleep -Seconds 5
    Restore-Live
    exit 5
}

Log 'CYCLE_OK leaving flash-next (fixed) running'
exit 0
