# M9 CONTROL window: run image-edit\ref_edit_behavior.py -- the REFERENCE diffusers pipeline grading its
# own pictures with the engine's three behavioural assertions -- on a GPU with nothing else on it.
#
# Why it needs a window at all: the control loads the shipped BF16 QwenImage21 transformer + VAE onto the
# card, which the resident infernix-serve holds at ~31.6 GiB. Running it beside the live server is the
# condition behind the 09:10 nvlddmkm bugcheck, so it runs only while :8888 is down and drained. The
# 33 GB text encoder stays on the CPU (the script pins _get_qwen_prompt_embeds to it), so the card is
# left to the DiT.
#
# Copied from image-spike/blackout_skinny.ps1 -- the payload is the only difference -- but it takes the
# robust Listener-Pid and the "never run beside a live server" abort from blackout_edit_prompt.ps1:
# after the bugcheck the TCP state table can show the listener with a bogus state, so a blackout must
# not conclude from -State Listen that nothing is serving.
#
# The server is relaunched through explorer.exe on every path, including the failure paths, exactly as
# the other blackout scripts do. Nothing is built here and no engine exe is touched: this window carries
# Python only.

param(
    [int]$GateMaxWaitSec = 900,       # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 25,   # how long this window announces that it holds the port
    [int]$GraceSec = 60,              # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 2400
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$pythonExe  = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$refBehavePy = Join-Path $workDir 'image-edit\ref_edit_behavior.py'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_ref_behave_restore.bat'
$outLog     = Join-Path $repo 'image-spike\ref_behave_blackout.out.log'
$errLog     = Join-Path $repo 'image-spike\ref_behave_blackout.err.log'
$runBat     = Join-Path $repo 'image-spike\_ref_behave_run.bat'

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    # After the 09:10 nvlddmkm bugcheck the TCP state table lies: netstat shows the listening socket
    # as "127.0.0.1:8888 0.0.0.0:0 CLOSED 21032" while the server still answers /health, so
    # -State Listen finds nothing and a blackout would run its GPU payload beside the live server.
    # The listener is the row with no peer (RemotePort 0), whatever state it claims.
    $r = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
         Where-Object { [int]$_.RemotePort -eq 0 } | Select-Object -First 1
    if ($r) { return [int]$r.OwningProcess }
    return 0
}
function Gpu-UsedMib {
    $v = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ($v) { return [int]$v }
    return -1
}
function Launch-Bat($bat) { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $bat + '"') | Out-Null }
function Wait-Healthy($iters) {
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 4
            if ([int]$h.StatusCode -eq 200) {
                $m = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 6
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'qwen3\.8')) { return $true }
            }
        } catch { }
    }
    return $false
}

$liveCmd = ''
function Restore-Live {
    if (-not $liveCmd) { Log 'RESTORE: no live command was captured, nothing to relaunch'; return }
    $lines = @('@echo off', ('cd /d "' + $desktopDir + '"'), $liveCmd)
    [System.IO.File]::WriteAllLines($restoreBat, $lines, [System.Text.Encoding]::ASCII)
    Launch-Bat $restoreBat
    Log 'RESTORE: the captured live command was relaunched (detached, visible window)'
}

foreach ($p in @($pythonExe, $refBehavePy, $desktopDir)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}

# The window gate. :8888 carries every other Claude Code session's turns, so the port is taken only
# when it is quiet; the gate refuses (exit 10) unless it is, or unless the wait expires.
$gate = Join-Path $repo 'image-spike\window_gate.ps1'
if (-not (Test-Path -LiteralPath $gate)) { Log ('FATAL missing gate: ' + $gate); exit 2 }
& $gate -Action StartWindow -Owner (Split-Path -Leaf $MyInvocation.MyCommand.Path) -Reason (Split-Path -Leaf $MyInvocation.MyCommand.Path) -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }

Log ('PHASE0 grace ' + $GraceSec + 's (let any in-flight request finish before the server goes down)')
Start-Sleep -Seconds $GraceSec

$livePid = Listener-Pid
if ($livePid -gt 0) {
    try { $liveCmd = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $livePid)).CommandLine } catch { $liveCmd = '' }
}
Log ('PHASE1 live pid=' + $livePid + ' cmdlen=' + $liveCmd.Length)
if ($livePid -gt 0) { taskkill /F /T /PID $livePid 2>&1 | Out-Null }
for ($i = 0; $i -lt 60; $i++) {
    if (Get-Process infernix-serve -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 1; continue }
    break
}
for ($i = 0; $i -lt 40; $i++) {
    $u = Gpu-UsedMib
    if ($u -ge 0 -and $u -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log ('PHASE1 drained gpu_used_mib=' + (Gpu-UsedMib))

# The safety net under that probe: never run the payload beside a server that is still serving. If the
# capture above saw pid=0 and /health still answers 200, nothing was stopped, and torch would stream
# 20 GB of DiT weights onto the card the resident LLM holds -- refuse the window instead.
$srvUp = $false
try {
    $hs = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/health') -UseBasicParsing -TimeoutSec 4
    if ([int]$hs.StatusCode -eq 200) { $srvUp = $true }
} catch { }
if ($srvUp -or (Get-Process infernix-serve -ErrorAction SilentlyContinue)) {
    Log 'PHASE1 ABORT: infernix-serve is still serving -- refusing to run the GPU payload beside it'
    & $gate -Action EndWindow | Out-Null
    & $gate -Action Release | Out-Null
    exit 4
}

$lines = @('@echo off',
           ('cd /d "' + $workDir + '"'),
           ('set "PYTHONUNBUFFERED=1"'),
           'set TR=9',
           ('"' + $pythonExe + '" -I "' + $refBehavePy + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
           'set TR=%errorlevel%',
           ('>>"' + $outLog + '" echo TESTEXIT=%TR%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the reference behavioural control'
$t0 = Get-Date
foreach ($d in @($outLog, $errLog)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Force }
}
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 10; $waited += 10 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('test elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

$exit = 'NOTFOUND'
if (Test-Path -LiteralPath $outLog) {
    $m = Select-String -LiteralPath $outLog -Pattern 'TESTEXIT=(\d+)' | Select-Object -Last 1
    if ($m) { $exit = $m.Matches[0].Groups[1].Value }
    Get-Content -LiteralPath $outLog | ForEach-Object { Log ('OUT| ' + $_) }
}
if (Test-Path -LiteralPath $errLog) {
    Get-Content -LiteralPath $errLog -Tail 60 | ForEach-Object { Log ('ERR| ' + $_) }
}
Log ('TEST_EXIT ref_behave=' + $exit)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT ref_behave=' + $exit + ' healthy=' + $healthy)
exit 0
