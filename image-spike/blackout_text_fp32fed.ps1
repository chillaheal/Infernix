# M4 seed window (image-edit plan Task 5). ONE payload:
#
#   image-edit\dump_text_fp32fed_probe.py -- the 36-layer BF16 tower over the small fixture run twice,
#   once fed the certified BF16 merger rows and once fed the FP32 ones from image-spike\oracle_fp32_vision,
#   to measure the floor the certified oracle cannot: both of its passes get the SAME BF16 picture rows
#   (dump_edit_text_oracle.py:494-496 asserts zero difference), so the two-pass spread the failing image
#   gate divides by contains no vision term while the engine's image rows arrive carrying the ViT.
#
# Why it needs the window: the probe is CPU-only (no CUDA call, no device_map, no :8888) but it loads the
# 33 GB BF16 text encoder, and infernix-serve holds ~48 GB of the 64 GB beside its 31.6 GiB of VRAM. The
# payload aborts below if the server is still up or the free RAM is under 36 GB -- the condition behind
# the 09:10 nvlddmkm bugcheck is never accepted.
#
# Structure copied from image-spike\blackout_m4_curve.ps1: the same robust Listener-Pid (the TCP table
# lied after the bugcheck), the same never-beside-a-live-server aborts, the same Read-Exit per payload,
# the same explorer.exe relaunch of the captured live command on every path. Nothing is built here.

param(
    [int]$GateMaxWaitSec = 900,       # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 30,   # how long this window announces that it holds the port
    [int]$GraceSec = 60,              # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 3600
)

$ErrorActionPreference = 'Continue'

$repo        = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$oracle      = Join-Path $workDir 'image-spike\oracle'
$textDir     = Join-Path $oracle 'tensors_edit_text_tower'
$visionDir   = Join-Path $oracle 'tensors_edit_vision'
$visionF32   = Join-Path $workDir 'image-spike\oracle_fp32_vision'
$outDir      = Join-Path $workDir 'image-spike\oracle_fp32fed_tower'
$desktopDir  = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port        = 8888
$restoreBat  = Join-Path $repo 'image-spike\_fp32fed_restore.bat'
$runBat      = Join-Path $repo 'image-spike\_fp32fed_run.bat'

$pythonExe   = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$probePy     = Join-Path $workDir 'image-edit\dump_text_fp32fed_probe.py'
$harnessPy   = Join-Path $workDir 'image-edit\dump_edit_text_oracle.py'
$gpuDiffusers = Join-Path $workDir 'image-spike\gpu_diffusers'
$processor   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'

$probeLog    = Join-Path $repo 'image-spike\text_fp32fed.out.log'
$probeErr    = Join-Path $repo 'image-spike\text_fp32fed.err.log'

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    # After the 09:10 nvlddmkm bugcheck the TCP state table lies: netstat shows the listening socket
    # as "127.0.0.1:8888 0.0.0.0:0 CLOSED 21032" while the server still answers /health, so
    # -State Listen finds nothing and a blackout would run its payload beside the live server.
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
function Mem-FreeGb {
    $o = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($o) { return [math]::Round(1.0 * $o.FreePhysicalMemory / 1048576.0, 1) }
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
function Read-Exit($log, $tag) {
    $exit = 'NOTFOUND'
    if (Test-Path -LiteralPath $log) {
        $m = Select-String -LiteralPath $log -Pattern 'TESTEXIT=(\d+)' | Select-Object -Last 1
        if ($m) { $exit = $m.Matches[0].Groups[1].Value }
        Get-Content -LiteralPath $log | ForEach-Object { Log ($tag + '| ' + $_) }
    }
    if ($exit -eq '77') { Log ($tag + ' TEST_SKIPPED: the artifact or oracle was not seen -- this is NOT a pass') }
    return $exit
}

$liveCmd = ''
function Restore-Live {
    if (-not $liveCmd) { Log 'RESTORE: no live command was captured, nothing to relaunch'; return }
    $lines = @('@echo off', ('cd /d "' + $desktopDir + '"'), $liveCmd)
    [System.IO.File]::WriteAllLines($restoreBat, $lines, [System.Text.Encoding]::ASCII)
    Launch-Bat $restoreBat
    Log 'RESTORE: the captured live command was relaunched (detached, visible window)'
}

foreach ($p in @($oracle, $textDir, $visionDir, $visionF32, $desktopDir, $pythonExe, $probePy,
                 $harnessPy, $gpuDiffusers, $processor)) {
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
Log ('PHASE1 drained gpu_used_mib=' + (Gpu-UsedMib) + ' mem_free_gb=' + (Mem-FreeGb))

# The safety net: never run the payload beside a server that is still serving. The dump would put a
# 33 GB CPU model on top of the resident LLM's ~48 GB -- both are the condition behind the 09:10
# nvlddmkm bugcheck.
$srvUp = $false
try {
    $hs = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/health') -UseBasicParsing -TimeoutSec 4
    if ([int]$hs.StatusCode -eq 200) { $srvUp = $true }
} catch { }
if ($srvUp -or (Get-Process infernix-serve -ErrorAction SilentlyContinue)) {
    Log 'PHASE1 ABORT: infernix-serve is still serving -- refusing to run the payload beside it'
    & $gate -Action EndWindow | Out-Null
    & $gate -Action Release | Out-Null
    exit 4
}
# The CPU half of the same reason, stated as a number: without the ~48 GB the server releases, the
# 33 GB text encoder cannot be loaded and the dump would die in the allocator instead of the model.
if ((Mem-FreeGb) -lt 36) {
    Log ('PHASE1 ABORT: only ' + (Mem-FreeGb) + ' GB free -- the CPU text-encoder probe needs ~36, refuse the window')
    & $gate -Action EndWindow | Out-Null
    & $gate -Action Release | Out-Null
    exit 5
}

$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           'set TP=9',
           ('"' + $pythonExe + '" -I "' + $probePy + '" > "' + $probeLog + '" 2> "' + $probeErr + '"'),
           'set TP=%errorlevel%',
           ('>>"' + $probeLog + '" echo TESTEXIT=%TP%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the FP32-fed conditioned-tower probe'
$t0 = Get-Date
foreach ($d in @($probeLog, $probeErr)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Force }
}
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 15; $waited += 15 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('payload elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

$probeExit = Read-Exit $probeLog 'F32'
if (Test-Path -LiteralPath $probeErr) {
    Get-Content -LiteralPath $probeErr -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) }
}
Log ('TEST_EXIT fp32fed_probe=' + $probeExit)
if (Test-Path -LiteralPath $outDir) {
    Get-ChildItem -LiteralPath $outDir | ForEach-Object { Log ('OUT| ' + $_.Name + ' ' + $_.Length) }
}

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT fp32fed_probe=' + $probeExit + ' healthy=' + $healthy)
exit 0
