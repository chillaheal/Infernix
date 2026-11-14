# Re-run THE SHIP GATE on the merged tree (image-generation = 23091f5, which merges image-edit 995f41c).
# One payload: the M9 behavioural test, run as the shipped CTest runs it -- the artifact and the processor
# set, and NOTHING ELSE. The opt-in portrait pass stays dark because INFERNIX_EDIT_PORTRAIT and
# INFERNIX_EDIT_OUTDIR are deliberately not set, so this is the byte-for-byte gate that earned
# reference_image=true, not the look.
#
# It streams BF16 weights of the v4 artifact onto the same card the resident server holds at ~31 GiB, so it
# can only run while :8888 is down. The live command line is captured from the listening process before it
# is stopped and relaunched through explorer.exe at the end of EVERY path, including the failure paths.
# The test runs directly, not through ctest: ctest turns the test's own skip (77, artifact or processor
# unset) into a passing run, and a skip here would be a false green.
#
# Structure is image-spike\blackout_portrait_look.ps1, with the portrait env removed.

param(
    [int]$GateMaxWaitSec = 900,
    [int]$GateExpectedMinutes = 15,
    [int]$GraceSec = 60,
    [int]$TestTimeoutSec = 1800
)

$ErrorActionPreference = 'Continue'

$repo        = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$behaviorExe = Join-Path $repo 'build\tests\Release\infernix_image_edit_behavior_test.exe'
$artifact    = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v4.infernix'
$processor   = Join-Path $workDir 'image-spike\artifact\processor'
$vcpkgBin    = 'C:\vcpkg\installed\x64-windows\bin'
$vcvars      = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$desktopDir  = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port        = 8888
$restoreBat  = Join-Path $repo 'image-spike\_merged_gate_restore.bat'
$runBat      = Join-Path $repo 'image-spike\_merged_gate_run.bat'
$behLog      = Join-Path $repo 'image-spike\merged_gate_blackout.out.log'
$behErr      = Join-Path $repo 'image-spike\merged_gate_blackout.err.log'

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
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
function Read-Exit($log, $tag) {
    $exit = 'NOTFOUND'
    if (Test-Path -LiteralPath $log) {
        $m = Select-String -LiteralPath $log -Pattern 'TESTEXIT=(\d+)' | Select-Object -Last 1
        if ($m) { $exit = $m.Matches[0].Groups[1].Value }
        Get-Content -LiteralPath $log | ForEach-Object { Log ($tag + '| ' + $_) }
    }
    if ($exit -eq '77') { Log ($tag + ' TEST_SKIPPED: the artifact or processor was not seen -- NOT a pass') }
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

foreach ($p in @($behaviorExe, $artifact, $processor, $vcpkgBin, $vcvars, $desktopDir)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}

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

# Never run the payload beside a server that is still serving (the condition behind the 09:10 nvlddmkm
# bugcheck). Refuse the window instead.
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

# Only the two env vars the shipped CTest sets. The portrait pass is opt-in on two others; they must NOT
# appear in the written bat, which is checked below, so what runs here is the gate and nothing else.
$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           ('call "' + $vcvars + '" >nul'),
           ('set "PATH=%PATH%;' + $vcpkgBin + '"'),
           ('set "INFERNIX_TEST_ARTIFACT=' + $artifact + '"'),
           ('set "INFERNIX_QWENVL_PROCESSOR=' + $processor + '"'),
           'set TB=9',
           ('"' + $behaviorExe + '" > "' + $behLog + '" 2> "' + $behErr + '"'),
           'set TB=%errorlevel%',
           ('>>"' + $behLog + '" echo TESTEXIT=%TB%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)
$need = @('INFERNIX_TEST_ARTIFACT=', 'INFERNIX_QWENVL_PROCESSOR=', 'TESTEXIT=')
foreach ($n in $need) {
    if (-not (Select-String -LiteralPath $runBat -Pattern ([regex]::Escape($n)) -Quiet)) {
        Log ('RUNBAT_REFUSED: the written bat lost the line starting ' + $n)
        & $gate -Action EndWindow | Out-Null
        & $gate -Action Release | Out-Null
        exit 5
    }
}
foreach ($n in @('INFERNIX_EDIT_PORTRAIT=', 'INFERNIX_EDIT_OUTDIR=', 'INFERNIX_EDIT_PROMPT=')) {
    if (Select-String -LiteralPath $runBat -Pattern ([regex]::Escape($n)) -Quiet) {
        Log ('RUNBAT_REFUSED: the written bat carries ' + $n + ' -- this run must be the gate alone')
        & $gate -Action EndWindow | Out-Null
        & $gate -Action Release | Out-Null
        exit 6
    }
}

Log 'PHASE2 running the ship gate on the merged tree'
$t0 = Get-Date
foreach ($d in @($behLog, $behErr)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Force }
}
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 10; $waited += 10 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('tests elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

$behExit = Read-Exit $behLog 'GATE'
if (Test-Path -LiteralPath $behErr) {
    Get-Content -LiteralPath $behErr -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) }
}
Log ('TEST_EXIT behavior_gate=' + $behExit)

& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT behavior_gate=' + $behExit + ' healthy=' + $healthy)
exit 0
