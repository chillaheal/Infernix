# M4 growth-curve window (image-edit plan Task 5). Three payloads in one blackout, in this order:
#
#   1. image-edit\dump_edit_text_oracle.py with INFERNIX_TEXT_CURVE=1 -- the torch anchor capturing
#      EVERY decoder layer's input, not only the three the deepstack adds land on. Blackout #37/#38 left
#      the FINAL image rows at eng 0.2129 against a band of 0.1362 (cos 0.12 -- the test's own semantics
#      call a low cos with a high ratio "a mechanism of the engine's own") while the text rows stayed
#      inside theirs. Three layers cannot say whether that error is born at layer 0 (a wrong scatter or
#      add) or accumulates with the text rows (the BF16 depth of the tower); 36 can.
#      It is a CPU job (the script's own header: no CUDA call, no device_map, device="cpu" everywhere),
#      but it loads the 33 GB BF16 text encoder into RAM -- and the resident infernix-serve holds ~48 GB
#      of the 64 GB. So it runs here, where the server is down, and nowhere else.
#   2. build\tests\Release\infernix_image_edit_prompt_cuda_test.exe -- which now sizes its snapshot
#      request to the whole tower when {case}_layer03_hidden_f32.npy exists, and prints rel_fro / cos /
#      gain per row class per layer. The three certified layers stay FATAL; the curve beyond them is
#      diagnostic so one layer cannot hide the shape of the rest.
#   3. image-edit\ref_edit_behavior.py -- the M9 reference control (the local pipeline's own __call__
#      grading its own pictures with the engine's three behavioural assertions). Blackout #39 got both
#      generations through and died in the grading helper on an RGBA raster; that is fixed, so this is
#      the first run that can say whether instruction B is satisfiable at all at 512/20/CFG-1.
#
# Copied from image-spike\blackout_edit_prompt.ps1 (the same robust Listener-Pid, the same "never run
# beside a live server" abort, the same Read-Exit per item and the same explorer.exe relaunch on every
# path). Nothing is built here -- the build ran before this window, in its own visible tab.

param(
    [int]$GateMaxWaitSec = 900,       # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 45,   # how long this window announces that it holds the port
    [int]$GraceSec = 60,              # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 3600
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$editExe    = Join-Path $repo 'build\tests\Release\infernix_image_edit_prompt_cuda_test.exe'
$artifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v4.infernix'
$oracle     = Join-Path $workDir 'image-spike\oracle'
$textDir    = Join-Path $oracle 'tensors_edit_text_tower'
$visionDir  = Join-Path $oracle 'tensors_edit_vision'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin'
$vcvars     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_m4_curve_restore.bat'
$runBat     = Join-Path $repo 'image-spike\_m4_curve_run.bat'

$pythonExe   = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$dumpTextPy  = Join-Path $workDir 'image-edit\dump_edit_text_oracle.py'
$refBehavePy = Join-Path $workDir 'image-edit\ref_edit_behavior.py'
$processor   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
$stageDir    = Join-Path $repo 'image-spike\vision_stages_engine'

$curveLog = Join-Path $repo 'image-spike\text_curve_dump.out.log'
$curveErr = Join-Path $repo 'image-spike\text_curve_dump.err.log'
$editLog  = Join-Path $repo 'image-spike\edit_prompt_blackout.out.log'
$editErr  = Join-Path $repo 'image-spike\edit_prompt_blackout.err.log'
$refLog   = Join-Path $repo 'image-spike\ref_behave_blackout.out.log'
$refErr   = Join-Path $repo 'image-spike\ref_behave_blackout.err.log'

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

foreach ($p in @($editExe, $artifact, $oracle, $textDir, $visionDir, $vcpkgBin, $vcvars, $desktopDir,
                 $pythonExe, $dumpTextPy, $refBehavePy, $processor)) {
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

# The safety net: never run the payload beside a server that is still serving. The curve dump would put
# a 33 GB CPU model on top of the resident LLM's ~48 GB, and the GPU items would stream BF16 weights onto
# the card it holds -- both are the condition behind the 09:10 nvlddmkm bugcheck.
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
    Log ('PHASE1 ABORT: only ' + (Mem-FreeGb) + ' GB free -- the CPU text-encoder dump needs ~36, refuse the window')
    & $gate -Action EndWindow | Out-Null
    & $gate -Action Release | Out-Null
    exit 5
}

# vcvars + vcpkg bin lines copied from image-edit\build_edit_prompt_target.bat / the other blackout
# scripts: the exe needs the CUDA runtime DLLs and the zlib the artifact reader links.
$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           ('call "' + $vcvars + '" >nul'),
           ('set "PATH=%PATH%;' + $vcpkgBin + '"'),
           ('set "INFERNIX_TEST_ARTIFACT=' + $artifact + '"'),
           ('set "INFERNIX_IMAGE_ORACLE=' + $oracle + '"'),
           ('set "INFERNIX_VISION_STAGE_DIR=' + $stageDir + '"'),
           ('set "INFERNIX_QWENVL_PROCESSOR=' + $processor + '"'),
           'set "INFERNIX_TEXT_CURVE=1"',
           'set TC=9',
           'set TE=9',
           'set TR=9',
           ('"' + $pythonExe + '" -I "' + $dumpTextPy + '" > "' + $curveLog + '" 2> "' + $curveErr + '"'),
           'set TC=%errorlevel%',
           ('"' + $editExe + '" > "' + $editLog + '" 2> "' + $editErr + '"'),
           'set TE=%errorlevel%',
           ('"' + $pythonExe + '" -I "' + $refBehavePy + '" > "' + $refLog + '" 2> "' + $refErr + '"'),
           'set TR=%errorlevel%',
           ('>>"' + $curveLog + '" echo TESTEXIT=%TC%'),
           ('>>"' + $editLog + '" echo TESTEXIT=%TE%'),
           ('>>"' + $refLog + '" echo TESTEXIT=%TR%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 dumping the 36-layer torch anchor, then the M4 curve, then the M9 reference control'
$t0 = Get-Date
foreach ($d in @($curveLog, $curveErr, $editLog, $editErr, $refLog, $refErr)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Force }
}
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 15; $waited += 15 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('payloads elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

$curveExit  = Read-Exit $curveLog 'CRV'
$editExit   = Read-Exit $editLog  'EDT'
$refExit    = Read-Exit $refLog   'REF'
foreach ($e in @($curveErr, $editErr, $refErr)) {
    if (Test-Path -LiteralPath $e) { Get-Content -LiteralPath $e -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) } }
}
Log ('TEST_EXIT curve_dump=' + $curveExit + ' edit_prompt=' + $editExit + ' ref_behave=' + $refExit)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT curve_dump=' + $curveExit + ' edit_prompt=' + $editExit + ' ref_behave=' + $refExit +
     ' healthy=' + $healthy)
exit 0
