# M4 conditioned-text-tower + M6 edit-step qualification, self-restoring (copy of
# image-spike/blackout_vision.ps1; the payload runs THREE executables -- the M3 vision tower as the chain's
# own regression, the M4 infernix_image_edit_prompt_cuda_test and the M6
# infernix_image_edit_pipeline_prod_cuda_test -- after producing the M6 torch one-step oracle on this
# window's GPU). They stream BF16 weights of the v4 artifact onto the same card
# the resident server holds at ~31 GiB, so they can only run while :8888 is down. The live command line
# is captured from the listening process before it is stopped and relaunched through explorer.exe at the
# end of every path, including the failure paths, exactly as the previous blackout scripts did.
#
# The payload runs the test executables directly instead of through ctest: ctest turns a 77 (the test's
# own "artifact/oracle unset" skip) into a passing run, and a skip here would be a false green.

param(
    [int]$GateMaxWaitSec = 900,      # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 20,  # how long this window announces that it holds the port
    [int]$GraceSec = 60,             # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 2400
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$visionExe  = Join-Path $repo 'build\tests\Release\infernix_image_vision_prod_cuda_test.exe'
$editExe    = Join-Path $repo 'build\tests\Release\infernix_image_edit_prompt_cuda_test.exe'
$artifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v4.infernix'
$oracle     = Join-Path $workDir 'image-spike\oracle'
$textDir    = Join-Path $oracle 'tensors_edit_text_tower'
$visionDir  = Join-Path $oracle 'tensors_edit_vision'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin'
$vcvars     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_edit_prompt_restore.bat'
$visionLog  = Join-Path $repo 'image-spike\vision_blackout12.out.log'
$visionErr  = Join-Path $repo 'image-spike\vision_blackout12.err.log'
$editLog    = Join-Path $repo 'image-spike\edit_prompt_blackout.out.log'
$editErr    = Join-Path $repo 'image-spike\edit_prompt_blackout.err.log'
$runBat     = Join-Path $repo 'image-spike\_edit_prompt_run.bat'
$stageDir   = Join-Path $repo 'image-spike\vision_stages_engine'

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
                # The resident LLM's model id is Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix; the id
                # carries no size suffix any more, so the old 'qwen3\.8-27b' pattern could never match
                # and Wait-Healthy reported FINAL_HEALTHY=False over a server that was serving fine.
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

foreach ($p in @($visionExe, $editExe, $artifact, $visionDir, $textDir, $vcpkgBin, $vcvars, $desktopDir,
                 $refBehavePy, $pythonExe)) {
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

# The safety net under that probe: never run the payload beside a server that is still serving. If
# the capture above saw pid=0 and /health still answers 200, nothing was stopped, and the six GPU
# exes would stream weights onto the card the resident LLM holds at ~31 GiB -- the condition behind
# the 09:10 nvlddmkm bugcheck. Refuse the window instead.
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

# vcvars + vcpkg bin lines copied from image-edit\build_vision_target.bat / run_host_image_tests.bat:
# the exes need the CUDA runtime DLLs and the zlib the artifact reader links.
# M6: the one-step torch oracle is produced INSIDE this window, on the GPU. The dump's own bat pins itself
# to CPU unless INFERNIX_DUMP_DEVICE=cuda says otherwise, because beside the live serve the card is not
# ours to take -- only a window like this one, where the server is down and drained, is. The new
# infernix_image_edit_pipeline_prod_cuda_test then reads the tensors that dump writes.
$dumpBat = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\run_dump_edit_step_gpu.bat'
$dumpLog = Join-Path $repo 'image-spike\edit_step_dump.out.log'
$dumpErr = Join-Path $repo 'image-spike\edit_step_dump.err.log'
$m6Exe   = Join-Path $repo 'build\tests\Release\infernix_image_edit_pipeline_prod_cuda_test.exe'
$m6Log   = Join-Path $repo 'image-spike\edit_pipeline_blackout.out.log'
$m6Err   = Join-Path $repo 'image-spike\edit_pipeline_blackout.err.log'
# M7: the ledger's measured-peak row. Needs only INFERNIX_TEST_ARTIFACT (the bat already sets it), runs
# the real pipeline on the real GPU, and asserts predicted >= measured on every case including the new
# edit route with the ViT lane.
$ledgerExe = Join-Path $repo 'build\tests\Release\infernix_image_need_ledger_real_test.exe'
$ledgerLog = Join-Path $repo 'image-spike\ledger_blackout.out.log'
$ledgerErr = Join-Path $repo 'image-spike\ledger_blackout.err.log'
# M9: the behavioural ship gate. It runs the whole edit route (render, tokenizer, ViT, conditioned
# tower, VAE encode, 20-step DiT) twice -- once per instruction -- and grades the PICTURES, so it needs
# the processor directory as well as the artifact. It is the longest item in the window; the timeout
# default covers it, and the caller can raise it with -TestTimeoutSec.
$processor   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
$behaviorExe = Join-Path $repo 'build\tests\Release\infernix_image_edit_behavior_test.exe'
$behaviorLog = Join-Path $repo 'image-spike\behavior_blackout.out.log'
$behaviorErr = Join-Path $repo 'image-spike\behavior_blackout.err.log'
# M9 CONTROL: image-edit\ref_edit_behavior.py grades the REFERENCE pipeline's pictures with the same
# three assertions and the same fixture, so an M9 red can be read as "this gate is unsatisfiable at
# 512/20/CFG-1" instead of "the engine is wrong" (blackout #37: A passed all three, B did not). It loads
# the shipped diffusers transformer + VAE onto THIS window's GPU, so it belongs here and nowhere else.
# Prints REF_BEHAVIOR_OK / REF_BEHAVIOR_FAIL and writes image-edit\evidence\ref_behave_{A,B}.png.
$refBehavePy  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\ref_edit_behavior.py'
$pythonExe    = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$refBehaveLog = Join-Path $repo 'image-spike\ref_behave_blackout.out.log'
$refBehaveErr = Join-Path $repo 'image-spike\ref_behave_blackout.err.log'
$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           ('call "' + $vcvars + '" >nul'),
           ('set "PATH=%PATH%;' + $vcpkgBin + '"'),
           ('set "INFERNIX_TEST_ARTIFACT=' + $artifact + '"'),
           ('set "INFERNIX_IMAGE_ORACLE=' + $oracle + '"'),
           ('set "INFERNIX_VISION_STAGE_DIR=' + $stageDir + '"'),
           ('set "INFERNIX_QWENVL_PROCESSOR=' + $processor + '"'),
           'set TD=9',
           'set TE1=9',
           'set TE2=9',
           'set TE3=9',
           'set TL=9',
           'set TB=9',
           'set TR=9',
           ('call "' + $dumpBat + '" > "' + $dumpLog + '" 2> "' + $dumpErr + '"'),
           'set TD=%errorlevel%',
           ('"' + $visionExe + '" > "' + $visionLog + '" 2> "' + $visionErr + '"'),
           'set TE1=%errorlevel%',
           ('"' + $editExe + '" > "' + $editLog + '" 2> "' + $editErr + '"'),
           'set TE2=%errorlevel%',
           ('"' + $m6Exe + '" > "' + $m6Log + '" 2> "' + $m6Err + '"'),
           'set TE3=%errorlevel%',
           ('"' + $ledgerExe + '" > "' + $ledgerLog + '" 2> "' + $ledgerErr + '"'),
           'set TL=%errorlevel%',
           ('"' + $behaviorExe + '" > "' + $behaviorLog + '" 2> "' + $behaviorErr + '"'),
           'set TB=%errorlevel%',
           ('"' + $pythonExe + '" -I "' + $refBehavePy + '" > "' + $refBehaveLog + '" 2> "' + $refBehaveErr + '"'),
           'set TR=%errorlevel%',
           ('>>"' + $dumpLog + '" echo TESTEXIT=%TD%'),
           ('>>"' + $visionLog + '" echo TESTEXIT=%TE1%'),
           ('>>"' + $editLog + '" echo TESTEXIT=%TE2%'),
           ('>>"' + $m6Log + '" echo TESTEXIT=%TE3%'),
           ('>>"' + $ledgerLog + '" echo TESTEXIT=%TL%'),
           ('>>"' + $behaviorLog + '" echo TESTEXIT=%TB%'),
           ('>>"' + $refBehaveLog + '" echo TESTEXIT=%TR%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the M3 vision regression and the M4 conditioned tower qualification'
$t0 = Get-Date
foreach ($d in @($stageDir, $visionLog, $visionErr, $editLog, $editErr, $dumpLog, $dumpErr, $m6Log, $m6Err,
                 $ledgerLog, $ledgerErr, $behaviorLog, $behaviorErr, $refBehaveLog, $refBehaveErr)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Recurse -Force }
}
New-Item -ItemType Directory -Path $stageDir | Out-Null
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 10; $waited += 10 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('tests elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

$dumpExit   = Read-Exit $dumpLog 'DMP'
$visionExit = Read-Exit $visionLog 'VIS'
$editExit   = Read-Exit $editLog 'EDT'
$m6Exit     = Read-Exit $m6Log 'M6P'
$ledgerExit   = Read-Exit $ledgerLog 'LED'
$behaviorExit = Read-Exit $behaviorLog 'BEH'
$refExit      = Read-Exit $refBehaveLog 'REF'
foreach ($e in @($dumpErr, $visionErr, $editErr, $m6Err, $ledgerErr, $behaviorErr, $refBehaveErr)) {
    if (Test-Path -LiteralPath $e) { Get-Content -LiteralPath $e -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) } }
}
Log ('TEST_EXIT dump=' + $dumpExit + ' vision=' + $visionExit + ' edit_prompt=' + $editExit +
     ' edit_pipeline=' + $m6Exit + ' ledger=' + $ledgerExit + ' behavior=' + $behaviorExit +
     ' ref_behave=' + $refExit)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT dump=' + $dumpExit + ' vision=' + $visionExit + ' edit_prompt=' + $editExit +
     ' edit_pipeline=' + $m6Exit + ' ledger=' + $ledgerExit + ' behavior=' + $behaviorExit +
     ' ref_behave=' + $refExit + ' healthy=' + $healthy)
exit 0
