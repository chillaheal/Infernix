# M4 + M9 window (image-edit plan Tasks 5 and 10). Three payloads in one blackout, in this order:
#
#   0. image-edit\dump_text_fp32fed_probe.py portrait -- the FP32-fed view of the PORTRAIT case, which no
#      oracle held until #47 (portrait fedview=0 / model_scatter=0: its floor never saw a third pass, so
#      it stayed the certified two-BF16-pass gate while small's got completed). CPU-only, but it runs the
#      36-layer encoder and needs the RAM the serve releases, so it cannot run beside it.
#
#   1. build\tests\Release\infernix_image_edit_prompt_cuda_test.exe -- the conditioned 36-layer tower.
#      Blackout #41 dumped the whole growth curve and it turned out to be measuring row 0: an
#      attention sink whose norm is 78x the mean row at layer 6 (197x at portrait) and which holds
#      99.8% of ||h||^2 from that layer onward. The per-layer aggregate divided by the FULL [S,D]
#      norm and selected ALL rows, so it was that row's gate -- and row 0 sits in the dropped system
#      prefix, which run_conditioned_text_tower never returns. The aggregate now selects the kept
#      rows (the contract, text_tower.h) and the sink row stays printed as a labelled diagnostic.
#   2. build\tests\Release\infernix_image_edit_behavior_test.exe -- the ship gate. Its second
#      instruction is now "Change the background to solid green", because the reference control
#      (image-edit\ref_edit_behavior.py, blackout #41) proved the checkpoint carries that one out
#      (field green=180.7 red=56.1 blue=35.7, mae=98.3, ssim=0.7406) and does NOT carry out either
#      bar instruction the plan named (top: mae=12.36 green=125.7; bottom: mae=14.45). An instruction
#      the reference cannot satisfy is not something the engine may be failed for.
#
# Deliberately absent: the torch curve dump. Its 36 layer files are on disk and the six certified
# tensors still hash to image-spike\oracle_precurve.sha256 (verified after #41); re-running the dump
# would overwrite the certified oracle directory again for no new information. Also absent: the
# Mem-FreeGb >= 36 abort, which existed for the 33 GB CPU text encoder that dump loads -- this window
# runs no CPU encoder, and if RAM were short the payload would fail visibly anyway. The rule that
# matters here is the other one: never run beside a live serve.
#
# Copied from image-spike\blackout_m4_curve.ps1 (the same robust Listener-Pid, the same Read-Exit per
# item, the same explorer.exe relaunch on every path). Nothing is built here.

param(
    [int]$GateMaxWaitSec = 900,       # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 45,   # how long this window announces that it holds the port
    [int]$GraceSec = 60,              # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 3600,
    # When set, the M4 test writes the ENGINE's 36 per-layer [S,hidden] snapshots there as raw float32
    # (test_image_edit_prompt_cuda.cu, INFERNIX_ENGINE_CURVE_DIR). #42 narrowed M4's red to two jumps
    # that land exactly on the layers where the reference's own one-layer step is large, which one
    # printed ratio per class cannot distinguish from an engine tensor sitting between two of the
    # reference's -- that needs the engine tensors themselves. Empty leaves the run as #42's.
    [string]$EngineCurveDir = '',
    # Task 10 Step 3's three remaining ship rows -- the VAE-encode cert, the M7 measured-peak ledger row,
    # and one text-only generation at oracle geometry -- appended to THIS window so the ship bit can be
    # judged with every row holding together, which is what the plan asks for. Without the switch the
    # window runs exactly what it ran before: the portrait dump, M4 and M9.
    [switch]$ShipRows
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$editExe    = Join-Path $repo 'build\tests\Release\infernix_image_edit_prompt_cuda_test.exe'
$behExe     = Join-Path $repo 'build\tests\Release\infernix_image_edit_behavior_test.exe'
$artifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v4.infernix'
$oracle     = Join-Path $workDir 'image-spike\oracle'
$textDir    = Join-Path $oracle 'tensors_edit_text_tower'
$visionDir  = Join-Path $oracle 'tensors_edit_vision'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin'
$vcvars     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_m4_m9_restore.bat'
$runBat     = Join-Path $repo 'image-spike\_m4_m9_run.bat'

$processor   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
$stageDir    = Join-Path $repo 'image-spike\vision_stages_engine'
# The two third-view directories. The M4 test probes every tensor in them by file existence, so a
# missing file leaves that gate byte-for-byte the certified two-view gate it has always been:
#   oracle_fp32fed_tower  the BF16 tower fed the FP32 ViT's rows (the completed image-row floor)
#   oracle_fp32_vision    the FP32 ViT's own merger/tap rows (the handoff's measured model scatter)
$f32fedDir   = Join-Path $workDir 'image-spike\oracle_fp32fed_tower'
$visionF32   = Join-Path $workDir 'image-spike\oracle_fp32_vision'
# The vae_encode ship row (Task 10 Step 3) skips with 77 unless its OWN oracle directory is set:
# test_image_vae_encode_prod_cuda.cu:94-100 returns 77 when INFERNIX_REF_ENCODE_ORACLE is absent, and
# :118 then reads the two legs oracle_ref_encode (portrait) and oracle_ref_encode_small under it. Both
# dirs hold ref_pixels/ref_lat_pre/ref_lat_norm/ref_lat_packed/latents_mean/latents_std + meta.json.
$refEncodeOracle = Join-Path $workDir 'image-edit\oracles'

$editLog  = Join-Path $repo 'image-spike\edit_prompt_blackout.out.log'
$editErr  = Join-Path $repo 'image-spike\edit_prompt_blackout.err.log'
$behLog   = Join-Path $repo 'image-spike\edit_behavior_blackout.out.log'
$behErr   = Join-Path $repo 'image-spike\edit_behavior_blackout.err.log'
# The portrait view of the FP32-fed tower, dumped in the SAME window as the test that reads it: the probe
# runs the 36-layer BF16 encoder on the CPU and needs the ~48 GB the serve holds (that is why the small
# one ran inside blackout_text_fp32fed.ps1), and #47 showed the portrait gate has no third view at all
# (fedview=0, model_scatter=0) -- it is still judged only by the two BF16-fed passes. Interpreter, -I and
# the log/echo convention copied from image-spike\blackout_text_fp32fed.ps1:39,163.
$pythonExe  = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$probePy    = Join-Path $workDir 'image-edit\dump_text_fp32fed_probe.py'
$probeLog   = Join-Path $repo 'image-spike\fp32fed_portrait_blackout.out.log'
$probeErr   = Join-Path $repo 'image-spike\fp32fed_portrait_blackout.err.log'
# The ship rows, target names copied from tests\models\qwen_image\tests.cmake (:238, :264, :254). They read
# the same environment the two above do (INFERNIX_TEST_ARTIFACT, INFERNIX_IMAGE_ORACLE, the stage dir and
# the processor), which is why they belong in this window's bat and not in a script of their own.
$shipTests = @()
if ($ShipRows) {
    foreach ($s in @(@('vae_encode',        'infernix_image_vae_encode_prod_cuda_test'),
                    @('ledger',            'infernix_image_need_ledger_real_test'),
                    @('pipeline_textonly', 'infernix_image_pipeline_prod_cuda_test'))) {
        $exe = Join-Path $repo ('build\tests\Release\' + $s[1] + '.exe')
        $lg  = Join-Path $repo ('image-spike\' + $s[0] + '_blackout.out.log')
        $er  = Join-Path $repo ('image-spike\' + $s[0] + '_blackout.err.log')
        $shipTests += [pscustomobject]@{ name = $s[0]; exe = $exe; log = $lg; err = $er
                                       ; run  = ('"' + $exe + '" > "' + $lg + '" 2> "' + $er + '"') }
    }
}

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
function Launch-Bat($bat) { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $bat + '"') | Out-Null }
function Wait-Healthy($iters) {
    # Blackout #45 printed FINAL_HEALTHY=False while the server answered /health 200 (checked right
    # after, PID alive, 31500 MiB), so a bare False told the user their Flash server was dead when it
    # was not. The last thing each endpoint did is now printed with the False.
    $lastH = 'never-reached'; $lastM = 'never-reached'
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 4
            $lastH = 'http=' + [int]$h.StatusCode
            if ([int]$h.StatusCode -eq 200) {
                $m = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 6
                $lastM = 'http=' + [int]$m.StatusCode
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'qwen3\.8')) { return $true }
                $lastM = $lastM + ' but no qwen3.8 model id'
            }
        } catch { $lastH = 'threw ' + $_.Exception.GetType().Name }
    }
    Log ('HEALTH last: /health ' + $lastH + ' | /v1/models ' + $lastM)
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

# One way out of a taken window, used by EVERY path after PHASE1. Blackout #46 killed the serve, hit
# its own run-bat guard, and exited 5 without restoring it -- the user restarted Flash by hand. A
# window that ends without bringing the server back is not allowed to exist.
function Leave-Window {
    & $gate -Action EndWindow | Out-Null
    Log 'LEAVE restoring the live server before releasing the window'
    Restore-Live
    $h = Wait-Healthy 180
    Log ('FINAL_HEALTHY=' + $h)
    & $gate -Action Release | Out-Null
    return $h
}

foreach ($p in @($editExe, $behExe, $artifact, $oracle, $textDir, $visionDir, $vcpkgBin, $vcvars,
                 $desktopDir, $processor, $pythonExe, $probePy, $refEncodeOracle)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}
foreach ($t in $shipTests) {
    if (-not (Test-Path -LiteralPath $t.exe)) { Log ('FATAL missing ship-row binary: ' + $t.exe); exit 2 }
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

# The safety net: never run the payload beside a server that is still serving. Running the DiT on top
# of the resident 31.6 GiB is the condition behind the 09:10 nvlddmkm bugcheck.
$srvUp = $false
try {
    $hs = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/health') -UseBasicParsing -TimeoutSec 4
    if ([int]$hs.StatusCode -eq 200) { $srvUp = $true }
} catch { }
if ($srvUp -or (Get-Process infernix-serve -ErrorAction SilentlyContinue)) {
    Log 'PHASE1 ABORT: infernix-serve is still serving -- refusing to run the payload beside it'
    & $gate -Action EndWindow | Out-Null
    # A server that survived PHASE1 is still the live one: relaunching it here would start a second
    # instance that cannot bind. Only put one back if the port is genuinely empty.
    if ((Listener-Pid) -eq 0) {
        Log 'ABORT: no listener left on :8888, restoring the captured live command'
        Restore-Live
        [void](Wait-Healthy 180)
    } else {
        Log 'ABORT: the live server still listens on :8888, nothing was relaunched'
    }
    & $gate -Action Release | Out-Null
    exit 4
}

# vcvars + vcpkg bin lines copied from image-edit\build_edit_behavior_target.bat and the other
# blackout scripts: the exes need the CUDA runtime DLLs and the zlib the artifact reader links.
# The curve line used to be interpolated as $(if ($EngineCurveDir) {...}) INSIDE this array literal.
# PowerShell 5.1 keeps a false-if subexpression there as a real $null element (reproduced: count stayed
# 19, element [12] was $null), and [System.IO.File]::WriteAllLines throws NullReferenceException on it.
# With ErrorActionPreference Continue that throw was a printed error, not a stop: blackout #45 went on
# to run the PREVIOUS window's bat, which is how the new floor printed fedview=0 / model_scatter=0.
# Optional lines are appended now, never interpolated.
# One variable, so the read-back guard below compares against the exact string that went into the bat.
$probeLine = ('"' + $pythonExe + '" -I "' + $probePy + '" portrait > "' + $probeLog + '" 2> "' + $probeErr + '"')
$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           ('call "' + $vcvars + '" >nul'),
           ('set "PATH=%PATH%;' + $vcpkgBin + '"'),
           ('set "INFERNIX_TEST_ARTIFACT=' + $artifact + '"'),
           ('set "INFERNIX_IMAGE_ORACLE=' + $oracle + '"'),
           ('set "INFERNIX_IMAGE_ORACLE_F32FED=' + $f32fedDir + '"'),
           ('set "INFERNIX_IMAGE_ORACLE_VISION_F32=' + $visionF32 + '"'),
           ('set "INFERNIX_REF_ENCODE_ORACLE=' + $refEncodeOracle + '"'),
           ('set "INFERNIX_VISION_STAGE_DIR=' + $stageDir + '"'),
           ('set "INFERNIX_QWENVL_PROCESSOR=' + $processor + '"'),
           'set TE=9',
           'set TB=9',
           'set TP=9')
if ($EngineCurveDir) { $lines += ('set "INFERNIX_ENGINE_CURVE_DIR=' + $EngineCurveDir + '"') }
$lines += @($probeLine,
            'set TP=%errorlevel%',
            # TESTEXIT=, not PROBEEXIT=: Read-Exit below keys on that pattern, and the probe is the third
            # payload of this window rather than a fourth thing to invent a convention for.
            ('>>"' + $probeLog + '" echo TESTEXIT=%TP%'),
            ('"' + $editExe + '" > "' + $editLog + '" 2> "' + $editErr + '"'),
            'set TE=%errorlevel%',
            ('"' + $behExe + '" > "' + $behLog + '" 2> "' + $behErr + '"'),
            'set TB=%errorlevel%',
            ('>>"' + $editLog + '" echo TESTEXIT=%TE%'),
            ('>>"' + $behLog + '" echo TESTEXIT=%TB%'))
# One reset/run/capture/echo quartet per requested ship row, appended the same way as the optional curve
# line -- never interpolated into a literal, which is what put a $null in front of WriteAllLines (#46).
$shipLines = @()
if ($ShipRows) {
    $si = 0
    foreach ($t in $shipTests) {
        $si++
        $v = 'TS' + $si
        $shipLines += ('set ' + $v + '=9')
        $shipLines += $t.run
        $shipLines += ('set ' + $v + '=%errorlevel%')
        $shipLines += ('>>"' + $t.log + '" echo TESTEXIT=%' + $v + '%')
    }
}
$lines += $shipLines
# Blackout #45 took the port and then ran the PREVIOUS window's bat: the file on disk still held
# INFERNIX_ENGINE_CURVE_DIR and none of the third-view variables, so the new floor printed
# fedview=0 / model_scatter=0 -- an environment that never happened, not a result. The write is now
# deleted-then-written and read back, and the window refuses to run a payload whose environment it
# cannot see in the file it just wrote.
$badRows = @($lines | Where-Object { ($null -eq $_) -or (-not ($_ -is [string])) })
if ($badRows.Count) {
    Log ('RUNBAT_ABORT: ' + $badRows.Count + ' non-string element(s) in the generated bat -- nothing was run')
    Leave-Window
    exit 6
}
if (Test-Path -LiteralPath $runBat) { Remove-Item -LiteralPath $runBat -Force }
try {
    [System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)
} catch {
    Log ('RUNBAT_WRITE_FAIL ' + $_.Exception.GetType().Name + ' :: ' + $_.Exception.Message + ' -- nothing was run')
    Leave-Window
    exit 6
}
$batBack = @()
try { $batBack = @(Get-Content -LiteralPath $runBat) } catch { $batBack = @() }
$need = @(('set "INFERNIX_IMAGE_ORACLE_F32FED=' + $f32fedDir + '"'),
          ('set "INFERNIX_IMAGE_ORACLE_VISION_F32=' + $visionF32 + '"'),
          ('set "INFERNIX_REF_ENCODE_ORACLE=' + $refEncodeOracle + '"'),
          # A bat left over from an earlier window would run the tests without ever dumping the portrait
          # view, and the gate would print fedview=0 as if the floor had been measured (#45's failure).
          $probeLine)
foreach ($t in $shipTests) { $need += $t.run }
$missing = @()
foreach ($n in $need) { if (-not ($batBack -contains $n)) { $missing += $n } }
$curveRows = @($batBack | Where-Object { $_ -like 'set "INFERNIX_ENGINE_CURVE_DIR=*' })
if ($EngineCurveDir -and -not $curveRows.Count) { $missing += 'the requested INFERNIX_ENGINE_CURVE_DIR' }
if (-not $EngineCurveDir -and $curveRows.Count) { $missing += ('a stale ' + $curveRows[0]) }
if ($missing.Count) {
    Log ('RUNBAT_REFUSED: the bat on disk does not hold ' + ($missing -join ' | ') + ' -- nothing was run')
    Log ('RUNBAT path=' + $runBat + ' lines=' + $batBack.Count +
         ' mtime=' + ((Get-Item -LiteralPath $runBat -ErrorAction SilentlyContinue).LastWriteTime))
    Leave-Window
    exit 5
}
Log ('RUNBAT ok lines=' + $batBack.Count + ' mtime=' + (Get-Item -LiteralPath $runBat).LastWriteTime.ToString('HH:mm:ss'))

Log 'PHASE2 running the conditioned tower (kept-row aggregate) then the behavioural ship gate'
$t0 = Get-Date
$allLogs = @($probeLog, $probeErr, $editLog, $editErr, $behLog, $behErr)
foreach ($t in $shipTests) { $allLogs += $t.log; $allLogs += $t.err }
foreach ($d in $allLogs) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Force }
}
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 15; $waited += 15 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('payloads elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

$probeExit = Read-Exit $probeLog 'FPR'
$editExit = Read-Exit $editLog 'EDT'
$behExit  = Read-Exit $behLog  'BEH'
$shipExits = @()
$sn = 0
foreach ($t in $shipTests) {
    $sn++
    $shipExits += ($t.name + '=' + (Read-Exit $t.log ('SHP' + $sn)))
}
$allErrs = @($probeErr, $editErr, $behErr)
foreach ($t in $shipTests) { $allErrs += $t.err }
foreach ($e in $allErrs) {
    if (Test-Path -LiteralPath $e) { Get-Content -LiteralPath $e -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) } }
}
Log ('TEST_EXIT edit_prompt=' + $editExit + ' edit_behavior=' + $behExit)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
Log 'PHASE3 restoring the live server'
$healthy = Leave-Window
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
# Built as a variable, never interpolated: a $(if ...) inside an argument is what handed WriteAllLines a
# $null element in #46, and there is no reason to keep the shape around after that.
$shipPart = ''
if ($shipExits.Count) { $shipPart = ' ship[' + ($shipExits -join ' ') + ']' }
Log ('RESULT probe_portrait=' + $probeExit + ' edit_prompt=' + $editExit +
     ' edit_behavior=' + $behExit + ' healthy=' + $healthy + $shipPart)
exit 0
