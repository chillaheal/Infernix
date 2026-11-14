# M3 vision-tower qualification, self-restoring (copy of image-spike/blackout_deepstack.ps1; only the
# payload differs). infernix_image_vision_prod_cuda_test streams the BF16 vision weights of the v4
# artifact onto the same card the resident server holds at ~31 GiB, so it can only run while :8888 is
# down. The live command line is captured from the listening process before it is stopped and
# relaunched through explorer.exe at the end of every path, including the failure paths, exactly as the
# previous blackout scripts did.
#
# The payload runs the test executable directly instead of through ctest: ctest turns a 77 (the test's
# own "artifact/oracle unset" skip) into a passing run, and a skip here would be a false green.

param(
    [int]$GateMaxWaitSec = 900,      # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 20,  # how long this window announces that it holds the port
    [int]$GraceSec = 60,             # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 1800
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$testExe    = Join-Path $repo 'build\tests\Release\infernix_image_vision_prod_cuda_test.exe'
$artifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v4.infernix'
$oracle     = Join-Path $workDir 'image-spike\oracle'
$visionDir  = Join-Path $oracle 'tensors_edit_vision'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin'
$vcvars     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_vision_restore.bat'
$outLog     = Join-Path $repo 'image-spike\vision_blackout.out.log'
$errLog     = Join-Path $repo 'image-spike\vision_blackout.err.log'
$runBat     = Join-Path $repo 'image-spike\_vision_run.bat'
$stageDir   = Join-Path $repo 'image-spike\vision_stages_engine'

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
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
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'Qwen3\.8-Flash-Next')) { return $true }
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

foreach ($p in @($testExe, $artifact, $visionDir, $vcpkgBin, $vcvars, $desktopDir)) {
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

# vcvars + vcpkg bin lines copied from image-edit\build_vision_target.bat / run_host_image_tests.bat:
# the exe needs the CUDA runtime DLLs and the zlib the artifact reader links.
$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           ('call "' + $vcvars + '" >nul'),
           ('set "PATH=%PATH%;' + $vcpkgBin + '"'),
           ('set "INFERNIX_TEST_ARTIFACT=' + $artifact + '"'),
           ('set "INFERNIX_IMAGE_ORACLE=' + $oracle + '"'),
           ('set "INFERNIX_VISION_STAGE_DIR=' + $stageDir + '"'),
           ('set TE=9'),
           ('"' + $testExe + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
           'set TE=%errorlevel%',
           ('>>"' + $outLog + '" echo TESTEXIT=%TE%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the M3 vision tower qualification'
$t0 = Get-Date
if (Test-Path -LiteralPath $stageDir) { Remove-Item -LiteralPath $stageDir -Recurse -Force }
New-Item -ItemType Directory -Path $stageDir | Out-Null
if (Test-Path -LiteralPath $outLog) { Remove-Item -LiteralPath $outLog -Force }
if (Test-Path -LiteralPath $errLog) { Remove-Item -LiteralPath $errLog -Force }
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
    Get-Content -LiteralPath $errLog -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) }
}
if ($exit -eq '77') { Log 'TEST_SKIPPED: the artifact or oracle was not seen -- this is NOT a pass' }
Log ('TEST_EXIT=' + $exit)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT vision=' + $exit + ' healthy=' + $healthy)
exit 0
