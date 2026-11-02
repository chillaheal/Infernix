# M2 deepstack ablation, self-restoring (copy of image-spike/blackout_skinny.ps1; only the payload
# differs). The ablation loads the Qwen3-VL text encoder in BF16 on the same card the resident server
# holds at ~31 GiB, so it can only run while :8888 is down. The live command line is captured from the
# listening process before it is stopped and relaunched through explorer.exe at the end of every path,
# including the failure paths, exactly as the previous blackout scripts did.

param(
    [int]$GateMaxWaitSec = 900,      # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 25,  # how long this window announces that it holds the port
    [int]$GraceSec = 60,             # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 1800
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$workDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next'
$python     = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$ablate     = Join-Path $workDir 'image-edit\deepstack_ablate.py'
$deltaJson  = Join-Path $workDir 'image-edit\deepstack_delta.json'
$textEnc    = Join-Path $workDir 'image-spike\artifact\text_encoder'
$processor  = Join-Path $workDir 'image-spike\artifact\processor'
$gpuDiff    = Join-Path $workDir 'image-spike\gpu_diffusers'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_deepstack_restore.bat'
$outLog     = Join-Path $repo 'image-spike\deepstack_blackout.out.log'
$errLog     = Join-Path $repo 'image-spike\deepstack_blackout.err.log'
$runBat     = Join-Path $repo 'image-spike\_deepstack_run.bat'

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

foreach ($p in @($python, $ablate, $textEnc, $processor, $gpuDiff, $desktopDir)) {
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

$lines = @('@echo off',
           ('cd /d "' + $workDir + '"'),
           ('"' + $python + '" -I "' + $ablate + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
           'set TE=0',
           'if errorlevel 1 set TE=1',
           ('>>"' + $outLog + '" echo TESTEXIT=%TE%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the M2 deepstack ablation'
$t0 = Get-Date
if (Test-Path -LiteralPath $deltaJson) { Remove-Item -LiteralPath $deltaJson -Force }
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 10; $waited += 10 }
if (-not $tp.HasExited) { Log 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('test elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's')

$exit = 'NOTFOUND'
if (Test-Path -LiteralPath $outLog) {
    $m = Select-String -LiteralPath $outLog -Pattern 'TESTEXIT=(\d+)' | Select-Object -Last 1
    if ($m) { $exit = $m.Matches[0].Groups[1].Value }
    Get-Content -LiteralPath $outLog | ForEach-Object { Log ('OUT| ' + $_) }
}
if (Test-Path -LiteralPath $errLog) {
    Get-Content -LiteralPath $errLog -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) }
}
Log ('TEST_EXIT=' + $exit + ' delta_json=' + (Test-Path -LiteralPath $deltaJson))

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT deepstack=' + $exit + ' healthy=' + $healthy)
exit 0
