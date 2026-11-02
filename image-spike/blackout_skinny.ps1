# The bundle A loose end, self-restoring: the BF16 skinny-GEMV (tokens <= 8) qualification could not be
# judged beside the live server (the card had ~1.5 GiB free and a wrong-looking result is then not
# evidence of anything), so stop the :8888 server, run the test on a GPU with nothing else on it, and
# relaunch the server with the exact command line it was running before.
#
# Copied from image-spike/blackout_bundle_a.ps1; only the test and the environment it needs differ.

param(
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 15,  # how long this window announces that it holds the port
    [int]$GraceSec = 60              # the exe is prebuilt here, so 15 keeps the downtime inside the retry horizon
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$testExe    = Join-Path $repo 'build\tests\Debug\infernix_image_skinny_linear_test.exe'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_skinny_restore.bat'
$outLog     = Join-Path $repo 'image-spike\skinny_blackout.out.log'
$errLog     = Join-Path $repo 'image-spike\skinny_blackout.err.log'
$runBat     = Join-Path $repo 'image-spike\_skinny_run.bat'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin;C:\vcpkg\installed\x64-windows\debug\bin'

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

foreach ($p in @($testExe, $desktopDir)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}
$env:Path = $vcpkgBin + ';' + $env:Path

# The window gate. :8888 carries every other Claude Code session's turns, so the port is taken only
# when it is quiet; the gate refuses (exit 10) unless it is, or unless -GateForce says otherwise.
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
           ('cd /d "' + $desktopDir + '"'),
           ('"' + $testExe + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
           'set TE=0',
           'if errorlevel 1 set TE=1',
           ('>>"' + $outLog + '" echo TESTEXIT=%TE%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the skinny linear qualification'
$t0 = Get-Date
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt 1200) { Start-Sleep -Seconds 10; $waited += 10 }
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
Log ('TEST_EXIT=' + $exit)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT skinny=' + $exit + ' healthy=' + $healthy)
exit 0
