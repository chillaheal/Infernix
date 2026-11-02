param(
    [string]$Only = '',              # 'ledger', 'prod' or 'failures' to run just one cert
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 60,  # three certs plus the restore, announced to whoever is waiting
    [int]$GraceSec = 60              # keep short when the cert exe is already built: less time with :8888 down
)

$ErrorActionPreference = 'Continue'

# Bundle A device certification, self-restoring: stop the live :8888 server, run the three image
# interlude certifications against the current Debug build, then relaunch the server with the exact
# command line it was running before. The tests need the GPU to themselves, which is why the server
# has to be down (the cert files' own headers say the same).
#
#   ledger    calibrates the weight-resident allowance the computed image need uses
#   prod      certified route + the new probe cases (the probe never disturbs a running Program)
#   failures  S1..S7, with S6 the new soft KV remap and S7 the terminal restoration fault LAST

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$testDir    = Join-Path $repo 'build\tests\Debug'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port       = 8888
$modelDir   = Join-Path $desktopDir 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact   = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
# The ngram volume on disk does NOT carry the Dense8 infix the live server's own command line names,
# and an Engine start with a missing ngram path is fatal, so this is the file that actually exists.
$ngram      = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$imageArtifact    = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$imageOracle      = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle'
$imageOraclePipe  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle_512_mainecoon\tensors_pipeline'
$restoreBat = Join-Path $repo 'image-spike\_bundleA_restore.bat'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin;C:\vcpkg\installed\x64-windows\debug\bin'

# Write-Host (the information stream) rather than the success stream: Run-Test returns the test's exit
# code, and a log line leaking into that return value would corrupt it.
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

$env:Path = $vcpkgBin + ';' + $env:Path
$env:INFERNIX_QWEN4_ARTIFACT      = $artifact
if (Test-Path -LiteralPath $ngram) { $env:INFERNIX_QWEN4_NGRAM = $ngram } else { Log ('NGRAM_MISSING ' + $ngram) }
$env:INFERNIX_TEST_ARTIFACT       = $imageArtifact
$env:INFERNIX_IMAGE_ORACLE        = $imageOracle
$env:INFERNIX_IMAGE_ORACLE_PIPELINE = $imageOraclePipe

foreach ($p in @($artifact, $imageArtifact, $imageOracle, $testDir)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}

function Run-Test($exe, $tag, $maxWait) {
    $outLog = Join-Path $desktopDir ('bundleA_' + $tag + '.out.log')
    $errLog = Join-Path $desktopDir ('bundleA_' + $tag + '.err.log')
    $runBat = Join-Path $desktopDir ('bundleA_' + $tag + '_run.bat')
    if (-not (Test-Path -LiteralPath $exe)) { Log ('TEST_EXE_MISSING ' + $exe); return 'MISSING' }
    if (Test-Path -LiteralPath $outLog) { Remove-Item -LiteralPath $outLog -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $errLog) { Remove-Item -LiteralPath $errLog -Force -ErrorAction SilentlyContinue }
    $lines = @('@echo off',
               ('cd /d "' + $desktopDir + '"'),
               ('"' + $exe + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
               'set TE=0',
               'if errorlevel 1 set TE=1',
               ('>>"' + $outLog + '" echo TESTEXIT=%TE%'))
    [System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)
    Log ('RUN ' + $tag)
    $t0 = Get-Date
    $tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
    $waited = 0
    while (-not $tp.HasExited -and $waited -lt $maxWait) { Start-Sleep -Seconds 10; $waited += 10 }
    if (-not $tp.HasExited) { Log ('TEST_TIMEOUT_KILLING ' + $tag); try { $tp.Kill() } catch { } }
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds
    $exit = 'NOTFOUND'
    if (Test-Path -LiteralPath $outLog) {
        $m = Select-String -LiteralPath $outLog -Pattern 'TESTEXIT=(\d+)' | Select-Object -Last 1
        if ($m) { $exit = $m.Matches[0].Groups[1].Value }
    }
    Log ('TEST_EXIT[' + $tag + ']=' + $exit + ' elapsed=' + $elapsed + 's')
    if (Test-Path -LiteralPath $outLog) {
        Get-Content -LiteralPath $outLog | ForEach-Object { Log ('OUT| ' + $tag + '| ' + $_) }
    }
    if (Test-Path -LiteralPath $errLog) {
        Get-Content -LiteralPath $errLog -Tail 40 | ForEach-Object { Log ('ERR| ' + $tag + '| ' + $_) }
    }
    return $exit
}

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

function Wanted($tag) { return ($Only -eq '' -or $Only -eq $tag) }
$ledger = 'SKIPPED'; $prod = 'SKIPPED'; $failures = 'SKIPPED'
if (Wanted 'ledger')   { $ledger   = Run-Test (Join-Path $testDir 'infernix_image_need_ledger_real_test.exe') 'ledger' 1800 }
if (Wanted 'prod')     { $prod     = Run-Test (Join-Path $testDir 'infernix_qwen4_exp_image_interlude_prod_real_test.exe') 'prod' 2400 }
# LAST: its final scenario leaves that Engine instance Degraded on purpose.
if (Wanted 'failures') { $failures = Run-Test (Join-Path $testDir 'infernix_qwen4_exp_image_interlude_failures_real_test.exe') 'failures' 2400 }

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT ledger=' + $ledger + ' prod=' + $prod + ' failures=' + $failures + ' healthy=' + $healthy)
exit 0
