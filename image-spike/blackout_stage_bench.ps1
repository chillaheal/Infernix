# Bundle B fas 0 in a short blackout: stop the live :8888 server, run the image stage bench on a GPU
# with nothing else on it (its pause number is meaningless beside a live LLM), then relaunch the server
# with the exact command line it was running before and wait for /health.
#
# The bench writes its pictures to <cwd>/images, so it runs with the Desktop folder as its working
# directory, the same place the live server writes them.

param([string]$Rows = '')  # e.g. 'cfg' to re-measure one row (see INFERNIX_BENCH_ROWS in the bench)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$bench      = Join-Path $repo 'build\bench\Release\infernix_image_stage_bench.exe'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$modelDir   = Join-Path $desktopDir 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact   = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram      = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$imageArtifact  = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$processorNote = 'the tower prompt processor directory the live server names with --image-text-processor'
$towerDir   = Join-Path $desktopDir 'models\image-processor'
$restoreBat = Join-Path $repo 'image-spike\_stageBench_restore.bat'
$runBat     = Join-Path $repo 'image-spike\_stageBench_run.bat'
$outLog     = Join-Path $repo 'image-spike\stage_bench.out.log'
$errLog     = Join-Path $repo 'image-spike\stage_bench.err.log'
$port       = 8888
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
            if ([int]$h.StatusCode -eq 200) { return $true }
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

foreach ($p in @($bench, $artifact, $imageArtifact, $towerDir, $desktopDir)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}

$env:Path = $vcpkgBin + ';' + $env:Path
$env:INFERNIX_QWEN4_ARTIFACT        = $artifact
if (Test-Path -LiteralPath $ngram) { $env:INFERNIX_QWEN4_NGRAM = $ngram } else { Log ('NGRAM_MISSING ' + $ngram) }
$env:INFERNIX_TEST_ARTIFACT         = $imageArtifact
$env:INFERNIX_IMAGE_TEXT_PROCESSOR  = $towerDir
if ($Rows -ne '') { $env:INFERNIX_BENCH_ROWS = $Rows }
Log ('tower processor (purpose: ' + $processorNote + '): ' + $towerDir)

Log 'PHASE0 grace 60s (let any in-flight request finish before the server goes down)'
Start-Sleep -Seconds 60

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

# The bench runs from a batch file with the Desktop folder as cwd, so its images/ land where the live
# server's do. The exit code is written into the log the same way the cert batch does it.
$lines = @('@echo off',
           ('cd /d "' + $desktopDir + '"'),
           ('"' + $bench + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
           'set TE=0',
           'if errorlevel 1 set TE=1',
           ('>>"' + $outLog + '" echo BENCHEXIT=%TE%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the stage bench (engine start plus eight rows)'
$t0 = Get-Date
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt 3000) { Start-Sleep -Seconds 10; $waited += 10 }
if (-not $tp.HasExited) { Log 'BENCH_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('bench elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's')

$exit = 'NOTFOUND'
if (Test-Path -LiteralPath $outLog) {
    $m = Select-String -LiteralPath $outLog -Pattern 'BENCHEXIT=(\d+)' | Select-Object -Last 1
    if ($m) { $exit = $m.Matches[0].Groups[1].Value }
    Get-Content -LiteralPath $outLog | ForEach-Object { Log ('OUT| ' + $_) }
}
if (Test-Path -LiteralPath $errLog) {
    Get-Content -LiteralPath $errLog -Tail 60 | ForEach-Object { Log ('ERR| ' + $_) }
}
Log ('BENCHEXIT=' + $exit)

Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
Log ('RESULT bench=' + $exit + ' healthy=' + $healthy)
exit 0
