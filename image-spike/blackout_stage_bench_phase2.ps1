# Bundle B fas 2 measurement in ONE blackout: stop the live :8888 server, run the image stage bench
# three times on a GPU with nothing else on it, then relaunch the server with the exact command line
# it was running before and wait for /health.
#
#   run A  full matrix, default route (overlap prefetch, cached pinned host)   -> the "after" headline
#   run B  warm,cfg rows, INFERNIX_IMAGE_OVERLAP=0 (serialized, no lookahead)  -> isolates the overlap
#   run C  warm,cfg rows, INFERNIX_IMAGE_HOST_CACHE_WC=1 (write-combined)      -> decides WC vs cached
#
# The pipeline's per-job "image job: timing" line (loop wall, copy est, mode, asyncEngineCount) goes to
# the per-run err/out logs, which is what attributes the numbers. Comparing A vs B on the same warm/cfg
# rows gives the overlap effect; A vs C gives the write-combined effect.

param(
    [string]$Runs = 'A,B,C',         # subset of the three measurements to run
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 60   # the three bench runs plus the restore, announced
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$bench      = Join-Path $repo 'build\bench\Release\infernix_image_stage_bench.exe'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$modelDir   = Join-Path $desktopDir 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact   = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram      = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$imageArtifact  = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$towerDir   = Join-Path $desktopDir 'models\image-processor'
$restoreBat = Join-Path $repo 'image-spike\_stageBenchP2_restore.bat'
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
function Set-Env($name, $value) {
    if ($value -eq '') { Remove-Item -LiteralPath ('Env:\' + $name) -ErrorAction SilentlyContinue }
    else { Set-Item -LiteralPath ('Env:\' + $name) -Value $value }
}
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

# The window gate. :8888 carries every other Claude Code session's turns, so the port is taken only
# when it is quiet; the gate refuses (exit 10) unless it is, or unless -GateForce says otherwise.
$gate = Join-Path $repo 'image-spike\window_gate.ps1'
if (-not (Test-Path -LiteralPath $gate)) { Log ('FATAL missing gate: ' + $gate); exit 2 }
& $gate -Action StartWindow -Owner (Split-Path -Leaf $MyInvocation.MyCommand.Path) -Reason (Split-Path -Leaf $MyInvocation.MyCommand.Path) -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }

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

# One bench run with the given rows and pipeline env, into <tag>.out.log / <tag>.err.log.
function Run-Bench($tag, $rows, $overlap, $wc) {
    Set-Env 'INFERNIX_BENCH_ROWS' $rows
    Set-Env 'INFERNIX_IMAGE_OVERLAP' $overlap
    Set-Env 'INFERNIX_IMAGE_HOST_CACHE_WC' $wc
    $outLog = Join-Path $repo ('image-spike\stage_bench_' + $tag + '.out.log')
    $errLog = Join-Path $repo ('image-spike\stage_bench_' + $tag + '.err.log')
    $runBat = Join-Path $repo ('image-spike\_stageBenchP2_run_' + $tag + '.bat')
    $lines = @('@echo off',
               ('cd /d "' + $desktopDir + '"'),
               ('"' + $bench + '" > "' + $outLog + '" 2> "' + $errLog + '"'),
               'set TE=0',
               'if errorlevel 1 set TE=1',
               ('>>"' + $outLog + '" echo BENCHEXIT=%TE%'))
    [System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)
    Log ('RUN ' + $tag + ' start rows=[' + $rows + '] overlap=[' + $overlap + '] wc=[' + $wc + ']')
    $t0 = Get-Date
    $tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
    $waited = 0
    while (-not $tp.HasExited -and $waited -lt 2400) { Start-Sleep -Seconds 10; $waited += 10 }
    if (-not $tp.HasExited) { Log ('BENCH_TIMEOUT_KILLING ' + $tag); try { $tp.Kill() } catch { } }
    Log ('RUN ' + $tag + ' elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's')
    $exit = 'NOTFOUND'
    if (Test-Path -LiteralPath $outLog) {
        $m = Select-String -LiteralPath $outLog -Pattern 'BENCHEXIT=(\d+)' | Select-Object -Last 1
        if ($m) { $exit = $m.Matches[0].Groups[1].Value }
        Get-Content -LiteralPath $outLog | ForEach-Object { Log ($tag + '| ' + $_) }
    }
    if (Test-Path -LiteralPath $errLog) {
        Get-Content -LiteralPath $errLog | Where-Object { $_ -match 'timing|RESULT|error|Exception' } |
            ForEach-Object { Log ($tag + 'E| ' + $_) }
    }
    Log ('RUN ' + $tag + ' BENCHEXIT=' + $exit)
}

try {
    $want = @($Runs.Split(',') | ForEach-Object { $_.Trim().ToUpper() })
    if ($want -contains 'A') { Run-Bench 'A_full_overlap'      'cold,warm,ref,cfg' '' '' }
    if ($want -contains 'B') { Run-Bench 'B_serialized_overlap0' 'warm,cfg' '0' '' }
    if ($want -contains 'C') { Run-Bench 'C_writecombined'     'warm,cfg' '' '1' }
} catch {
    Log ('RUN PHASE THREW: ' + $_.Exception.Message)
}

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
& $gate -Action Release | Out-Null
Log ('FINAL_HEALTHY=' + $healthy)
exit 0
