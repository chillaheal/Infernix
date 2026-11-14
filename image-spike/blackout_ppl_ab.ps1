# FP4-Local vs Dense8 perplexity A/B, self-restoring (copy of image-spike/blackout_vision.ps1; only the
# payload differs). infernix-perplexity streams the 78 GB artifact through the SSD expert tier onto the
# same card the resident server holds at ~31 GiB, so it can only run while :8888 is down. The live
# command line is captured from the listening process before it is stopped and relaunched through
# explorer.exe at the end of every path, including the failure paths, exactly as the previous blackout
# scripts did.
#
# Two arms, same corpus/mode/context/stride so --kl-reference is legal:
#   arm 1  FP4-Local, quick profile, --save-top-tokens   (the reference run)
#   arm 2  Dense8,    quick profile, --kl-reference       (KL and top-1 agreement against arm 1)

param(
    [int]$GateMaxWaitSec = 900,      # wait for the port to be quiet instead of refusing at once
    [int]$GateExpectedMinutes = 70,  # how long this window announces that it holds the port
    [int]$GraceSec = 60,             # let an in-flight request finish before the server goes down
    [int]$TestTimeoutSec = 7200      # both arms: two 78 GB startups plus the scoring
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$ppExe      = Join-Path $repo 'build\apps\Release\infernix-perplexity.exe'
$manifest   = Join-Path $repo 'eval\corpora\perplexity-1m\manifest.json'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$modelDir   = Join-Path $desktopDir 'models'
$artFp4l    = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00001-of-00003.infernix'
$ngramFp4l  = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram'
$denseDir   = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artDense   = Join-Path $denseDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngramDense = Join-Path $denseDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$refBin     = Join-Path $repo 'image-spike\ppl_out\ref-fp4local-quick.bin'
$outRoot    = Join-Path $repo 'image-spike\ppl_out'
$out1       = Join-Path $outRoot 'fp4local_quick'
$out2       = Join-Path $outRoot 'dense8_quick'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin'
$vcvars     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$port       = 8888
$restoreBat = Join-Path $repo 'image-spike\_ppl_ab_restore.bat'
$outLog     = Join-Path $repo 'image-spike\ppl_ab_blackout.out.log'
$errLog     = Join-Path $repo 'image-spike\ppl_ab_blackout.err.log'
$runBat     = Join-Path $repo 'image-spike\_ppl_ab_run.bat'
# The evaluators refuse a non-empty --output directory, so their stdout/stderr go beside it, never in it.
$log1       = Join-Path $repo 'image-spike\ppl_arm1.out.log'
$err1       = Join-Path $repo 'image-spike\ppl_arm1.err.log'
$log2       = Join-Path $repo 'image-spike\ppl_arm2.out.log'
$err2       = Join-Path $repo 'image-spike\ppl_arm2.err.log'

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

foreach ($p in @($ppExe, $manifest, $artFp4l, $ngramFp4l, $artDense, $ngramDense, $vcpkgBin, $vcvars, $desktopDir)) {
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

# vcvars + vcpkg bin lines copied from image-spike\blackout_vision.ps1: the exe needs the CUDA runtime
# DLLs and the zlib the artifact reader links.
$common = '--corpus "' + $manifest + '" --quick --kv-dtype nvfp4 --rope-yarn-factor 1.5'
$lines = @('@echo off',
           ('cd /d "' + $repo + '"'),
           ('call "' + $vcvars + '" >nul'),
           ('set "PATH=%PATH%;' + $vcpkgBin + '"'),
           ('"' + $ppExe + '" "' + $artFp4l + '" ' + $common + ' --ngram-volume "' + $ngramFp4l + '" --output "' + $out1 + '" --save-top-tokens "' + $refBin + '" > "' + $log1 + '" 2> "' + $err1 + '"'),
           'set E1=%errorlevel%',
           ('>>"' + $outLog + '" echo ARM1_EXIT=%E1%'),
           ('"' + $ppExe + '" "' + $artDense + '" ' + $common + ' --ngram-volume "' + $ngramDense + '" --output "' + $out2 + '" --kl-reference "' + $refBin + '" > "' + $log2 + '" 2> "' + $err2 + '"'),
           'set E2=%errorlevel%',
           ('>>"' + $outLog + '" echo ARM2_EXIT=%E2%'))
[System.IO.File]::WriteAllLines($runBat, $lines, [System.Text.Encoding]::ASCII)

Log 'PHASE2 running the FP4-Local / Dense8 perplexity A/B (quick profile)'
$t0 = Get-Date
foreach ($d in @($out1, $out2)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Recurse -Force }
    New-Item -ItemType Directory -Path $d | Out-Null
}
if (Test-Path -LiteralPath $refBin) { Remove-Item -LiteralPath $refBin -Force }
if (Test-Path -LiteralPath $outLog) { Remove-Item -LiteralPath $outLog -Force }
if (Test-Path -LiteralPath $errLog) { Remove-Item -LiteralPath $errLog -Force }
$tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
$waited = 0
while (-not $tp.HasExited -and $waited -lt $TestTimeoutSec) { Start-Sleep -Seconds 15; $waited += 15 }
if (-not $tp.HasExited) { Log 'PAYLOAD_TIMEOUT_KILLING'; try { $tp.Kill() } catch { } }
Log ('payload elapsed=' + [int]((Get-Date) - $t0).TotalSeconds + 's gpu_used_mib=' + (Gpu-UsedMib))

foreach ($a in @(@('ARM1', $log1, $err1), @('ARM2', $log2, $err2))) {
    $tag = $a[0]
    $o = $a[1]
    $e = $a[2]
    if (Test-Path -LiteralPath $o) { Get-Content -LiteralPath $o | ForEach-Object { Log ($tag + ' OUT| ' + $_) } }
    if (Test-Path -LiteralPath $e) { Get-Content -LiteralPath $e -Tail 25 | ForEach-Object { Log ($tag + ' ERR| ' + $_) } }
}
$e1 = 'NOTFOUND'
$e2 = 'NOTFOUND'
if (Test-Path -LiteralPath $outLog) {
    $m1 = Select-String -LiteralPath $outLog -Pattern 'ARM1_EXIT=(\d+)' | Select-Object -Last 1
    if ($m1) { $e1 = $m1.Matches[0].Groups[1].Value }
    $m2 = Select-String -LiteralPath $outLog -Pattern 'ARM2_EXIT=(\d+)' | Select-Object -Last 1
    if ($m2) { $e2 = $m2.Matches[0].Groups[1].Value }
}
Log ('ARM1_EXIT=' + $e1 + ' ARM2_EXIT=' + $e2)

# The gate stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
Log 'PHASE3 restoring the live server'
Restore-Live
$healthy = Wait-Healthy 180
Log ('FINAL_HEALTHY=' + $healthy)
if (-not $healthy) { Log 'server not healthy after restore; the capture above holds the reason' }
& $gate -Action Release | Out-Null
Log ('RESULT ppl_ab=' + $e1 + '/' + $e2 + ' healthy=' + $healthy)
exit 0
