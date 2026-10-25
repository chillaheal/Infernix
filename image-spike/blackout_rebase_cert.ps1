param(
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 75,  # announced to whoever is waiting on the port
    [int]$GraceSec = 30,             # the cert exes are already built, so the grace can be short
    [string]$ExactOldExe = '',       # a pre-rebase route-trace exe: gives the literal bit-identical pair
    [switch]$ForceGate,              # the user's own call: take the port despite BUSY, the sessions on it lose their turns
    [switch]$SkipDeploy              # run the certs without rebuilding/replacing the serve exe
)

$ErrorActionPreference = 'Continue'

# The window that finishes the origin/main rebase: the gpu;real certifications of the coexistence work,
# the Release relink and deploy of the serve that carries the merge, and the exact route's ids.
#
#   CERT     ctest -C Debug over the image coexistence set (yield, interlude op/failures/prod/dit/full,
#            queue-deadline, expert-fault). Every one of them has its own oracle: the DiT/prod steps gate
#            on FINAL_LATENT < 0.5 and FINAL_IMAGE < 0.3 against the official pipeline tensors, the
#            failure cert on the state machine, the queue cert on its own hold measurement.
#   EXACT    the route trace through the public Engine on the real artifact. By default the prompts run
#            twice (trace off, then on) and the ids must be equal. With -ExactOldExe pointing at a binary
#            built BEFORE the merge, that binary writes --ids-out and the rebuilt one must match it with
#            --ids-expect: the literal bit-identical check across the rebase.
#   DEPLOY   cmake --build build --config Release --target infernix-serve, copied over the Desktop exe
#            (backed up first). This has to happen INSIDE the window: the running server holds the file.
#
# Self-restoring like the other blackout scripts: the gate takes the port only when it is quiet (exit 10
# otherwise, with nothing stopped), the shield is down and the port free before serve is relaunched, and
# the server comes back on the exact command line it was captured with.

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$buildDir   = Join-Path $repo 'build'
$testDir    = Join-Path $buildDir 'tests\Debug'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = Join-Path $desktopDir 'infernix-serve.exe'
$backupExe  = Join-Path $desktopDir 'infernix-serve.exe.pre-rebase-cert.bak'
$startBat   = Join-Path $desktopDir 'start flash next.bat'
$buildExe   = Join-Path $buildDir 'apps\Release\infernix-serve.exe'
$cmake      = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'
$gate       = Join-Path $repo 'image-spike\window_gate.ps1'
$restoreBat = Join-Path $repo 'image-spike\_rebaseCert_restore.bat'
$log        = Join-Path $repo 'image-spike\blackout_rebase_cert.log'
$vcpkgBin   = 'C:\vcpkg\installed\x64-windows\bin;C:\vcpkg\installed\x64-windows\debug\bin'

$modelDir   = Join-Path $desktopDir 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact   = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram      = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$imageArtifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$imageOracle     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle'
$imageOraclePipe = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle_512_mainecoon\tensors_pipeline'

$prompts    = Join-Path $repo 'image-spike\exact_prompts.json'
$traceBin   = Join-Path $desktopDir 'rebaseCert_trace.bin'
$idsBefore  = Join-Path $desktopDir 'rebaseCert_ids_before.txt'
$exactExeNew = Join-Path $testDir 'infernix_qwen4_exp_route_trace_real_test.exe'
$port       = 8888

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
function Serve-Pids {
    # The shield ALSO listens on :8888 while the window is open, so the listener is never the test for
    # "the server is down" - the process is. Killing by listener would take the shield down instead.
    return @(Get-Process -Name infernix-serve -ErrorAction SilentlyContinue | ForEach-Object { [int]$_.Id })
}
function Kill-Serve {
    $ids = Serve-Pids
    if ($ids.Count -gt 0) {
        taskkill /F /T /PID ($ids -join ' /PID ') 2>&1 | Out-Null
        Log ('killed serve pid=' + ($ids -join ','))
    }
    for ($i = 0; $i -lt 40; $i++) {
        if ((Serve-Pids).Count -eq 0) { break }
        Start-Sleep -Seconds 2
    }
}
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

$liveCmd    = ''
$deployed   = $false
$windowOpen = $false
function Bring-Back-Old {
    if ($deployed -and (Test-Path -LiteralPath $backupExe)) {
        Copy-Item -LiteralPath $backupExe -Destination $desktopExe -Force
        Log ('RESTORE: the backed up exe is back, size=' + (Get-Item -LiteralPath $backupExe).Length)
    }
    if ($windowOpen) { & $gate -Action EndWindow | Out-Null; $script:windowOpen = $false; Log 'ENDWINDOW (restore path)' }
    if (-not $liveCmd) { Log 'RESTORE: no live command was captured, nothing to relaunch'; return }
    [System.IO.File]::WriteAllLines($restoreBat, (@('@echo off', ('cd /d "' + $desktopDir + '"'), $liveCmd)), [System.Text.Encoding]::ASCII)
    Launch-Bat $restoreBat
    Log 'RESTORE: the captured live command was relaunched (detached, visible window)'
}

# ---- PHASE 0: every path and every environment bit checked before the port is touched ----
foreach ($p in @($artifact, $imageArtifact, $imageOracle, $imageOraclePipe, $testDir, $buildDir, $cmake, $gate, $startBat, $desktopExe)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}
if ($ExactOldExe -and -not (Test-Path -LiteralPath $ExactOldExe)) { Log ('FATAL -ExactOldExe missing: ' + $ExactOldExe); exit 2 }
if (-not (Test-Path -LiteralPath $prompts)) {
    # The route-trace test's own prompt-list format (see tests/models/qwen4_exp/test_route_trace_real.cpp).
    $p = @(
        [pscustomobject]@{ name = 'rebase-a'; max_tokens = 96;  thinking = $false; messages = @([pscustomobject]@{ role = 'user'; content = 'Describe in one short paragraph what a CUDA graph captures.' }) },
        [pscustomobject]@{ name = 'rebase-b'; max_tokens = 96;  thinking = $false; messages = @([pscustomobject]@{ role = 'user'; content = 'List three reasons an expert cache misses on a routed MoE.' }) }
    )
    ($p | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $prompts -Encoding ASCII
    Log ('WROTE prompts ' + $prompts)
}
$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break }
}
if (-not $serveLine) { Log 'FATAL no serve line in the start bat'; exit 2 }
Log ('PHASE0 ok serve_line_len=' + $serveLine.Length + ' gpu_used_mib=' + (Gpu-UsedMib))

$env:Path = $vcpkgBin + ';' + $env:Path
$env:INFERNIX_QWEN4_ARTIFACT        = $artifact
if (Test-Path -LiteralPath $ngram) { $env:INFERNIX_QWEN4_NGRAM = $ngram } else { Log ('NGRAM_MISSING ' + $ngram) }
$env:INFERNIX_TEST_ARTIFACT         = $imageArtifact
$env:INFERNIX_IMAGE_ORACLE          = $imageOracle
$env:INFERNIX_IMAGE_ORACLE_PIPELINE = $imageOraclePipe

# ---- PHASE 1: take the port through the gate, or stop here with nothing changed ----
# The gate's -Action is Position=0, so an ARRAY splat binds the literal '-Action' as its VALUE and the
# call dies on the ValidateSet: splat a hashtable (names only) and preset the exit code to 1 so a call
# that never binds counts as a refusal instead of falling through with the port untouched.
$LASTEXITCODE = 1
$gateParams = @{
    Action          = 'StartWindow'
    Owner           = (Split-Path -Leaf $MyInvocation.MyCommand.Path)
    Reason          = 'blackout_rebase_cert.ps1 (gpu cert + exact route + deploy)'
    MaxWaitSec      = $GateMaxWaitSec
    ExpectedMinutes = $GateExpectedMinutes
}
if ($ForceGate) { $gateParams.Force = $true }
& $gate @gateParams
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }
$windowOpen = $true
Log ('PHASE1 grace ' + $GraceSec + 's, gpu_used_mib=' + (Gpu-UsedMib))
Start-Sleep -Seconds $GraceSec

$servePids = Serve-Pids
if ($servePids.Count -eq 0) {
    Log 'FATAL no infernix-serve process is running; refusing to take the window'
    & $gate -Action EndWindow | Out-Null; $windowOpen = $false; exit 2
}
try { $liveCmd = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $servePids[0])).CommandLine } catch { $liveCmd = '' }
if (-not $liveCmd) { Log 'FATAL could not capture the live command line; refusing to stop the server'; & $gate -Action EndWindow | Out-Null; $windowOpen = $false; exit 2 }
Kill-Serve
if ((Serve-Pids).Count -ne 0) { Log 'FATAL infernix-serve is still running after the kill'; Bring-Back-Old; exit 2 }
for ($i = 0; $i -lt 20; $i++) {
    # The certs need the GPU, not only a dead process: the expert arena and the KV mapping leave with it.
    if ((Gpu-UsedMib) -lt 4000) { break }
    Start-Sleep -Seconds 3
}
Log ('PHASE1 serve stopped, listener=' + (Listener-Pid) + ' (the shield may hold it), gpu_used_mib=' + (Gpu-UsedMib))

$okCert = 'FAIL'; $okExact = 'FAIL'; $okDeploy = 'NA'; $healthy = $false
try {
    # ---- PHASE 2: the literal before/after of the exact route, from the binary that predates the merge ----
    if ($ExactOldExe) {
        Log 'PHASE2 exact route: OLD build writes ids'
        & $ExactOldExe $prompts $traceBin --ids-out $idsBefore *>> $log
        $oldExit = $LASTEXITCODE
        Log ('EXACT_OLD exit=' + $oldExit + ' ids_before=' + ((Get-Content -LiteralPath $idsBefore -ErrorAction SilentlyContinue | Measure-Object).Count) + ' lines')
    }

    # ---- PHASE 3: the gpu;real certifications of the merged tree ----
    Log 'PHASE3 ctest gpu;real coexistence set (this is the long phase)'
    $pattern = 'infernix_qwen4_exp_(image_yield_real_test|image_interlude_op_real_test|image_interlude_failures_real_test|image_interlude_prod_real_test|image_interlude_dit_real_test|image_interlude_full_real_test|image_queue_deadline_real_test|expert_fault_real_test)'
    $ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
    & $ctest --test-dir $buildDir -C Debug --output-on-failure -R $pattern *>> $log
    $certExit = $LASTEXITCODE
    # A gpu;real test that skips (exit 77, artifact missing) still leaves a clean ctest run, so count verdicts.
    $passed  = (Select-String -LiteralPath $log -Pattern '\.\.\.\s+Passed' | Measure-Object).Count
    $skipped = (Select-String -LiteralPath $log -Pattern '\.\.\.\s+Skipped' | Measure-Object).Count
    Log ('PHASE3 ctest exit=' + $certExit + ' passed=' + $passed + ' skipped=' + $skipped)
    if ($certExit -eq 0 -and $passed -ge 8) { $okCert = 'PASS' }

    # ---- PHASE 4: the exact route on the merged tree ----
    Log 'PHASE4 exact route on the merged tree'
    if ($ExactOldExe) {
        & $exactExeNew $prompts $traceBin --ids-expect $idsBefore *>> $log
    } else {
        & $exactExeNew $prompts $traceBin *>> $log
    }
    $exactExit = $LASTEXITCODE
    Log ('PHASE4 exact exit=' + $exactExit + ' mode=' + $(if ($ExactOldExe) { 'ids-expect vs pre-rebase' } else { 'trace off/on self-equal' }))
    if ($exactExit -eq 0) { $okExact = 'PASS' }

    # ---- PHASE 5: the Release relink and the deploy of the merged serve ----
    if (-not $SkipDeploy) {
        Log 'PHASE5 Release relink of infernix-serve (the server is down, so the file is free)'
        & $cmake --build $buildDir --config Release -j 8 --target infernix-serve *>> $log
        $buildExit = $LASTEXITCODE
        if ($buildExit -ne 0) { Log ('DEPLOY_FAILED build exit=' + $buildExit) }
        else {
            Copy-Item -LiteralPath $desktopExe -Destination $backupExe -Force
            Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force
            $deployed = $true
            $okDeploy = 'PASS'
            Log ('DEPLOY ok new exe size=' + (Get-Item -LiteralPath $desktopExe).Length + ' backup=' + (Get-Item -LiteralPath $backupExe).Length)
        }
    }
} catch {
    Log ('EXCEPTION ' + $_.Exception.Message)
}

# ---- PHASE 6: the shield down, the port free, then serve back on its own captured command line ----
& $gate -Action EndWindow | Out-Null
$windowOpen = $false
Log 'ENDWINDOW'
for ($i = 0; $i -lt 20; $i++) {
    if ((Listener-Pid) -eq 0 -and (Gpu-UsedMib) -lt 8000) { break }
    Start-Sleep -Seconds 3
}
Log ('port free, gpu_used_mib=' + (Gpu-UsedMib))
Launch-Bat ([System.IO.Path]::GetFullPath($startBat))
$healthy = Wait-Healthy 130
Log ('PHASE6 healthy=' + $healthy + ' gpu_used_mib=' + (Gpu-UsedMib))
if (-not $healthy -and $deployed) {
    Log 'the NEW build did not come up: the backed up exe goes back and the start is tried once more'
    Copy-Item -LiteralPath $backupExe -Destination $desktopExe -Force
    Launch-Bat ([System.IO.Path]::GetFullPath($startBat))
    $healthy = Wait-Healthy 130
    Log ('retry with the pre-window exe healthy=' + $healthy)
}
if (-not $healthy) {
    Log 'SERVE DID NOT COME BACK: the machine is left with the port free, restore by hand if needed'
}
$note = if ($ExactOldExe) { 'ids-expect-vs-pre-rebase' } else { 'trace-off-on' }
Log ('RESULT rebaseCert cert=' + $okCert + ' exact=' + $okExact + ' exact_mode=' + $note + ' deploy=' + $okDeploy + ' healthy=' + $healthy + ' gpu_mib=' + (Gpu-UsedMib))
Log ('log: ' + $log)
if ($okCert -eq 'PASS' -and $okExact -eq 'PASS' -and $healthy) { exit 0 } else { exit 1 }
