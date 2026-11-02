$ErrorActionPreference = 'Continue'
# STEP-8D ROUND 2. Round 1 (blackout_step8d_test.ps1) certified s8dprod + s8dfail GREEN
# (evidence reused, not re-run here) and exposed two reds, both fixed:
#   RED1 s8dtower: fp8 linear: unsupported shape (tower k/v 1024x4096 not in kShapes)
#        -> FIXED: src/ops/linear/fp8/shapes/n1024_k4096.cu + registry; test target rebuilt EXIT=0.
#   RED2 live phase: the launch used the old start-bat $argLine whose --ngram-volume path
#        does not exist (bogus Dense8 name) -> startup FATAL. FIXED: launch from $liveCmd,
#        the exact command line the healthy server was running (same source PHASE 5 restore uses).
# Also fixes round 1's empty TEST_EXIT values: exit code is now captured via TESTEXIT= in the
# test's own stdout log (cmd wrapper), not Start-Process ExitCode.
# Window: grace 90 s -> stop+drain -> relink infernix-serve (new FP8 shape) -> s8dtower cert ->
# LIVE (image POST, chat POST, second image POST) -> drain -> restore $liveCmd detached via explorer.
$root     = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$exe      = Join-Path $root 'infernix-serve.exe'
$port     = 8888
$freeMin  = 20000
$repo     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$repoExe  = Join-Path $repo 'build\tests\Release'
$buildSrv = Join-Path $repo 'build\apps\Release\infernix-serve.exe'
$testTow  = Join-Path $repoExe 'infernix_image_tower_provider_real_test.exe'
$modelDir = Join-Path $root 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram    = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'  # exact name from Glob
$imageOracle     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle'
$imageOraclePipe = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle_512_mainecoon\tensors_pipeline'
$imageArtifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$processorDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
# Fallback args only (PHASE 1 normally captures the real command line). ngram token corrected to
# the Glob-observed file name; the historical Dense8-named ngram path in the start bat never existed.
$argLine  = ' "' + $artifact + '" --ngram-volume "' + $ngram + '" --host 127.0.0.1 --port 8888 --max-context 200000 --kv-capacity auto --kv-dtype fp8 --default-max-tokens 16384 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --default-thinking-budget 8192 --thinking-budget-policy clamp --vision --vision-offload on --spec mtp --draft-tokens 5 --lm-head-draft --preserve-thinking --request-log-jsonl requests.jsonl --max-concurrency 1 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --prefill-chunk 2048 --log-level info --prefix-cache-file "C:\Users\Micke\Desktop\Infernix FlashNext\prefix-cache-flash-next.bin" --expert-ram-mib 41500  --ngram-draft-tokens 15 --ngram-min-match 12  --log-stats-panel on --host-context-mib 4096'

function FreeMib {
  $v = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ($v) { return [int]$v } else { return -1 }
}

function ListenerPid {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.OwningProcess } else { return $null }
}

function RunTest($testExe, $tag, $ppm) {
  if (-not (Test-Path $testExe)) { Write-Host ('TEST_EXE_MISSING ' + $testExe); return }
  $outLog = Join-Path $root ('interlude_' + $tag + '_test.out.log')
  $errLog = Join-Path $root ('interlude_' + $tag + '_test.err.log')
  $runBat = Join-Path $root ('interlude_' + $tag + '_run.bat')
  if (Test-Path $outLog) { Remove-Item $outLog -Force -ErrorAction SilentlyContinue }
  if (Test-Path $errLog) { Remove-Item $errLog -Force -ErrorAction SilentlyContinue }
  $env:INFERNIX_QWEN4_ARTIFACT    = $artifact
  $env:INFERNIX_QWEN4_NGRAM       = $ngram
  $env:INFERNIX_IMAGE_ORACLE      = $imageOracle
  $env:INFERNIX_IMAGE_ORACLE_PIPELINE = $imageOraclePipe
  $env:INFERNIX_QWENVL_PROCESSOR  = $processorDir
  $env:INFERNIX_TEST_ARTIFACT     = $imageArtifact
  $env:INFERNIX_IMAGE_OUT_PPM     = $ppm
  ('@echo off' + [Environment]::NewLine +
   'cd /d "' + $root + '"' + [Environment]::NewLine +
   '"' + $testExe + '" > "' + $outLog + '" 2> "' + $errLog + '"' + [Environment]::NewLine +
   'echo TESTEXIT=%ERRORLEVEL%>> "' + $outLog + '"') | Set-Content -Path $runBat -Encoding ASCII -Force
  $t0 = Get-Date
  $tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
  $waited = 0
  while (-not $tp.HasExited -and $waited -lt 1200) { Start-Sleep -Seconds 3; $waited += 3 }
  $elapsed = [int]((Get-Date) - $t0).TotalSeconds
  if (-not $tp.HasExited) { Write-Host 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch {} }
  $exit = 'TIMEOUT'
  if (Test-Path $outLog) {
    $m = Select-String -Path $outLog -Pattern 'TESTEXIT=(-?\d+)' | Select-Object -Last 1
    if ($m) { $exit = $m.Matches[0].Groups[1].Value }
  }
  Write-Host ('=== TEST_EXIT[' + $tag + ']=' + $exit + ' elapsed=' + $elapsed + 's ===')
  Write-Host ('--- ' + $tag + ' STDOUT (full) ---')
  if (Test-Path $outLog) { Get-Content $outLog | ForEach-Object { Write-Host $_ } }
  Write-Host ('--- ' + $tag + ' STDERR (tail 100) ---')
  if (Test-Path $errLog) { Get-Content $errLog -Tail 100 | ForEach-Object { Write-Host $_ } }
}

Write-Host '===== PHASE 0: GRACE 90s ====='
Start-Sleep -Seconds 90

Write-Host '===== PHASE 1: STOP 8888 + DRAIN ====='
$liveModelId = ''
$liveCmd     = ''
$c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($c) {
  $p = $c.OwningProcess
  try {
    $m = (Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/models') -UseBasicParsing -TimeoutSec 4).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { $liveModelId = $Matches[1] }
  } catch { $liveModelId = '' }
  $pp = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $p) -ErrorAction SilentlyContinue
  if ($pp) { $liveCmd = $pp.CommandLine }
  Write-Host ('stopping pid=' + $p + ' model=' + $liveModelId)
  taskkill /F /PID $p 2>&1 | Out-Null
} else {
  Write-Host 'no listener; nothing to stop'
}
Write-Host ('captured live command line: ' + $liveCmd)
$i = 0
while ($i -lt 30) {
  Start-Sleep -Seconds 1
  if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { break }
  $i++
}
Write-Host 'port free'
$f = -1; $j = 0
while ($j -lt 60) {
  $f = FreeMib
  if ($f -ge $freeMin) { break }
  Start-Sleep -Seconds 2
  $j++
}
Write-Host ('free_mib_before_job=' + $f)
Start-Sleep -Seconds 3

Write-Host '===== PHASE 2: RELINK THE SERVER EXE (with the N1024K4096 FP8 shape) ====='
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$srvLog = Join-Path $repo 'out\step8d2_serve_build.log'
& cmake --build (Join-Path $repo 'build') --config Release --target infernix-serve -j *>> $srvLog 2>&1
Write-Host ('STEP8D2_BUILD_EXIT=' + $LASTEXITCODE)
$srvOk = ($LASTEXITCODE -eq 0) -and (Test-Path $buildSrv)
if (-not $srvOk) { Write-Host 'SERVE_BUILD_FAILED - live HTTP phase will be SKIPPED; tower cert still runs' }

Write-Host '===== PHASE 3: s8dtower CERT (prod+fail certs certified in round 1, evidence reused) ====='
foreach ($p in @($artifact, $ngram, $imageOracle, $imageOraclePipe, $imageArtifact, $processorDir, $testTow)) {
  if (-not (Test-Path $p)) { Write-Host ('PREREQ_MISSING ' + $p) }
}
RunTest $testTow 's8d2tower' 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\interlude_s8d2tower_mine.ppm'

Write-Host '===== PHASE 4: LIVE HTTP ON THE REBUILT SERVER (launched from the captured live command line) ====='
if (-not $srvOk) {
  Write-Host 'LIVE_PHASE_SKIPPED'
} elseif ($liveCmd -eq '') {
  Write-Host 'LIVE_PHASE_SKIPPED_NO_CAPTURED_COMMAND'
} else {
  $outLog = Join-Path $root 'serve_s8d2.out.log'
  $errLog = Join-Path $root 'serve_s8d2.err.log'
  $srvBat = Join-Path $root 'serve_s8d2.bat'
  ('@echo off' + [Environment]::NewLine +
   'cd /d "' + $root + '"' + [Environment]::NewLine +
   $liveCmd + ' --image-artifact "' + $imageArtifact + '" --image-text-processor "' + $processorDir + '" > "' + $outLog + '" 2> "' + $errLog + '"') | Set-Content -Path $srvBat -Encoding ASCII -Force
  Start-Process -FilePath explorer.exe -ArgumentList ('"' + $srvBat + '"') | Out-Null
  $liveId = ''
  $t = 0
  while ($t -lt 160) {
    Start-Sleep -Seconds 4
    try {
      $m = (Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/models') -UseBasicParsing -TimeoutSec 4).Content
      if ($m -match '"id"\s*:\s*"([^"]+)"') { $liveId = $Matches[1]; break }
    } catch {}
    $t++
  }
  if ($liveId -eq '') {
    Write-Host 'LIVE_SERVE_START_FAILED'
    if (Test-Path $errLog) { Get-Content $errLog -Tail 40 | ForEach-Object { Write-Host $_ } }
    $held = ListenerPid
    if ($held) { taskkill /F /PID $held 2>&1 | Out-Null; Start-Sleep -Seconds 3 }
  } else {
    Write-Host ('LIVE_SERVE_UP model=' + $liveId)
    $body1 = @{ model = $liveId; prompt = 'A white smoke Maine Coon cat'; size = '512x512'; seed = 42 } | ConvertTo-Json -Compress
    try {
      $r1 = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/images/generations') -Method Post `
              -ContentType 'application/json' -Body $body1 -UseBasicParsing -TimeoutSec 900
      $j1 = $r1.Content | ConvertFrom-Json
      $b64 = $j1.data[0].b64_json
      $png = [Convert]::FromBase64String($b64)
      $magic = ($png[0] -eq 137 -and $png[1] -eq 80 -and $png[2] -eq 78 -and $png[3] -eq 71)
      Write-Host ('IMAGE_POST1 status=' + $r1.StatusCode + ' bytes=' + $png.Length + ' png_magic=' + $magic)
      [IO.File]::WriteAllBytes((Join-Path $repo 'out\step8d2_post1.png'), $png)
    } catch { Write-Host ('IMAGE_POST1_FAILED ' + $_.Exception.Message) }
    try {
      $body2 = @{ model = $liveId; messages = @(@{ role = 'user'; content = 'Reply with exactly: OK' }); max_tokens = 16 } | ConvertTo-Json -Compress -Depth 5
      $r2 = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/chat/completions') -Method Post `
              -ContentType 'application/json' -Body $body2 -UseBasicParsing -TimeoutSec 600
      Write-Host ('LLM_POST status=' + $r2.StatusCode)
      Write-Host ('LLM_BODY ' + $r2.Content.Substring(0, [Math]::Min(400, $r2.Content.Length)))
    } catch { Write-Host ('LLM_POST_FAILED ' + $_.Exception.Message) }
    try {
      $body3 = @{ model = $liveId; prompt = 'A red fox reading a book'; size = '512x512'; seed = 7 } | ConvertTo-Json -Compress
      $r3 = Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/images/generations') -Method Post `
              -ContentType 'application/json' -Body $body3 -UseBasicParsing -TimeoutSec 900
      $j3 = $r3.Content | ConvertFrom-Json
      $png3 = [Convert]::FromBase64String($j3.data[0].b64_json)
      Write-Host ('IMAGE_POST2 status=' + $r3.StatusCode + ' bytes=' + $png3.Length)
      [IO.File]::WriteAllBytes((Join-Path $repo 'out\step8d2_post2.png'), $png3)
    } catch { Write-Host ('IMAGE_POST2_FAILED ' + $_.Exception.Message) }
    Write-Host 'stopping the test server (restore must relaunch the PRODUCTION command)'
    $held = ListenerPid
    if ($held) { taskkill /F /PID $held 2>&1 | Out-Null; Start-Sleep -Seconds 3 }
    $i = 0
    while ($i -lt 30) {
      Start-Sleep -Seconds 1
      if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { break }
      $i++
    }
    Write-Host 'test server stopped, port free'
    $f = -1; $j = 0
    while ($j -lt 60) { $f = FreeMib; if ($f -ge $freeMin) { break }; Start-Sleep -Seconds 2; $j++ }
    Write-Host ('free_mib_after_live=' + $f)
  }
}

Write-Host '===== PHASE 5: ENSURE A HEALTHY SERVER ON :8888 ====='
if (-not (Test-Path $exe)) { Write-Host ('EXE_MISSING ' + $exe); exit 1 }

function CurrentModelId {
  try {
    $m = (Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/models') -UseBasicParsing -TimeoutSec 4).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { return $Matches[1] }
  } catch {}
  return ''
}
function WaitHealthy {
  for ($t = 0; $t -lt 100; $t++) {
    $id = CurrentModelId
    if ($id -ne '' -and ($liveModelId -eq '' -or $id -eq $liveModelId)) { return $true }
    Start-Sleep -Seconds 3
  }
  return $false
}

Start-Sleep -Seconds 10
$spawned = $null
$held = ListenerPid
if ($held) {
  $heldId = CurrentModelId
  if ($liveModelId -ne '' -and $heldId -ne '' -and $heldId -ne $liveModelId) {
    Write-Host ('WARNING: listener pid=' + $held + ' is model ' + $heldId + ' but the pre-window model was ' + $liveModelId + ' - foreign server, NOT touching it')
  } else {
    Write-Host ('deferring to existing listener pid=' + $held + ' model=' + $heldId)
  }
} else {
  Write-Host 'no listener after the defer; restarting detached via explorer (survives this terminal)'
  $bat = Join-Path $root 'restore_serve.bat'
  $cmdToRun = '"' + $exe + '"' + $argLine
  if ($liveCmd -ne '') { $cmdToRun = $liveCmd }
  ('cd /d "' + $root + '"' + [Environment]::NewLine + $cmdToRun) | Set-Content -Path $bat -Encoding ASCII -Force
  Start-Process -FilePath explorer.exe -ArgumentList ('"' + $bat + '"') | Out-Null
  $spawned = 'detached'
}
$ok = WaitHealthy
$finalId = CurrentModelId
Write-Host ('spawned=' + $spawned + ' now_listening=' + (ListenerPid) + ' model=' + $finalId + ' server_up=' + $ok)
if ($ok) { Write-Host ('SERVER_RESTORED_OK (model ' + $finalId + ' matches pre-window ' + $liveModelId + ')') }
else      { Write-Host ('SERVER_RESTORED_FAILED (expected ' + $liveModelId + ', got ' + $finalId + ')') }
Write-Host ('RUN END ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss,ff'))
