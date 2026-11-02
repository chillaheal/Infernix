$ErrorActionPreference = 'Continue'
# STEP-8D ROUND 3. Evidence carried from rounds 1-2 (disclosed reuse, NOT re-run here):
#   - s8dprod + s8dfail certs: round 1 GREEN (blackout_step8d.outer.log).
#   - LIVE phase (raw-prompt POST -> PNG inside the live server, chat after image job, second
#     POST): round 2 GREEN (blackout_step8d2.outer.log); the serve exe has not changed since its
#     round-2 relink, so no relink and no live phase here.
# Round 2 REDs resolved here:
#   - tower cert rewritten: FP8 route now gated vs a naive FP64 tower decoded from the SAME v2
#     artifact's FP8 RowScale planes (tol 1e-1, same decoded weights); pre_norm kept as METRICS,
#     preserving the round-2 device-vs-pre_norm 1.394e-01 disclosure.
#   - FP8 op test now covers the four image shapes {4096,4096} {12288,4096} {4096,12288}
#     {1024,4096} against the FP64 oracle at the full A16 ladder (the shapes were never in it).
#   - exit capture fixed: the wrapper writes TESTEXIT via "if errorlevel" (rounds 1-2 lost the
#     %ERRORLEVEL% expansion); cert judgment also reads the content lines from the printed stdout.
# Window: grace 90 s -> stop+drain -> op cert -> tower cert -> drain -> restore $liveCmd detached.
$root     = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$exe      = Join-Path $root 'infernix-serve.exe'
$port     = 8888
$freeMin  = 20000
$repo     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$repoExe  = Join-Path $repo 'build\tests\Release'
$testTow  = Join-Path $repoExe 'infernix_image_tower_provider_real_test.exe'
$testFpa  = Join-Path $repoExe 'infernix_linear_fp8_a16_test.exe'
$modelDir = Join-Path $root 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram    = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
$imageOracle     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle'
$imageOraclePipe = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle_512_mainecoon\tensors_pipeline'
$imageArtifact   = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$processorDir    = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
$argLine  = ' "' + $artifact + '" --ngram-volume "' + $ngram + '" --host 127.0.0.1 --port 8888 --max-context 200000 --kv-capacity auto --kv-dtype fp8 --default-max-tokens 16384 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --default-thinking-budget 8192 --thinking-budget-policy clamp --vision --vision-offload on --spec mtp --draft-tokens 5 --lm-head-draft --preserve-thinking --request-log-jsonl requests.jsonl --max-concurrency 1 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --prefill-chunk 2048 --log-level info --prefix-cache-file "C:\Users\Micke\Desktop\Infernix FlashNext\prefix-cache-flash-next.bin" --expert-ram-mib 41500  --ngram-draft-tokens 15 --ngram-min-match 12  --log-stats-panel on --host-context-mib 4096'

function FreeMib {
  $v = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ($v) { return [int]$v } else { return -1 }
}

function ListenerPid {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.OwningProcess } else { return $null }
}

function RunTest($testExe, $tag, $ppm, $maxWait) {
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
   'if errorlevel 1 (echo TESTEXIT=1>> "' + $outLog + '") else (echo TESTEXIT=0>> "' + $outLog + '")') | Set-Content -Path $runBat -Encoding ASCII -Force
  $t0 = Get-Date
  $tp = Start-Process -FilePath cmd.exe -ArgumentList ('/c "' + $runBat + '"') -WindowStyle Hidden -PassThru
  $waited = 0
  while (-not $tp.HasExited -and $waited -lt $maxWait) { Start-Sleep -Seconds 5; $waited += 5 }
  $elapsed = [int]((Get-Date) - $t0).TotalSeconds
  if (-not $tp.HasExited) { Write-Host 'TEST_TIMEOUT_KILLING'; try { $tp.Kill() } catch {} }
  $exit = 'TIMEOUT'
  if (Test-Path $outLog) {
    $m = Select-String -Path $outLog -Pattern 'TESTEXIT=(\d+)' | Select-Object -Last 1
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

Write-Host '===== PHASE 2: FP8 A16 OP CERT (image shapes now included) ====='
foreach ($p in @($artifact, $testFpa)) {
  if (-not (Test-Path $p)) { Write-Host ('PREREQ_MISSING ' + $p) }
}
RunTest $testFpa 's8d3fpa16' '' 1800

Write-Host '===== PHASE 3: TOWER CERT (rewritten decoded-weights oracle) ====='
foreach ($p in @($imageOracle, $imageArtifact, $processorDir, $testTow)) {
  if (-not (Test-Path $p)) { Write-Host ('PREREQ_MISSING ' + $p) }
}
RunTest $testTow 's8d3tower' 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\interlude_s8d3tower_mine.ppm' 3000

Write-Host '===== PHASE 4: DRAIN ====='
$held = ListenerPid
if ($held) { taskkill /F /PID $held 2>&1 | Out-Null; Start-Sleep -Seconds 3 }
$f = -1; $j = 0
while ($j -lt 60) { $f = FreeMib; if ($f -ge $freeMin) { break }; Start-Sleep -Seconds 2; $j++ }
Write-Host ('free_mib_after_jobs=' + $f)

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
