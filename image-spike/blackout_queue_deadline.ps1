param(
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 25,  # grace + cert + restore, announced to whoever is waiting
    [int]$GraceSec = 90              # the cert exe is prebuilt here, so 15 keeps the downtime shorter
)

$ErrorActionPreference = 'Continue'
# BLACKOUT WINDOW FOR THE ADMISSION-QUEUE DEADLINE CERT.
# Modelled on blackout_step8d3_test.ps1: grace -> stop :8888 + drain -> run ONE Debug cert exe -> drain
# -> restore the server detached via explorer and prove it is healthy on the same model id.
# Nothing here builds or relinks infernix-serve.exe; the existing Desktop exe is restarted as it was.
$root     = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$exe      = Join-Path $root 'infernix-serve.exe'
$port     = 8888
$freeMin  = 20000
$repo     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$repoExe  = Join-Path $repo 'build\tests\Debug'
$testQ    = Join-Path $repoExe 'infernix_qwen4_exp_image_queue_deadline_real_test.exe'
$vcppBin  = 'C:\vcpkg\installed\x64-windows\bin'
$modelDir = Join-Path $root 'models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix'
$artifact = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix'
$ngram    = Join-Path $modelDir 'Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'

function FreeMib {
  $v = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ($v) { return [int]$v } else { return -1 }
}

function ListenerPid {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.OwningProcess } else { return $null }
}

function RunTest($testExe, $tag, $maxWait) {
  if (-not (Test-Path $testExe)) { Write-Host ('TEST_EXE_MISSING ' + $testExe); return }
  $outLog = Join-Path $root ('interlude_' + $tag + '_test.out.log')
  $errLog = Join-Path $root ('interlude_' + $tag + '_test.err.log')
  $runBat = Join-Path $root ('interlude_' + $tag + '_run.bat')
  if (Test-Path $outLog) { Remove-Item $outLog -Force -ErrorAction SilentlyContinue }
  if (Test-Path $errLog) { Remove-Item $errLog -Force -ErrorAction SilentlyContinue }
  $env:INFERNIX_QWEN4_ARTIFACT = $artifact
  $env:INFERNIX_QWEN4_NGRAM    = $ngram
  ('@echo off' + [Environment]::NewLine +
   'cd /d "' + $root + '"' + [Environment]::NewLine +
   'set "PATH=' + $vcppBin + ';%PATH%"' + [Environment]::NewLine +
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
  Write-Host ('--- ' + $tag + ' STDERR (tail 60) ---')
  if (Test-Path $errLog) { Get-Content $errLog -Tail 60 | ForEach-Object { Write-Host $_ } }
}

Write-Host '===== PHASE -2: CAPTURE THE LIVE SERVER (once the window opens, :8888 is the 503 shield) ====='
$liveModelId = ''
$liveCmd     = ''
$c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($c) {
  try {
    $m = (Invoke-WebRequest -Uri ('http://127.0.0.1:' + $port + '/v1/models') -UseBasicParsing -TimeoutSec 4).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { $liveModelId = $Matches[1] }
  } catch { $liveModelId = '' }
  $pp = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess) -ErrorAction SilentlyContinue
  if ($pp) { $liveCmd = $pp.CommandLine }
  Write-Host ('captured pid=' + $c.OwningProcess + ' model=' + $liveModelId)
} else {
  Write-Host 'no listener; nothing to capture'
}
Write-Host ('captured live command line: ' + $liveCmd)

Write-Host '===== PHASE -1: WINDOW GATE ====='
# :8888 carries every other Claude Code session's turns, so the port is taken only when it is quiet.
$gate = Join-Path $repo 'image-spike\window_gate.ps1'
if (-not (Test-Path $gate)) { Write-Host ('GATE_MISSING ' + $gate); exit 2 }
& $gate -Action StartWindow -Owner 'blackout_queue_deadline.ps1' -Reason 'blackout_queue_deadline.ps1 (koen-deadline-cert)' -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Write-Host 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }

Write-Host ('===== PHASE 0: GRACE ' + $GraceSec + 's =====')
Start-Sleep -Seconds $GraceSec

Write-Host '===== PHASE 1: STOP THE SERVE PROCESS + DRAIN ====='
# By PROCESS, never by listener: while the window is open the :8888 listener IS the gate's 503 shield, and
# killing the listener would take the shield down and leave the server holding the GPU.
$servePids = @(Get-Process -Name infernix-serve -ErrorAction SilentlyContinue | ForEach-Object { [int]$_.Id })
if ($servePids.Count -gt 0) {
  taskkill /F /T /PID ($servePids -join ' /PID ') 2>&1 | Out-Null
  Write-Host ('killed serve pid=' + ($servePids -join ','))
} else {
  Write-Host 'no infernix-serve process was running'
}
$i = 0
while ($i -lt 40) {
  if (@(Get-Process -Name infernix-serve -ErrorAction SilentlyContinue).Count -eq 0) { break }
  Start-Sleep -Seconds 2
  $i++
}
Write-Host ('serve process gone, listener now=' + (ListenerPid))
$f = -1; $j = 0
while ($j -lt 60) {
  $f = FreeMib
  if ($f -ge $freeMin) { break }
  Start-Sleep -Seconds 2
  $j++
}
Write-Host ('free_mib_before_job=' + $f)
Start-Sleep -Seconds 3

Write-Host '===== PHASE 2: ADMISSION-QUEUE DEADLINE CERT ====='
foreach ($p in @($artifact, $testQ, $vcppBin)) {
  if (-not (Test-Path $p)) { Write-Host ('PREREQ_MISSING ' + $p) }
}
RunTest $testQ 'qdeadline' 900

Write-Host '===== PHASE 3: DRAIN ====='
# The cert exe is the only job here, and the :8888 listener is now the shield: EndWindow is what lets it
# go. Killing that listener would leave the window lock held by a dead owner.
$f = -1; $j = 0
while ($j -lt 60) { $f = FreeMib; if ($f -ge $freeMin) { break }; Start-Sleep -Seconds 2; $j++ }
Write-Host ('free_mib_after_jobs=' + $f)

# The 503 stub must let go of :8888 before serve is relaunched, or the relaunch cannot bind.
& $gate -Action EndWindow | Out-Null
for ($k = 0; $k -lt 20; $k++) {
  if (-not (ListenerPid)) { break }
  Start-Sleep -Seconds 2
}

Write-Host '===== PHASE 4: ENSURE A HEALTHY SERVER ON :8888 ====='
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
  if ($liveCmd -eq '') {
    Write-Host 'NO LIVE COMMAND CAPTURED - not starting a bare-argument server; the user starts it themselves'
  } else {
    Write-Host 'no listener after the defer; restarting the captured command line detached via explorer'
    $bat = Join-Path $root 'restore_serve.bat'
    ('cd /d "' + $root + '"' + [Environment]::NewLine + $liveCmd) | Set-Content -Path $bat -Encoding ASCII -Force
    Start-Process -FilePath explorer.exe -ArgumentList ('"' + $bat + '"') | Out-Null
    $spawned = 'detached'
  }
}
$ok = WaitHealthy
$finalId = CurrentModelId
Write-Host ('spawned=' + $spawned + ' now_listening=' + (ListenerPid) + ' model=' + $finalId + ' server_up=' + $ok)
if ($ok) { Write-Host ('SERVER_RESTORED_OK (model ' + $finalId + ' matches pre-window ' + $liveModelId + ')') }
else      { Write-Host ('SERVER_RESTORED_FAILED (expected ' + $liveModelId + ', got ' + $finalId + ')') }
& $gate -Action Release | Out-Null
Write-Host ('RUN END ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss,ff'))
