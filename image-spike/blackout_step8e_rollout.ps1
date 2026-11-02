$ErrorActionPreference = 'Continue'
# STEP 8E ROLLOUT (not a test): relaunch the LIVE serve WITH the two image flags, preserving production
# nvfp4 KV, then verify the interlude under nvfp4. The image interlude is CERTIFIED only under fp8 KV +
# the BUILD exe (8D round-2); the production server is nvfp4 KV + the DESKTOP exe, so nvfp4 is UNVERIFIED.
# A failing verification RESTORES the original flagless command; the script never leaves a sick server.
# Flags copied verbatim from serve_s8d2.bat (a real working launcher), never reconstructed.
$root   = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$exe    = Join-Path $root 'infernix-serve.exe'
$port   = 8888
$freeMin = 20000
$repo   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$imageArtifact = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$processorDir  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
$NL = [Environment]::NewLine
$base = 'http://127.0.0.1:' + $port

function FreeMib {
  $v = (nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
  if ($v) { return [int]$v } else { return -1 }
}
function ListenerPid {
  $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.OwningProcess } else { return $null }
}
function CurrentModelId {
  try {
    $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 4).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { return $Matches[1] }
  } catch {}
  return ''
}
function WaitHealthy($expect, $rounds) {
  for ($t = 0; $t -lt $rounds; $t++) {
    $id = CurrentModelId
    if ($id -ne '' -and ($expect -eq '' -or $id -eq $expect)) { return $true }
    Start-Sleep -Seconds 3
  }
  return $false
}
function PostJson($url, $json, $timeout) {
  $r = @{ status = 0; body = '' }
  try {
    $resp = Invoke-WebRequest -Uri $url -Method POST -ContentType 'application/json' -Body $json -UseBasicParsing -TimeoutSec $timeout
    $r.status = [int]$resp.StatusCode; $r.body = $resp.Content
  } catch {
    $ex = $_.Exception
    if ($ex.Response) {
      try { $sr = New-Object System.IO.StreamReader($ex.Response.GetResponseStream()); $r.body = $sr.ReadToEnd(); $sr.Close() } catch {}
      try { $r.status = [int]$ex.Response.StatusCode } catch { $r.status = -1 }
    } else { $r.status = -1; $r.body = $ex.Message }
  }
  return $r
}
function GetCode($body) { if ($body -match '"code"\s*:\s*"([^"]+)"') { return $Matches[1] } return '' }

Write-Host '===== PHASE 0: GRACE 90s ====='
Start-Sleep -Seconds 90

Write-Host '===== PHASE 1: STOP live server + DRAIN ====='
$liveModelId = ''
$liveCmd     = ''
$c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($c) {
  $p = $c.OwningProcess
  try {
    $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 4).Content
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
while ($i -lt 30) { Start-Sleep -Seconds 1; if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { break }; $i++ }
Write-Host 'port free'
$f = -1; $j = 0
while ($j -lt 60) { $f = FreeMib; if ($f -ge $freeMin) { break }; Start-Sleep -Seconds 2; $j++ }
Write-Host ('free_mib_before_job=' + $f)
Start-Sleep -Seconds 3

Write-Host '===== PHASE 2: RELAUNCH WITH IMAGE FLAGS (nvfp4 KV preserved) ====='
foreach ($p in @($imageArtifact, $processorDir)) { if (-not (Test-Path $p)) { Write-Host ('PREREQ_MISSING ' + $p) } }
$flags = ' --image-artifact "' + $imageArtifact + '" --image-text-processor "' + $processorDir + '"'
if ($liveCmd -eq '') { $cmdToRun = '"' + $exe + '"' } else { $cmdToRun = $liveCmd }
$newCmd = $cmdToRun + $flags
$outLog = Join-Path $root 'rollout_8e.out.log'
$errLog = Join-Path $root 'rollout_8e.err.log'
$bat    = Join-Path $root 'rollout_8e_serve.bat'
('cd /d "' + $root + '"' + $NL + $newCmd + ' > "' + $outLog + '" 2> "' + $errLog + '"') | Set-Content -Path $bat -Encoding ASCII -Force
Write-Host ('launching (detached via explorer, visible console): ' + $newCmd)
Start-Process -FilePath explorer.exe -ArgumentList ('"' + $bat + '"') | Out-Null
$up = WaitHealthy $liveModelId 140
$startedId = CurrentModelId
Write-Host ('image_server_up=' + $up + ' model=' + $startedId)

$imgPass = $false
$chatPass = $false
if ($up) {
  Write-Host '===== PHASE 3: VERIFY UNDER nvfp4 ====='
  $id = $startedId
  $imgUrl = $base + '/v1/images/generations'
  $r1 = PostJson $imgUrl ('{"model":"'+$id+'","prompt":"a black cat on a windowsill at dusk","size":"64x64","seed":7,"response_format":"b64_json"}') 300
  Write-Host ('IMG1_64x64_HTTP=' + $r1.status + ' CODE=' + (GetCode $r1.body) + ' LEN=' + $r1.body.Length + ' HAS_B64=' + ($r1.body -match '"b64_json"'))
  $r2 = PostJson $imgUrl ('{"model":"'+$id+'","prompt":"a red fox in snow","size":"832x1216","seed":7}') 600
  Write-Host ('IMG2_832x1216_HTTP=' + $r2.status + ' CODE=' + (GetCode $r2.body) + ' LEN=' + $r2.body.Length + ' HAS_B64=' + ($r2.body -match '"b64_json"'))
  $rc = PostJson ($base + '/v1/chat/completions') ('{"model":"'+$id+'","messages":[{"role":"user","content":"reply with the single word OK"}],"max_tokens":24}') 300
  Write-Host ('CHAT_AFTER_HTTP=' + $rc.status + ' LEN=' + $rc.body.Length)
  $r3 = PostJson $imgUrl ('{"model":"'+$id+'","prompt":"a lighthouse at night","size":"64x64","seed":9}') 300
  Write-Host ('IMG3_AFTER_CHAT_HTTP=' + $r3.status + ' HAS_B64=' + ($r3.body -match '"b64_json"'))
  $imgPass  = ($r1.status -eq 200) -and ($r1.body -match '"b64_json"')
  $chatPass = ($rc.status -eq 200)
} else {
  Write-Host 'PHASE 3 SKIPPED: image server never became healthy'
}

Write-Host '===== PHASE 4: DECIDE ====='
if ($up -and $imgPass -and $chatPass) {
  Write-Host 'ROLLOUT_OK: image route live + interlude restored LLM under nvfp4. Durable launcher "start flash next.bat" NOT yet updated (do it after this confirms).'
} else {
  Write-Host 'ROLLOUT_FAILED: rolling back to the original flagless command'
  $held = ListenerPid
  if ($held) { taskkill /F /PID $held 2>&1 | Out-Null; Start-Sleep -Seconds 5 }
  $rbat = Join-Path $root 'restore_serve_8e.bat'
  $rb = if ($liveCmd -ne '') { $liveCmd } else { '"' + $exe + '"' }
  ('cd /d "' + $root + '"' + $NL + $rb) | Set-Content -Path $rbat -Encoding ASCII -Force
  Start-Process -FilePath explorer.exe -ArgumentList ('"' + $rbat + '"') | Out-Null
  $ok2 = WaitHealthy $liveModelId 140
  Write-Host ('ROLLED_BACK_FLAGLESS_SERVER_UP=' + $ok2 + ' model=' + (CurrentModelId))
}
Write-Host ('RUN END ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss,ff'))
