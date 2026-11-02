$ErrorActionPreference = 'Continue'
# STEP 8E-nvfp4: the PREVIOUS rollout preserved $liveCmd, which turned out to be the fp8 SPIKE, so it
# re-verified fp8, not production. This run FORCES the production command: line 3 of "start flash next.bat"
# (desktop exe + nvfp4 KV) verbatim + the two image flags. Certifies the interlude on the REAL stack.
# On any failure it restores the ORIGINAL live command (current fp8 spike), never leaving a dead server.
$ErrorActionPreference = 'Continue'
$root     = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$port     = 8888
$freeMin  = 20000
$prodBat  = Join-Path $root 'start flash next.bat'
$repo     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$imageArtifact = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v2.infernix'
$processorDir  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor'
$NL   = [Environment]::NewLine
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

# Extract the serve command line (the infernix-serve.exe line) from the production launcher, verbatim.
$prodLine = ''
$batLines = Get-Content -LiteralPath $prodBat
foreach ($ln in $batLines) { if ($ln -match 'infernix-serve\.exe') { $prodLine = $ln.Trim(); break } }
Write-Host ('PROD_SERVE_LINE_FOUND=' + ($prodLine -ne ''))
Write-Host ('PROD_HAS_NVFP4=' + ($prodLine -match '--kv-dtype\s+nvfp4'))
Write-Host ('PROD_IS_DESKTOP_EXE=' + ($prodLine -notmatch 'build\\apps'))
if ($prodLine -match '--image-artifact') { Write-Host 'WARN prod line already has image flags' }

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
} else { Write-Host 'no listener; nothing to stop' }
Write-Host ('captured live command line (rollback target): ' + $liveCmd)
$i = 0
while ($i -lt 30) { Start-Sleep -Seconds 1; if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { break }; $i++ }
$f = -1; $j = 0
while ($j -lt 60) { $f = FreeMib; if ($f -ge $freeMin) { break }; Start-Sleep -Seconds 2; $j++ }
Write-Host ('free_mib_before_job=' + $f)
Start-Sleep -Seconds 3

Write-Host '===== PHASE 2: RELAUNCH = PROD nvfp4 + IMAGE FLAGS ====='
foreach ($p in @($prodBat, $imageArtifact, $processorDir)) { if (-not (Test-Path -LiteralPath $p)) { Write-Host ('PREREQ_MISSING ' + $p) } }
$flags = ' --image-artifact "' + $imageArtifact + '" --image-text-processor "' + $processorDir + '"'
if ($prodLine -eq '') { Write-Host 'ABORT: no prod line captured'; exit 1 }
$newCmd = $prodLine + $flags
$outLog = Join-Path $root 'rollout_8e_nvfp4.out.log'
$errLog = Join-Path $root 'rollout_8e_nvfp4.err.log'
$bat    = Join-Path $root 'rollout_8e_nvfp4_serve.bat'
('cd /d "' + $root + '"' + $NL + $newCmd + ' > "' + $outLog + '" 2> "' + $errLog + '"') | Set-Content -Path $bat -Encoding ASCII -Force
Write-Host ('launching (detached via explorer, visible console): ' + $newCmd)
Start-Process -FilePath explorer.exe -ArgumentList ('"' + $bat + '"') | Out-Null
$up = WaitHealthy $liveModelId 160
$startedId = CurrentModelId
Write-Host ('nvfp4_image_server_up=' + $up + ' model=' + $startedId)

$imgPass = $false
$chatPass = $false
if ($up) {
  Write-Host '===== PHASE 3: VERIFY UNDER nvfp4 (desktop exe) ====='
  $id = $startedId
  $imgUrl = $base + '/v1/images/generations'
  $r1 = PostJson $imgUrl ('{"model":"'+$id+'","prompt":"a black cat on a windowsill at dusk","size":"64x64","seed":7,"response_format":"b64_json"}') 420
  Write-Host ('IMG1_64x64_HTTP=' + $r1.status + ' CODE=' + (GetCode $r1.body) + ' LEN=' + $r1.body.Length + ' HAS_B64=' + ($r1.body -match '"b64_json"'))
  $rc = PostJson ($base + '/v1/chat/completions') ('{"model":"'+$id+'","messages":[{"role":"user","content":"reply with the single word OK"}],"max_tokens":24}') 300
  Write-Host ('CHAT_AFTER_HTTP=' + $rc.status + ' LEN=' + $rc.body.Length)
  $r3 = PostJson $imgUrl ('{"model":"'+$id+'","prompt":"a lighthouse at night","size":"64x64","seed":9}') 300
  Write-Host ('IMG3_AFTER_CHAT_HTTP=' + $r3.status + ' HAS_B64=' + ($r3.body -match '"b64_json"'))
  $imgPass  = ($r1.status -eq 200) -and ($r1.body -match '"b64_json"')
  $chatPass = ($rc.status -eq 200)
} else {
  Write-Host 'PHASE 3 SKIPPED: nvfp4 image server never became healthy'
}

Write-Host '===== PHASE 4: DECIDE ====='
if ($up -and $imgPass -and $chatPass) {
  Write-Host 'NVFP4_ROLLOUT_OK: desktop exe + nvfp4 KV + image route live AND interlude restored LLM. Update durable launcher "start flash next.bat" now.'
} else {
  Write-Host 'NVFP4_ROLLOUT_FAILED: restoring the ORIGINAL live command (fp8 spike) so a working server remains'
  $held = ListenerPid
  if ($held) { taskkill /F /PID $held 2>&1 | Out-Null; Start-Sleep -Seconds 5 }
  $rbat = Join-Path $root 'restore_serve_8e_nvfp4.bat'
  $rb = if ($liveCmd -ne '') { $liveCmd } else { '"' + $prodLine + '"' }
  ('cd /d "' + $root + '"' + $NL + $rb) | Set-Content -Path $rbat -Encoding ASCII -Force
  Start-Process -FilePath explorer.exe -ArgumentList ('"' + $rbat + '"') | Out-Null
  $ok2 = WaitHealthy $liveModelId 160
  Write-Host ('RESTORED_ORIGINAL_SERVER_UP=' + $ok2 + ' model=' + (CurrentModelId))
}
Write-Host ('RUN END ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss,ff'))
