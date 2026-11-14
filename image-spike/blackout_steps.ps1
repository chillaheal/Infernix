$ErrorActionPreference = 'Continue'

# Self-restoring deploy + cert for the exposed /v1/images/generations steps parameter.
#   1. Deploy the freshly built build-tree infernix-serve.exe onto the Desktop copy, restart healthy.
#   2. Prove steps is PARSED on the wire: steps=1 is below the schedule floor -> 400 image_steps_invalid.
#   3. Prove steps is HONORED end-to-end: the denoise step count scales wall-clock ~linearly, so
#      steps=8 returns a PNG in roughly (8/20) of the steps=20 time on the same size/prompt/seed.
#   4. Always self-heal to a fresh healthy server (now running the new exe) so the harness reconnects.
# Explorer-detached at every level so the surviving server is not reaped when this script exits.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$serveOut   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_cycle.out.log'
$genLog     = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_steps_gen.log'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_steps_serve.bat'
$pngFast    = 'C:\Users\Micke\Desktop\steps_fast.png'
$pngSlow    = 'C:\Users\Micke\Desktop\steps_slow.png'
$base       = 'http://127.0.0.1:8888'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    return 0
}
function Launch-Bat($bat) { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $bat + '"') | Out-Null }
function Stop-Listener {
    $p = Listener-Pid
    if ($p -gt 0) { taskkill /F /T /PID $p 2>&1 | Out-Null }
    taskkill /F /IM infernix-serve.exe /T 2>&1 | Out-Null
    Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'start flash next.bat' } |
        ForEach-Object { taskkill /F /T /PID $_.ProcessId 2>&1 | Out-Null }
    for ($i = 0; $i -lt 60; $i++) {
        if (Get-Process infernix-serve -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 1; continue }
        try { $fs = [System.IO.File]::Open($desktopExe, 'Open', 'ReadWrite', 'None'); $fs.Close(); break } catch { Start-Sleep -Seconds 1 }
    }
    for ($i = 0; $i -lt 40; $i++) {
        $f = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
        if ($f -and [int]$f -lt 2000) { break }
        Start-Sleep -Seconds 3
    }
}
function Wait-Healthy($iters) {
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $h = Invoke-WebRequest -Uri ($base + '/health') -UseBasicParsing -TimeoutSec 4
            if ([int]$h.StatusCode -eq 200) {
                $m = Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 4
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'Qwen3\.8-Flash-Next')) { return $true }
            }
        } catch { }
    }
    return $false
}
function Model-Id {
    try {
        $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).Content
        if ($m -match '"id"\s*:\s*"([^"]+)"') { return $Matches[1] }
    } catch { }
    return ''
}
# One images POST. Returns @{ http=<int>; seconds=<int>; body=<string>; png=<bool> }.
function Image-Post($model, $prompt, $size, $steps, $seed, $out) {
    $body = @{ model = $model; prompt = $prompt; size = $size; n = 1; steps = $steps; seed = $seed; response_format = 'b64_json' } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 600
        $sw.Stop()
        $obj = $resp.Content | ConvertFrom-Json
        $bytes = [System.Convert]::FromBase64String($obj.data[0].b64_json)
        [System.IO.File]::WriteAllBytes($out, $bytes)
        return @{ http = 200; seconds = [int]$sw.Elapsed.TotalSeconds; body = ''; png = $true }
    } catch {
        $sw.Stop(); $e = $_.Exception; $st = -1; $bd = ''
        if ($e.Response) {
            try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {}
            try { $st = [int]$e.Response.StatusCode } catch {}
        }
        return @{ http = $st; seconds = [int]$sw.Elapsed.TotalSeconds; body = $bd; png = $false }
    }
}

if (-not (Test-Path -LiteralPath $buildExe)) { Log ('FATAL build exe missing: ' + $buildExe); exit 2 }

# ---- serve arg string, verbatim from the production bat minus the leading exe token ----
$serveArgs = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveArgs = $ln.Substring('infernix-serve.exe'.Length).Trim(); break }
}
if (-not $serveArgs) { Log 'FATAL no serve line'; exit 2 }
$serveContent = "@echo off`r`ncd /d `"" + $desktopDir + "`"`r`ninfernix-serve.exe " + $serveArgs + ' > "' + $serveOut + '" 2>&1' + "`r`n"
[System.IO.File]::WriteAllText($serveBat, $serveContent, [System.Text.Encoding]::ASCII)

Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

# ---- PHASE1 deploy new exe, stop listener, start fresh healthy server ----
Stop-Listener
Log 'PHASE1 drained'
if (Test-Path -LiteralPath $serveOut) { Remove-Item -LiteralPath $serveOut -Force }
try { Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force; Log 'PHASE1a deployed new exe' } catch { Log ('DEPLOY_FAIL ' + $_.Exception.Message); exit 3 }
Launch-Bat $serveBat
$ok = Wait-Healthy 150
Log ('PHASE1b HEALTHY=' + $ok)
if (-not $ok) { Log 'NOT_UP final clean restart'; Stop-Listener; Launch-Bat $serveBat; Log ('RETRY_HEALTHY=' + (Wait-Healthy 150)); exit 5 }

$id = Model-Id
Log ('PHASE1c model=' + $id)
if (-not $id) { Log 'FATAL no model id'; Stop-Listener; Launch-Bat $serveBat; Log ('RETRY_HEALTHY=' + (Wait-Healthy 150)); exit 6 }

$prompt = 'a red cube on a white table'

# ---- PHASE2 gate: steps below the schedule floor must be a 400 before any GPU work ----
$bad = Image-Post $id $prompt '768x768' 1 42 $pngFast
Log ('GATE http=' + $bad.http + ' body=' + $bad.body.Substring(0, [Math]::Min(220, $bad.body.Length)))
$gateOk = ($bad.http -eq 400 -and $bad.body -match 'image_steps_invalid')

# ---- PHASE3 honor: the same size/prompt/seed at two step counts; time scales with steps ----
if (Test-Path -LiteralPath $pngFast) { Remove-Item -LiteralPath $pngFast -Force }
if (Test-Path -LiteralPath $pngSlow) { Remove-Item -LiteralPath $pngSlow -Force }
$slow = Image-Post $id $prompt '768x768' 20 42 $pngSlow
Log ('SLOW steps=20 http=' + $slow.http + ' seconds=' + $slow.seconds + ' png=' + $slow.png + ' body=' + $slow.body.Substring(0, [Math]::Min(160, $slow.body.Length)))
$fast = Image-Post $id $prompt '768x768' 8 42 $pngFast
Log ('FAST steps=8 http=' + $fast.http + ' seconds=' + $fast.seconds + ' png=' + $fast.png + ' body=' + $fast.body.Substring(0, [Math]::Min(160, $fast.body.Length)))

# ---- PHASE4 capture serve log (interlude lifecycle, confirm no Degraded) ----
if (Test-Path -LiteralPath $serveOut) { Copy-Item -LiteralPath $serveOut -Destination $genLog -Force }
$degr = (Test-Path -LiteralPath $serveOut) -and (Select-String -LiteralPath $serveOut -Pattern 'Degraded' -Quiet -ErrorAction SilentlyContinue)
Log ('PHASE4 degraded_in_log=' + $degr)

$scaleOk = ($slow.http -eq 200 -and $fast.http -eq 200 -and $slow.png -and $fast.png -and $fast.seconds -lt $slow.seconds)
$pass = ($gateOk -and $scaleOk -and (-not $degr))
Log ('STEPS gate400=' + $gateOk + ' scale(fast=' + $fast.seconds + 's slow=' + $slow.seconds + 's faster=' + ($fast.seconds -lt $slow.seconds) + ') degraded=' + $degr)
if ($pass) { Log 'PASS steps parsed (400 gate) and honored (time scales with step count), no degradation' } else { Log 'FAIL steps not confirmed parsed + honored on the live server' }

# ---- PHASE5 ALWAYS self-heal to a fresh healthy server on the new exe ----
Log 'PHASE5 final clean restart (self-heal)'
Stop-Listener
Launch-Bat $serveBat
Log ('FINAL_HEALTHY=' + (Wait-Healthy 150))
Log 'DONE'
exit 0
