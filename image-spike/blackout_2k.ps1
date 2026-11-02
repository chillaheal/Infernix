$ErrorActionPreference = 'Continue'

# Self-restoring deploy + cert for MAX-RESOLUTION support (spatial_attn online-softmax rewrite,
# S caps 16384 -> 20480, spatial_attn N cap 16384 -> 65536):
#   1. Deploy the freshly built build-tree infernix-serve.exe onto the Desktop copy, restart healthy.
#   2. Small sanity gen (768x768 steps=4) proves the deploy itself still generates.
#   3. Cert: 2048x2048 steps=2 returns a PNG in HTTP 200 (this exact size died mid-request on the
#      O(N) shared-memory spatial_attn kernel before the rewrite).
#   4. Always self-heal to a fresh healthy server (now running the new exe) so the harness reconnects.
# Explorer-detached at every level so the surviving server is not reaped when this script exits.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$serveOut   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_cycle.out.log'
$genLog     = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_2k_gen.log'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_2k_serve.bat'
$pngSmall   = 'C:\Users\Micke\Desktop\2k_sanity.png'
$pngBig     = 'C:\Users\Micke\Desktop\2k_cert.png'
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
function Image-Post($model, $prompt, $size, $steps, $seed, $out, $tosec) {
    $body = @{ model = $model; prompt = $prompt; size = $size; n = 1; steps = $steps; seed = $seed; response_format = 'b64_json' } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec $tosec
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

# ---- PHASE1q device tests in the server-free window (VAE decode scratch now retires per buffer,
# so the prod-route numerical oracle + attention tests need the whole card). ----
$env:INFERNIX_IMAGE_ORACLE = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle'
$testDir = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release'
foreach ($t in @('infernix_image_vae_spatial_attn_test', 'infernix_image_vae_prod_cuda_test')) {
    $out = (& (Join-Path $testDir ($t + '.exe')) 2>&1 | Out-String)
    $code = $LASTEXITCODE
    $tailLines = ($out -split "`r?`n" | Where-Object { $_ -ne '' } | Select-Object -Last 3) -join ' | '
    Log ('TEST ' + $t + ' exit=' + $code + ' :: ' + $tailLines)
}

Launch-Bat $serveBat
$ok = Wait-Healthy 150
Log ('PHASE1b HEALTHY=' + $ok)
if (-not $ok) { Log 'NOT_UP final clean restart'; Stop-Listener; Launch-Bat $serveBat; Log ('RETRY_HEALTHY=' + (Wait-Healthy 150)); exit 5 }

$id = Model-Id
Log ('PHASE1c model=' + $id)
if (-not $id) { Log 'FATAL no model id'; Stop-Listener; Launch-Bat $serveBat; Log ('RETRY_HEALTHY=' + (Wait-Healthy 150)); exit 6 }

$prompt = 'a red cube on a white table'

# ---- PHASE2 small sanity: the deploy still generates at a known-good size ----
$small = Image-Post $id $prompt '768x768' 4 42 $pngSmall 900
Log ('SMALL 768x768 steps=4 http=' + $small.http + ' seconds=' + $small.seconds + ' png=' + $small.png + ' body=' + $small.body.Substring(0, [Math]::Min(220, $small.body.Length)))

# ---- PHASE3 cert: the max resolution 2048x2048 end-to-end ----
$big = Image-Post $id $prompt '2048x2048' 2 42 $pngBig 1800
Log ('BIG 2048x2048 steps=2 http=' + $big.http + ' seconds=' + $big.seconds + ' png=' + $big.png + ' body=' + $big.body.Substring(0, [Math]::Min(220, $big.body.Length)))

# ---- PHASE4 capture serve log (interlude lifecycle, confirm no Degraded) ----
if (Test-Path -LiteralPath $serveOut) { Copy-Item -LiteralPath $serveOut -Destination $genLog -Force }
$degr = (Test-Path -LiteralPath $serveOut) -and (Select-String -LiteralPath $serveOut -Pattern 'Degraded' -Quiet -ErrorAction SilentlyContinue)
Log ('PHASE4 degraded_in_log=' + $degr)

$pass = ($small.http -eq 200 -and $small.png -and $big.http -eq 200 -and $big.png -and (-not $degr))
Log ('CERT2K small(200=' + ($small.http -eq 200) + ') big(200=' + ($big.http -eq 200) + ' png=' + $big.png + ' seconds=' + $big.seconds + ') degraded=' + $degr)
if ($pass) { Log 'PASS 2048x2048 generated end-to-end on the max-resolution engine' } else { Log 'FAIL 2048x2048 not confirmed on the live server' }

# ---- PHASE5 ALWAYS self-heal to a fresh healthy server on the new exe ----
Log 'PHASE5 final clean restart (self-heal)'
Stop-Listener
Launch-Bat $serveBat
Log ('FINAL_HEALTHY=' + (Wait-Healthy 150))
Log 'DONE'
exit 0
