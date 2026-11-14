$ErrorActionPreference = 'Continue'

# DIAGNOSTIC 2: cross the image interlude over an ACTIVE large LLM PREFILL, capture the exact
# interlude exception text (carried in the 503 body as image_engine_degraded -> error.what()),
# then ALWAYS self-heal to a fresh healthy server. Self-restoring, explorer-detached at every level.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$serveOut   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_cycle.out.log'
$genLog     = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_diag2_gen.log'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_diag2_serve.bat'
$genPs1     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\gen_katt.ps1'
$png        = 'C:\Users\Micke\Desktop\Infernix FlashNext\maine-coon-katt.png'

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
    for ($i = 0; $i -lt 30; $i++) { if ((Listener-Pid) -eq 0) { break }; Start-Sleep -Seconds 2 }
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
            $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 4
            if ([int]$h.StatusCode -eq 200) {
                $m = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 4
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'Qwen3\.8-Flash-Next')) { return $true }
            }
        } catch { }
    }
    return $false
}
function Model-Id {
    try {
        $m = (Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 8).Content
        if ($m -match '"id"\s*:\s*"([^"]+)"') { return $Matches[1] }
    } catch { }
    return ''
}
function Start-BigPrefill($id, $prompt) {
    return Start-Job -ScriptBlock {
        param($mid, $p)
        $b = @{ model = $mid; messages = @(@{ role = 'user'; content = $p }); max_tokens = 16; temperature = 0 } | ConvertTo-Json -Depth 6 -Compress
        try {
            $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/chat/completions' -Method POST -ContentType 'application/json' -Body $b -UseBasicParsing -TimeoutSec 420
            return 'LLM_HTTP=' + [int]$r.StatusCode
        } catch {
            $s = -1; $bd = ''
            if ($_.Exception.Response) {
                try { $sr = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream()); $bd = $sr.ReadToEnd() } catch { }
                try { $s = [int]$_.Exception.Response.StatusCode } catch { }
            }
            return 'LLM_HTTP=' + $s + ' BODY=' + $bd
        }
    } -ArgumentList $id, $prompt
}

# ---- serve line ----
$serveArgs = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveArgs = $ln.Substring('infernix-serve.exe'.Length).Trim(); break }
}
if (-not $serveArgs) { Log 'FATAL no serve line'; exit 2 }
$serveContent = "@echo off`r`ncd /d `"" + $desktopDir + "`"`r`ninfernix-serve.exe " + $serveArgs + ' > "' + $serveOut + '" 2>&1' + "`r`n"
[System.IO.File]::WriteAllText($serveBat, $serveContent, [System.Text.Encoding]::ASCII)

Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

Stop-Listener
if (Test-Path -LiteralPath $serveOut) { Remove-Item -LiteralPath $serveOut -Force }
Launch-Bat $serveBat
$ok = Wait-Healthy 130
Log ('PHASE1 HEALTHY=' + $ok)
if (-not $ok) {
    Log 'NOT_UP clean restart'
    Stop-Listener
    Launch-Bat $serveBat
    Log ('RETRY_HEALTHY=' + (Wait-Healthy 130))
    exit 5
}

# ---- PHASE2: start a big-prefill LLM request, then cross the interlude over it ----
$id = Model-Id
$prompt = ('The quick brown fox jumps over the lazy dog near the riverbank under a pale sky. ' * 4500)
Log ('PHASE2 prefill prompt_chars=' + $prompt.Length)
$job = Start-BigPrefill $id $prompt
$active = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Milliseconds 700
    if ((Test-Path -LiteralPath $serveOut) -and (Select-String -LiteralPath $serveOut -Pattern 'prefill' -Quiet -ErrorAction SilentlyContinue)) { $active = $true; break }
}
Log ('PHASE2 prefill_active=' + $active)
if (Test-Path -LiteralPath $png) { Remove-Item -LiteralPath $png -Force }
$g = (& $genPs1 -Size '768x768' -Out $png 2>&1) -join "`n"
Log ('GEN| ' + ($g -replace "`n", ' | '))
Log ('GEN_PNG=' + (Test-Path -LiteralPath $png))

$llm = ''
try { $llm = (Receive-Job -Job $job -Wait -Timeout 60) -join ' ; ' } catch { $llm = 'RECEIVE_TIMEOUT' }
Log ('LLM| ' + $llm)
try { Remove-Job -Job $job -Force } catch { }

# ---- PHASE3 capture serve log ----
if (Test-Path -LiteralPath $serveOut) { Copy-Item -LiteralPath $serveOut -Destination $genLog -Force }
Log 'PHASE3 serve log'
if (Test-Path -LiteralPath $serveOut) {
    Get-Content -LiteralPath $serveOut -Tail 130 | ForEach-Object { Log ('TAIL| ' + $_) }
    Select-String -LiteralPath $serveOut -Pattern 'interlude|yielded|restoration|Degraded|submit door|lease|frontier|out of memory|CUDA|KV pool grew' -ErrorAction SilentlyContinue |
        Select-Object -Last 50 | ForEach-Object { Log ('GREP| ' + $_.Line) }
} else { Log 'NO_SERVE_LOG' }

# ---- PHASE4 ALWAYS self-heal to a fresh healthy server ----
Log 'PHASE4 final clean restart'
Stop-Listener
Launch-Bat $serveBat
Log ('FINAL_HEALTHY=' + (Wait-Healthy 150))
Log ('RESULT png=' + (Test-Path -LiteralPath $png) + ' genlog=' + (Test-Path -LiteralPath $genLog))
exit 0
