$ErrorActionPreference = 'Continue'

# #82 + #83 deploy: build the serve that carries the public image surface (image model id,
# response_format url, GET /v1/images/{id}, GET /v1/capabilities) and the server-executed chat image
# tool, install it on the Desktop instance, and certify the new surface, a plain chat regression, and a
# chat request that must come back holding a fetchable picture URL, against the LIVE server.
# Self-restoring: every failure path relaunches the captured live server command.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe'
$startBat   = 'C:\Users\Micke\Desktop\Infernix FlashNext\start flash next.bat'
$buildExe   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-serve.exe'
$buildDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build'
$cmake      = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'
$spikeDir   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike'
$serveLog   = 'C:\Users\Micke\Desktop\Infernix FlashNext\serve_deploy_url.log'
$serveBat   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_deploy_url_serve.bat'
$restoreBat = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_deploy_url_restore.bat'
$buildLog   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\deploy_url_build.log'
$pngCert    = 'C:\Users\Micke\Desktop\Infernix FlashNext\deploy-url-cert.png'
$pngChatAuto = 'C:\Users\Micke\Desktop\Infernix FlashNext\chat-auto-image-cert.png'
$base       = 'http://127.0.0.1:8888'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) { if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break } }
if (-not $serveLine) { Log 'FATAL no serve line in start bat'; exit 2 }
Log ('SERVE_LINE_OK len=' + $serveLine.Length + ' images=' + [bool]($serveLine -match '--image-artifact'))

Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30

# ---- PHASE 1: capture live cmd, stop it, drain VRAM ----
$c = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$livePid = 0; $liveCmd = ''
if ($c) {
    $livePid = [int]$c.OwningProcess
    try { $liveCmd = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $livePid)).CommandLine } catch { $liveCmd = '' }
}
Log ('LIVE pid=' + $livePid + ' cmdlen=' + $liveCmd.Length)
if ($livePid -gt 0) {
    taskkill /F /T /PID $livePid 2>&1 | Out-Null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        if (-not (Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue)) { break }
    }
}
for ($i = 0; $i -lt 40; $i++) {
    $f = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if ($f -and [int]$f -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log 'PHASE1 drained'

function Restore-Live {
    if (-not $liveCmd) { Log 'RESTORE: no live cmd (nothing was running)'; return }
    Set-Content -LiteralPath $restoreBat -Value (@('@echo off', ('cd /d "' + $desktopDir + '"'), $liveCmd)) -Encoding ASCII
    Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $restoreBat + '"') | Out-Null
    Log 'RESTORE: live command relaunched via explorer (survives this script, visible window)'
}

# ---- PHASE 2: build serve (Release) ----
Log 'PHASE2 build infernix-serve'
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
& $cmake --build $buildDir --target infernix-serve --config Release -j 6 2>&1 | Tee-Object -FilePath $buildLog | Select-Object -Last 20
$buildRc = $LASTEXITCODE
Log ('PHASE2 build rc=' + $buildRc)
if ($buildRc -ne 0) {
    Log 'BUILD_FAILED: restoring live server'
    Restore-Live
    exit 3
}

# ---- PHASE 3+4: deploy + launch ----
Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force
Log ('PHASE3 copied exe size=' + (Get-Item -LiteralPath $desktopExe).Length)
if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
$serveCmd = ('"' + $desktopExe + '" ' + $serveLine.Substring('infernix-serve.exe'.Length).Trim() + ' > "' + $serveLog + '" 2>&1')
Set-Content -LiteralPath $serveBat -Value @('@echo off', ('cd /d "' + $desktopDir + '"'), $serveCmd) -Encoding ASCII
function Kill-Serve {
    $s = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($s) { taskkill /F /T /PID ([int]$s.OwningProcess) 2>&1 | Out-Null; Log ('killed serve pid=' + $s.OwningProcess) }
    taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
    Start-Sleep -Seconds 5
}
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
Log 'PHASE4 serve launched via explorer (survives this script, visible window)'

function Server-Up {
    try { return ([int](Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200) } catch { return $false }
}
$healthy = $false
for ($i = 0; $i -lt 130; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5
        if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { $healthy = $true; break }
    } catch { }
}
Log ('PHASE4 healthy=' + $healthy)
if (-not $healthy) {
    Log 'SERVER_NOT_UP: restoring live server'
    Kill-Serve
    Restore-Live
    exit 4
}

# ---- PHASE 5: certify the new public surface ----
$chatId = ''; $imageId = ''; $okModels = $false
try {
    $models = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    $ids = @($models.data | ForEach-Object { $_.id })
    $chatId = [string]$ids[0]
    $imageId = ''
    foreach ($i in $ids) { if ([string]$i -ne $chatId) { $imageId = [string]$i } }
    $okModels = (($ids.Count -ge 2) -and ($imageId -ne '') -and ($imageId -ne $chatId))
} catch { Log ('MODELS_ERR ' + $_.Exception.Message) }
Log ('MODELS ids=' + ($ids -join ',') + ' ok=' + $okModels)

$okCaps = $false
try {
    $caps = (Invoke-WebRequest -Uri ($base + '/v1/capabilities') -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    $okCaps = (($caps.object -eq 'infernix.capabilities') -and
               ($null -ne $caps.image_generation) -and
               ([string]$caps.image_generation.model -eq $imageId) -and
               (@($caps.image_generation.response_formats) -contains 'url') -and
               ($caps.image_generation.automatic_from_chat -eq $true) -and
               ([string]$caps.image_generation.chat_tool -eq 'generate_image') -and
               ([int]$caps.image_generation.chat_max_images_per_request -ge 1))
    Log ('CAPS object=' + $caps.object + ' image_model=' + $caps.image_generation.model +
         ' formats=' + (@($caps.image_generation.response_formats) -join '+') +
         ' automatic_from_chat=' + $caps.image_generation.automatic_from_chat +
         ' chat_tool=' + $caps.image_generation.chat_tool + ' ok=' + $okCaps)
} catch { Log ('CAPS_ERR ' + $_.Exception.Message) }

# response_format=url round trip: the delivered URL must fetch the PNG back from this same server.
$okUrl = $false; $url = ''
$prompt = 'A red fox curled up asleep on pale snow at dawn, soft fur detail, photorealistic'
$body = @{ model = $imageId; prompt = $prompt; size = '512x512'; n = 1; response_format = 'url' } | ConvertTo-Json -Compress
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 900
    $obj = $resp.Content | ConvertFrom-Json
    $url = [string]$obj.data[0].url
    $sw.Stop()
    Log ('GENURL HTTP=200 SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' url=' + $url)
    if ($url) {
        $got = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 30
        $bytes = $got.Content
        $isPng = ($bytes.Length -gt 8 -and $bytes[0] -eq 137 -and $bytes[1] -eq 80 -and $bytes[2] -eq 78 -and $bytes[3] -eq 71)
        [System.IO.File]::WriteAllBytes($pngCert, $bytes)
        $okUrl = ($isPng -and $bytes.Length -gt 10000 -and ($got.Headers['Cache-Control'] -match 'no-store'))
        Log ('FETCHURL HTTP=' + [int]$got.StatusCode + ' BYTES=' + $bytes.Length + ' PNG=' + $isPng +
             ' CACHE=' + $got.Headers['Cache-Control'] + ' ok=' + $okUrl)
    }
} catch {
    $sw.Stop(); $e = $_.Exception; $st = -1; $bd = ''
    if ($e.Response) {
        try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {}
        try { $st = [int]$e.Response.StatusCode } catch {}
    }
    Log ('GENURL HTTP=' + $st + ' SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' BODY=' + $bd)
}

# 404 for an id that is not an artifact id (name can never become a path)
$ok404 = $false
try {
    $r404 = Invoke-WebRequest -Uri ($base + '/v1/images/not-an-id') -UseBasicParsing -TimeoutSec 10
    Log ('ART404 HTTP=' + [int]$r404.StatusCode)
} catch {
    $st404 = -1
    try { $st404 = [int]$_.Exception.Response.StatusCode } catch {}
    $ok404 = ($st404 -eq 404)
    Log ('ART404 HTTP=' + $st404 + ' ok=' + $ok404)
}

# chat regression: the model gate still accepts the chat id. The image gen above closes the submit door
# for its whole interlude, so a 503 request-queue-timeout here is contention, not a defect: retry.
$okChat = $false
$chatBody = @{ model = $chatId; messages = @(@{ role = 'user'; content = 'Reply with exactly: OK' }); max_tokens = 16 } | ConvertTo-Json -Depth 5 -Compress
for ($try = 1; $try -le 6; $try++) {
    try {
        $rc = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body $chatBody -UseBasicParsing -TimeoutSec 300
        $oc = $rc.Content | ConvertFrom-Json
        $okChat = ([int]$rc.StatusCode -eq 200 -and $oc.choices.Count -ge 1)
        Log ('CHAT try=' + $try + ' HTTP=' + [int]$rc.StatusCode + ' finish=' + $oc.choices[0].finish_reason + ' ok=' + $okChat)
        if ($okChat) { break }
    } catch {
        $stc = -1
        try { $stc = [int]$_.Exception.Response.StatusCode } catch {}
        Log ('CHAT try=' + $try + ' HTTP=' + $stc + ' err=' + $_.Exception.Message)
    }
    Start-Sleep -Seconds 20
}

# the image id must NOT be valid on the chat route (chat accepts only the chat id): 404 model_not_found
$okChatGate = $false
try {
    $rb = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body (@{ model = $imageId; messages = @(@{ role = 'user'; content = 'x' }) } | ConvertTo-Json -Depth 5 -Compress) -UseBasicParsing -TimeoutSec 60
    Log ('CHATGATE HTTP=' + [int]$rb.StatusCode)
} catch {
    $stg = -1
    try { $stg = [int]$_.Exception.Response.StatusCode } catch {}
    $okChatGate = ($stg -eq 404)
    Log ('CHATGATE HTTP=' + $stg + ' ok=' + $okChatGate)
}

# #83: a plain chat request, no tools field, asks for a picture. The server runs its hidden
# generate_image tool inside this same request, so the client must get one assistant answer that
# carries a fetchable artifact URL and no trace of a tool call or tool result.
$okChatAuto = $false
# max_tokens must leave room for the thinking budget the server inherits (--default-thinking-budget 8192):
# with 96 the whole budget went to reasoning, content came back empty and no tool round could start.
$autoBody = @{ model = $chatId; max_tokens = 2048; messages = @(
    @{ role = 'user'; content = 'Draw a small picture of a red fox curled up on pale snow and give me the link to it.' }
) } | ConvertTo-Json -Depth 5 -Compress
$swAuto = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $ra = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body $autoBody -UseBasicParsing -TimeoutSec 900
    $oa = $ra.Content | ConvertFrom-Json
    $answer = [string]$oa.choices[0].message.content
    $finish = [string]$oa.choices[0].finish_reason
    $swAuto.Stop()
    $autoUrl = ''
    if ($answer -match '(http://[^ ""''<>)]+/v1/images/[A-Za-z0-9._-]+)') { $autoUrl = $matches[1] }
    $sawToolCall = ($null -ne $oa.choices[0].message.tool_calls)
    Log ('CHATAUTO HTTP=' + [int]$ra.StatusCode + ' SECONDS=' + [int]$swAuto.Elapsed.TotalSeconds +
         ' finish=' + $finish + ' tool_calls=' + (-not (-not $sawToolCall)) +
         ' tokens=' + $oa.usage.prompt_tokens + '/' + $oa.usage.completion_tokens +
         ' answer=' + $answer.Substring(0, [Math]::Min(160, $answer.Length)))
    if ($autoUrl) {
        $ra2 = Invoke-WebRequest -Uri $autoUrl -UseBasicParsing -TimeoutSec 30
        $bytes2 = $ra2.Content
        $isPng2 = ($bytes2.Length -gt 8 -and $bytes2[0] -eq 137 -and $bytes2[1] -eq 80 -and $bytes2[2] -eq 78 -and $bytes2[3] -eq 71)
        [System.IO.File]::WriteAllBytes($pngChatAuto, $bytes2)
        $okChatAuto = ($isPng2 -and $bytes2.Length -gt 10000 -and (-not $sawToolCall) -and ($finish -ne 'tool_calls'))
        Log ('CHATAUTOFETCH HTTP=' + [int]$ra2.StatusCode + ' BYTES=' + $bytes2.Length +
             ' PNG=' + $isPng2 + ' ok=' + $okChatAuto)
    } else {
        Log 'CHATAUTO no artifact URL in the answer'
    }
} catch {
    $swAuto.Stop(); $e = $_.Exception; $sta = -1; $bda = ''
    if ($e.Response) {
        try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bda = $sr.ReadToEnd(); $sr.Close() } catch {}
        try { $sta = [int]$e.Response.StatusCode } catch {}
    }
    Log ('CHATAUTO HTTP=' + $sta + ' SECONDS=' + [int]$swAuto.Elapsed.TotalSeconds + ' BODY=' + $bda)
}

# ---- PHASE 6: serve log tail ----
if (Test-Path -LiteralPath $serveLog) {
    Get-Content -LiteralPath $serveLog -Tail 40 | ForEach-Object { Log ('SVRLOG| ' + $_) }
} else { Log 'NO_SERVE_LOG' }

# ---- PHASE 7: decide ----
$allOk = ($okModels -and $okCaps -and $okUrl -and $ok404 -and $okChat -and $okChatGate -and $okChatAuto)
if ($allOk) {
    if (-not (Server-Up)) {
        Log 'PHASE7: server dropped during gen; relaunching so a working server remains'
        Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
        for ($i = 0; $i -lt 130; $i++) { Start-Sleep -Seconds 5; if (Server-Up) { break } }
    }
    Log ('RESULT: SUCCESS models=' + $okModels + ' caps=' + $okCaps + ' url=' + $okUrl +
         ' notfound=' + $ok404 + ' chat=' + $okChat + ' chatgate=' + $okChatGate +
         ' chatauto=' + $okChatAuto + ' - flash-next left running')
    exit 0
} else {
    Log ('RESULT: FAILED models=' + $okModels + ' caps=' + $okCaps + ' url=' + $okUrl +
         ' notfound=' + $ok404 + ' chat=' + $okChat + ' chatgate=' + $okChatGate + ' - restoring live server')
    Kill-Serve
    Start-Sleep -Seconds 10
    Restore-Live
    exit 5
}
