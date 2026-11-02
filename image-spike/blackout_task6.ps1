param(
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 30   # how long this window announces that it holds the port
)

$ErrorActionPreference = 'Continue'

# Bundle C task 6: deploy the serve that carries the chat `image` argument and the generated-image
# retention sweep, then certify both in the DELIVERED process (not in a host test).
#
# Phases: capture and stop the live :8888, back up the Desktop exe, build the Release app target,
# copy it to the Desktop instance, seed two retention probes while the server is down, start the
# server with the start bat's own command line, and only then run the live gates:
#
#   REF        POST /v1/images/generations response_format=url -> the delivered URL fetches a PNG
#   RETENTION  a picture older than the retention that no job record owns is gone after start-up,
#              a fresh one is kept (this is the sweep running in the shipped process)
#   CHATIMAGE  a chat request that makes the model call generate_image with the tool's `image`
#              argument set to the URL just drawn; the answer must carry a NEW fetchable URL
#   CHATATTACH the same with `attachment:0` against a request that carries an uploaded picture
#
# Self-restoring: every failure path puts the backed up exe back and relaunches the command line the
# server was actually running with. A successful run leaves the NEW build serving :8888.

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = Join-Path $desktopDir 'infernix-serve.exe'
$backupExe  = Join-Path $desktopDir 'infernix-serve.exe.pre-task6.bak'
$startBat   = Join-Path $desktopDir 'start flash next.bat'
$buildExe   = Join-Path $repo 'build\apps\Release\infernix-serve.exe'
$buildDir   = Join-Path $repo 'build'
$cmake      = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'
$imagesDir  = Join-Path $desktopDir 'images'
$oldProbe   = Join-Path $imagesDir 'img-9998-9.png'
$freshProbe = Join-Path $imagesDir 'img-9997-9.png'
$serveLog   = Join-Path $desktopDir 'serve_task6.log'
$serveBat   = Join-Path $repo 'image-spike\_task6_serve.bat'
$restoreBat = Join-Path $repo 'image-spike\_task6_restore.bat'
$buildLog   = Join-Path $repo 'image-spike\deploy_task6_build.log'
$pngRef     = Join-Path $desktopDir 'task6-reference.png'
$pngChat    = Join-Path $desktopDir 'task6-chat-image.png'
$pngAttach  = Join-Path $desktopDir 'task6-chat-attachment.png'
$base       = 'http://127.0.0.1:8888'
$port       = 8888

# Write-Host (the information stream) rather than the success stream: the gate helpers below RETURN
# values, and a log line leaking into that return value would corrupt them.
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
function Kill-Serve {
    $p = Listener-Pid
    if ($p -gt 0) { taskkill /F /T /PID $p 2>&1 | Out-Null; Log ('killed serve pid=' + $p) }
    taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
    for ($i = 0; $i -lt 30; $i++) {
        if ((Listener-Pid) -eq 0) { break }
        Start-Sleep -Seconds 2
    }
}
function Server-Healthy($iters) {
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $r = Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5
            if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { return $true }
        } catch { }
    }
    return $false
}

$liveCmd = ''
$deployed = $false
function Restore-Old {
    # Put the backed up binary back before anything is relaunched, so the fallback really is the
    # server that was running before this script.
    & $gate -Action EndWindow | Out-Null   # let go of :8888 before anything is started again
    if ($deployed -and (Test-Path -LiteralPath $backupExe)) {
        Copy-Item -LiteralPath $backupExe -Destination $desktopExe -Force
        Log ('RESTORE: the backed up exe is back in place, size=' + (Get-Item -LiteralPath $backupExe).Length)
    }
    if (-not $liveCmd) { Log 'RESTORE: no live command was captured, nothing to relaunch'; return }
    Set-Content -LiteralPath $restoreBat -Value (@('@echo off', ('cd /d "' + $desktopDir + '"'), $liveCmd)) -Encoding ASCII
    Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $restoreBat + '"') | Out-Null
    Log 'RESTORE: the captured live command was relaunched (detached, visible window)'
}

# ---- PHASE 0: every path this script needs must exist before anything is stopped ----
foreach ($p in @($desktopExe, $startBat, $buildDir, $cmake)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}
$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break }
}
if (-not $serveLine) { Log 'FATAL no serve line in the start bat'; exit 2 }
Log ('SERVE_LINE_OK len=' + $serveLine.Length + ' image_artifact=' + [bool]($serveLine -match '--image-artifact'))

# The window gate. :8888 carries every other Claude Code session's turns, so the port is taken only
# when it is quiet; the gate refuses (exit 10) unless it is, or unless -GateForce says otherwise.
$gate = Join-Path $repo 'image-spike\window_gate.ps1'
if (-not (Test-Path -LiteralPath $gate)) { Log ('FATAL missing gate: ' + $gate); exit 2 }
& $gate -Action StartWindow -Owner (Split-Path -Leaf $MyInvocation.MyCommand.Path) -Reason 'blackout_task6.ps1 (deploy + cert)' -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }

Log 'PHASE0 grace 30s (let any in-flight request finish before the server goes down)'
Start-Sleep -Seconds 30

# ---- PHASE 1: capture and stop the live server ----
$livePid = Listener-Pid
if ($livePid -gt 0) {
    try { $liveCmd = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $livePid)).CommandLine } catch { $liveCmd = '' }
}
Log ('PHASE1 live pid=' + $livePid + ' cmdlen=' + $liveCmd.Length)
if ($livePid -gt 0) { Kill-Serve }
for ($i = 0; $i -lt 40; $i++) {
    $u = Gpu-UsedMib
    if ($u -ge 0 -and $u -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log ('PHASE1 drained gpu_used_mib=' + (Gpu-UsedMib))

# ---- PHASE 2: back up the binary that is serving today ----
if (-not (Test-Path -LiteralPath $backupExe)) {
    Copy-Item -LiteralPath $desktopExe -Destination $backupExe -Force
    Log ('PHASE2 backed up the live exe to ' + $backupExe + ' size=' + (Get-Item -LiteralPath $backupExe).Length)
} else {
    Log ('PHASE2 a backup already exists, keeping it: size=' + (Get-Item -LiteralPath $backupExe).Length)
}

# ---- PHASE 3: build the Release serve ----
Log 'PHASE3 build infernix-serve (Release)'
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
& $cmake --build $buildDir --target infernix-serve --config Release -j 6 2>&1 | Tee-Object -FilePath $buildLog | Select-Object -Last 15
$buildRc = $LASTEXITCODE
Log ('PHASE3 build rc=' + $buildRc)
if ($buildRc -ne 0 -or -not (Test-Path -LiteralPath $buildExe)) {
    Log 'BUILD_FAILED: nothing was copied, restoring the live server'
    Restore-Old
    exit 3
}

# ---- PHASE 4: deploy, and seed the retention probes while the server is down ----
Copy-Item -LiteralPath $buildExe -Destination $desktopExe -Force
$deployed = $true
Log ('PHASE4 deployed size=' + (Get-Item -LiteralPath $desktopExe).Length + ' mtime=' + (Get-Item -LiteralPath $desktopExe).LastWriteTime)
if (-not (Test-Path -LiteralPath $imagesDir)) { New-Item -ItemType Directory -Path $imagesDir | Out-Null }
$pngMagic = [byte[]]@(137, 80, 78, 71, 13, 10, 26, 10)
[System.IO.File]::WriteAllBytes($oldProbe, $pngMagic)
(Get-Item -LiteralPath $oldProbe).LastWriteTime = (Get-Date).AddDays(-20)
[System.IO.File]::WriteAllBytes($freshProbe, $pngMagic)
Log ('PHASE4 probes seeded old=' + (Get-Item -LiteralPath $oldProbe).LastWriteTime +
     ' fresh=' + (Get-Item -LiteralPath $freshProbe).LastWriteTime)

# ---- PHASE 5: start the server with the start bat's own command line ----
& $gate -Action EndWindow | Out-Null   # the 503 stub has to release :8888 before serve binds it again
if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
$serveCmd = ('"' + $desktopExe + '" ' + $serveLine.Substring('infernix-serve.exe'.Length).Trim() + ' > "' + $serveLog + '" 2>&1')
Set-Content -LiteralPath $serveBat -Value (@('@echo off', ('cd /d "' + $desktopDir + '"'), $serveCmd)) -Encoding ASCII
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
Log 'PHASE5 serve launched via explorer (detached, visible window)'
$healthy = Server-Healthy 130
Log ('PHASE5 healthy=' + $healthy)
if (-not $healthy) {
    Log 'SERVER_NOT_UP: restoring the previous binary and command line'
    Kill-Serve
    Restore-Old
    (Server-Healthy 120) | Out-Null
    Log ('RESULT task6 ref=NA chatimage=NA attach=NA retention=NA healthy=' + (Server-Healthy 1) + ' FAILED=startup')
    exit 4
}

# ---- PHASE 6: the retention sweep, observed in the delivered process ----
$oldGone = $false
for ($i = 0; $i -lt 18; $i++) {
    if (-not (Test-Path -LiteralPath $oldProbe)) { $oldGone = $true; break }
    Start-Sleep -Seconds 10
}
$freshKept = Test-Path -LiteralPath $freshProbe
Log ('RETENTION old_probe_gone=' + $oldGone + ' fresh_probe_kept=' + $freshKept)
$ok404Old = $true
try {
    $rOld = Invoke-WebRequest -Uri ($base + '/v1/images/img-9998-9') -UseBasicParsing -TimeoutSec 10
    $ok404Old = $false
    Log ('RETIRED_URL HTTP=' + [int]$rOld.StatusCode + ' (expected 404)')
} catch {
    $st = -1
    try { $st = [int]$_.Exception.Response.StatusCode } catch {}
    $ok404Old = ($st -eq 404)
    Log ('RETIRED_URL HTTP=' + $st + ' ok=' + $ok404Old)
}
if (Test-Path -LiteralPath $freshProbe) { Remove-Item -LiteralPath $freshProbe -Force }

# ---- PHASE 7: the model ids and the chat tool the build advertises ----
$chatId = ''; $imageId = ''; $okModels = $false
try {
    $models = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    $ids = @($models.data | ForEach-Object { $_.id })
    $chatId = [string]$ids[0]; $imageId = ''
    foreach ($i in $ids) { if ([string]$i -ne $chatId) { $imageId = [string]$i } }
    $okModels = (($ids.Count -ge 2) -and ($imageId -ne ''))
} catch { Log ('MODELS_ERR ' + $_.Exception.Message) }
Log ('MODELS ids=' + ($ids -join ',') + ' ok=' + $okModels)
$okCaps = $false
try {
    $caps = (Invoke-WebRequest -Uri ($base + '/v1/capabilities') -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    $okCaps = ([string]$caps.image_generation.chat_tool -eq 'generate_image' -and
               $caps.image_generation.automatic_from_chat -eq $true)
    Log ('CAPS chat_tool=' + $caps.image_generation.chat_tool + ' automatic_from_chat=' +
         $caps.image_generation.automatic_from_chat + ' ok=' + $okCaps)
} catch { Log ('CAPS_ERR ' + $_.Exception.Message) }

function Post-Chat($body, $label, $seconds) {
    # One chat request; returns the object plus the answer text and any new artifact URL. A model that
    # does not call the tool is a finding to report, so the raw answer comes back either way.
    $out = @{ ok = $false; status = -1; answer = ''; url = ''; seconds = 0; body = '' }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 900
        $o = $r.Content | ConvertFrom-Json
        $sw.Stop()
        $out.status = [int]$r.StatusCode
        $out.answer = [string]$o.choices[0].message.content
        $out.seconds = [int]$sw.Elapsed.TotalSeconds
        if ($out.answer -match '(http://[^ ""''<>)]+/v1/images/[A-Za-z0-9._-]+)') { $out.url = $matches[1] }
        $out.ok = ($out.status -eq 200)
    } catch {
        $sw.Stop(); $e = $_.Exception
        if ($e.Response) {
            try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $out.body = $sr.ReadToEnd(); $sr.Close() } catch {}
            try { $out.status = [int]$e.Response.StatusCode } catch {}
        }
        $out.seconds = [int]$sw.Elapsed.TotalSeconds
        Log ($label + ' HTTP=' + $out.status + ' SECONDS=' + $out.seconds + ' BODY=' + $out.body.Substring(0, [Math]::Min(240, $out.body.Length)))
    }
    return $out
}
function Picture-Bytes($url, $saveTo) {
    # Fetch an artifact URL and say whether it really is a picture.
    $res = @{ ok = $false; bytes = 0; png = $false; data = [byte[]]@() }
    try {
        $g = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 30
        $b = $g.Content
        $res.bytes = $b.Length
        $res.png = ($b.Length -gt 8 -and $b[0] -eq 137 -and $b[1] -eq 80 -and $b[2] -eq 78 -and $b[3] -eq 71)
        $res.ok = ($res.png -and $b.Length -gt 10000)
        $res.data = $b
        if ($res.png) { [System.IO.File]::WriteAllBytes($saveTo, $b) }
    } catch { Log ('FETCH_ERR ' + $url + ' ' + $_.Exception.Message) }
    return $res
}

# ---- PHASE 8: the picture the chat legs will refer to ----
$refUrl = ''; $okRef = $false
$body = @{ model = $imageId; prompt = 'A red fox curled up asleep on pale snow at dawn, soft fur detail, photorealistic'; size = '512x512'; n = 1; response_format = 'url' } | ConvertTo-Json -Compress
try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 900
    $obj = $resp.Content | ConvertFrom-Json
    $refUrl = [string]$obj.data[0].url
    $sw.Stop()
    Log ('REF GENURL SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' url=' + $refUrl)
    $got = Picture-Bytes $refUrl $pngRef
    $okRef = $got.ok
    Log ('REF FETCH BYTES=' + $got.bytes + ' PNG=' + $got.png + ' ok=' + $okRef)
} catch {
    $e = $_.Exception; $bd = ''
    if ($e.Response) { try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {} }
    Log ('REF GENURL FAILED BODY=' + $bd.Substring(0, [Math]::Min(240, $bd.Length)))
}

# ---- PHASE 9: chat with the tool's image argument set to that URL ----
$okChatImage = $false
if ($okRef) {
    $ask = 'Use the generate_image tool exactly once. Set its image argument to ' + $refUrl +
           ' and its prompt to "the same scene at night under a moonlit sky, same composition". ' +
           'Answer with only the link to the new picture.'
    $chatBody = @{ model = $chatId; max_tokens = 2048; messages = @(@{ role = 'user'; content = $ask }) } | ConvertTo-Json -Depth 6 -Compress
    $r = Post-Chat $chatBody 'CHATIMAGE' 900
    Log ('CHATIMAGE HTTP=' + $r.status + ' SECONDS=' + $r.seconds + ' url=' + $r.url +
         ' answer=' + $r.answer.Substring(0, [Math]::Min(200, $r.answer.Length)))
    if ($r.url -and ($r.url -ne $refUrl)) {
        $got2 = Picture-Bytes $r.url $pngChat
        $okChatImage = $got2.ok
        Log ('CHATIMAGEFETCH BYTES=' + $got2.bytes + ' PNG=' + $got2.png + ' ok=' + $okChatImage)
    } else {
        Log 'CHATIMAGE no new artifact URL in the answer (the model did not pass the reference, or did not draw)'
    }
}

# ---- PHASE 10: chat with attachment:0 against an uploaded picture ----
$okAttach = $false
if ($okRef) {
    $b64 = [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($pngRef))
    $ask2 = 'The picture attached to this request is my reference. Use the generate_image tool exactly once, ' +
            'with its image argument set to attachment:0 and its prompt to "this same picture in autumn colours". ' +
            'Answer with only the link to the new picture.'
    $parts = @(@{ type = 'text'; text = $ask2 },
               @{ type = 'image_url'; image_url = @{ url = ('data:image/png;base64,' + $b64) } })
    $body2 = @{ model = $chatId; max_tokens = 2048; messages = @(@{ role = 'user'; content = $parts }) } | ConvertTo-Json -Depth 8 -Compress
    Log ('CHATATTACH request bytes=' + $body2.Length)
    $r2 = Post-Chat $body2 'CHATATTACH' 900
    Log ('CHATATTACH HTTP=' + $r2.status + ' SECONDS=' + $r2.seconds + ' url=' + $r2.url +
         ' answer=' + $r2.answer.Substring(0, [Math]::Min(200, $r2.answer.Length)))
    if ($r2.url) {
        $got3 = Picture-Bytes $r2.url $pngAttach
        $okAttach = ($got3.ok -and ($r2.url -ne $refUrl))
        Log ('CHATATTACHFETCH BYTES=' + $got3.bytes + ' PNG=' + $got3.png + ' ok=' + $okAttach)
    } else {
        Log 'CHATATTACH no artifact URL in the answer'
    }
}

# ---- PHASE 11: serve log, then the verdict ----
if (Test-Path -LiteralPath $serveLog) {
    Get-Content -LiteralPath $serveLog -Tail 30 | ForEach-Object { Log ('SVRLOG| ' + $_) }
} else { Log 'NO_SERVE_LOG' }
$stillUp = Server-Healthy 1
& $gate -Action Release | Out-Null
Log ('RESULT task6 ref=' + $okRef + ' chatimage=' + $okChatImage + ' attach=' + $okAttach +
     ' retention_old_gone=' + $oldGone + ' retention_fresh_kept=' + $freshKept +
     ' retired_url_404=' + $ok404Old + ' models=' + $okModels + ' caps=' + $okCaps +
     ' healthy=' + $stillUp + ' - the NEW build is left serving :8888')
exit 0
