param(
    [string]$ExpertRamMib = '8000',    # '41500' runs the user's real host-expert budget
    [string]$LogOnly = '',             # re-judge an existing serve log; touches no server at all
    [switch]$LogDebug,                 # run the experiment serve with --log-level debug
    [int]$GateMaxWaitSec = 0,          # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 20     # the experiment plus putting the user's server back
)

$ErrorActionPreference = 'Continue'

# Reproduction, in a blackout window and NOT against the server the user is running: start the HEAD
# build's serve binary directly (the build tree exe, its own working directory, its own log), let it
# hold an image interlude, and submit an ordinary chat request while the interlude is in flight.
#
# The question this answers: does the LLM get admitted while the image job owns the yielded VRAM?
# The contract (the prod cert's own gate) says the submit door is closed for the whole job, so the
# chat request must WAIT. If it runs instead, the two lanes are on the device at once and the image
# job is the one that OOMs -- which is what took the deployed process down at 08:02.
#
# --expert-ram-mib is set to 8000 for the experiment on purpose: the production value (41500) makes
# the picture fail earlier, at the pinned block cache (defect 1), so no interlude would ever be long
# enough to cross. With 8 GiB of host expert tier the page-locked block cache fits, the interlude
# runs its full length, and the admission question is the only thing left under test.
#
# The window always closes the same way: the experiment serve is stopped and the user's own start bat
# is relaunched, so the Desktop instance is back serving before this script prints its verdict.

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$buildExe   = Join-Path $repo 'build\apps\Release\infernix-serve.exe'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$startBat   = Join-Path $desktopDir 'start flash next.bat'
$runDir     = Join-Path $repo 'image-spike\repro_run'
$serveBat   = Join-Path $repo 'image-spike\_repro_serve.bat'
$backBat    = Join-Path $repo 'image-spike\_repro_back.bat'
$serveLog   = Join-Path $runDir 'serve_repro.log'
$chatOut    = Join-Path $runDir 'chat_result.txt'
$imgOut     = Join-Path $runDir 'image_result.txt'
$imgJson    = Join-Path $runDir 'image_request.json'
$chatJson   = Join-Path $runDir 'chat_request.json'
$port       = 8888
$base       = 'http://127.0.0.1:8888'

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
function Stop-On-Port {
    $p = Listener-Pid
    if ($p -gt 0) { taskkill /F /T /PID $p 2>&1 | Out-Null; Log ('stopped pid=' + $p) }
    taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
    for ($i = 0; $i -lt 30; $i++) {
        if ((Listener-Pid) -eq 0) { break }
        Start-Sleep -Seconds 2
    }
}
function Wait-Healthy($iters) {
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $r = Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5
            if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { return $true }
        } catch { }
    }
    return $false
}

# The verdict, from the server's own timestamps rather than from any client. Written as a function so
# an existing serve log can be re-judged with -LogOnly without stopping or starting a server.
function Get-InterludeVerdict($logPath) {
    if (-not (Test-Path -LiteralPath $logPath)) { Log 'NO_SERVE_LOG'; return 'NO_SERVE_LOG' }
    $fmt = 'yyyy-MM-dd HH:mm:ss.fff'
    $imageLines = @()
    $reqLines   = @()
    Get-Content -LiteralPath $logPath | ForEach-Object {
        if ($_ -match '^(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d\d\d).*image job: ') { $imageLines += $_ }
        if ($_ -match '^(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d\d\d).*req#\d+ (started|done)') { $reqLines += $_ }
    }
    $imageStart = $null; $imageEnd = $null
    if ($imageLines.Count -gt 0) {
        $imageStart = [datetime]::ParseExact(($imageLines[0] -split '  ')[0], $fmt, $null)
        $imageEnd   = [datetime]::ParseExact(($imageLines[-1] -split '  ')[0], $fmt, $null)
    }
    Log ('WINDOW image job lines=' + $imageLines.Count + ' first=' + $imageStart + ' last=' + $imageEnd)
    # Pair each "req#N started" with its "req#N done" before judging anything. Admission is
    # started + queue, NOT done - queue: the done line carries the stamp of when the request ENDED, so
    # subtracting the queue lands somewhere inside the decode. Live proof 2026-11-01, same 512-token
    # chat body with and without a picture: 9.6 s end to end at a 0.10 s queue in the control, versus
    # 55.3 s at a 43.2 s queue behind the picture. done - queue placed that request 31 s into a job it
    # in fact waited out to the last 0.02 s; started + queue puts it 0.02 s before the job's end.
    $startedAt = @{}
    foreach ($ln in $reqLines) {
        if ($ln -match 'req#(\d+) started') {
            $startedAt[[int]$Matches[1]] = [datetime]::ParseExact(($ln -split '  ')[0], $fmt, $null)
        }
    }
    $admitted = $false
    foreach ($ln in $reqLines) {
        $ts = [datetime]::ParseExact(($ln -split '  ')[0], $fmt, $null)
        $what = $ln.Substring(24, [Math]::Min(120, $ln.Length - 24))
        if ($ln -notmatch 'req#\d+ done') {
            Log ('HANDLER| ts=' + $ts + ' | ' + $what)
            continue
        }
        if ($ln -notmatch 'queue ([0-9.]+)s') {
            Log ('DONE| ts=' + $ts + ' no engine queue time | ' + $what)
            continue
        }
        $q = [double]$Matches[1]
        if ($ln -notmatch 'req#(\d+) done') {
            Log ('DONE| ts=' + $ts + ' no request id to pair with a started line | ' + $what)
            continue
        }
        $reqId = [int]$Matches[1]
        if (-not $startedAt.ContainsKey($reqId)) {
            Log ('ADMIT| req#' + $reqId + ' has no started line, admission cannot be placed | ' + $what)
            continue
        }
        $admission = $startedAt[$reqId].AddSeconds($q)
        $inWindow = [bool]($imageStart -and $admission -gt $imageStart -and $imageEnd -and $admission -lt $imageEnd)
        if ($inWindow) { $admitted = $true }
        Log ('ADMIT| queued_s=' + $q + ' admission=' + $admission.ToString($fmt) +
             ' inside_interlude=' + $inWindow + ' | ' + $what)
    }
    Get-Content -LiteralPath $logPath | Where-Object { $_ -match 'image job: |req#|yield: |restored|out of memory|CUDA_CHECK|retention' } |
        ForEach-Object { Log ('LOG| ' + $_) }
    if ($admitted) { return 'ADMITTED_DURING_INTERLUDE' }
    if ($imageEnd) { return 'QUEUED_CORRECTLY' }
    return 'NO_IMAGE_JOB'
}

if ($LogOnly) {
    Write-Host ((Get-Date).ToString('HH:mm:ss') + ' VERDICT ' + (Get-InterludeVerdict $LogOnly))
    exit 0
}

foreach ($p in @($buildExe, $startBat)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}
if (-not (Test-Path -LiteralPath $runDir)) { New-Item -ItemType Directory -Path $runDir | Out-Null }

# The user's own flags, with the host expert tier shrunk so the interlude can be reached at all.
$serveLine = ''
foreach ($ln in (Get-Content -LiteralPath $startBat)) {
    if ($ln -match '^\s*infernix-serve\.exe') { $serveLine = $ln.Trim(); break }
}
if (-not $serveLine) { Log 'FATAL no serve line in the start bat'; exit 2 }
$flags = $serveLine.Substring('infernix-serve.exe'.Length).Trim()
$flags = $flags -replace '--expert-ram-mib [0-9]+', ('--expert-ram-mib ' + $ExpertRamMib)
if ($LogDebug) { $flags = $flags + ' --log-level debug' }  # the ledger's pin room is a Debug line
Log ('FLAGS expert-ram-mib set to ' + $ExpertRamMib + ' debug=' + $LogDebug + ' patched=' +
     [bool]($flags -match ('--expert-ram-mib ' + $ExpertRamMib)))

# The window gate. This script takes :8888 for itself even though it serves the experiment on its own
# build, so it asks first: -LogOnly above never reaches this line and needs no gate.
$gate = Join-Path $repo 'image-spike\window_gate.ps1'
if (-not (Test-Path -LiteralPath $gate)) { Log ('FATAL missing gate: ' + $gate); exit 2 }
# -NoShield: this window binds :8888 itself with the experiment serve, so a 503 stub would compete
# with it. It still gets the marker and the lock.
& $gate -Action StartWindow -NoShield -Owner (Split-Path -Leaf $MyInvocation.MyCommand.Path) -Reason 'repro_interlude_admission.ps1 (eget test-serve pa :8888)' -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }

Log 'PHASE0 grace 30s'
Start-Sleep -Seconds 30
Log ('PHASE1 stopping whatever serves :' + $port)
Stop-On-Port
for ($i = 0; $i -lt 40; $i++) {
    $u = Gpu-UsedMib
    if ($u -ge 0 -and $u -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log ('PHASE1 drained gpu_used_mib=' + (Gpu-UsedMib))

if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
Set-Content -LiteralPath $serveBat -Value (@('@echo off', ('cd /d "' + $runDir + '"'),
                                            ('"' + $buildExe + '" ' + $flags + ' > "' + $serveLog + '" 2>&1'))) -Encoding ASCII
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
Log 'PHASE2 HEAD build launched in its own directory (visible window)'
$up = Wait-Healthy 150
Log ('PHASE2 healthy=' + $up)

$verdict = 'NO_SERVE'
if ($up) {
    $chatId = ''; $imageId = ''
    try {
        $models = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
        $ids = @($models.data | ForEach-Object { $_.id })
        $chatId = [string]$ids[0]
        foreach ($i in $ids) { if ([string]$i -ne $chatId) { $imageId = [string]$i } }
    } catch { Log ('MODELS_ERR ' + $_.Exception.Message) }
    Log ('IDS chat=' + $chatId + ' image=' + $imageId)

    @{ model = $imageId; prompt = 'A red fox curled up asleep on pale snow at dawn, soft fur detail, photorealistic';
       size = '512x512'; n = 1; response_format = 'url' } | ConvertTo-Json -Compress |
        Set-Content -LiteralPath $imgJson -Encoding ASCII
    @{ model = $chatId; max_tokens = 512; messages = @(@{ role = 'user';
         content = 'List fifteen short facts about the RTX 5090, one per line, with a number in front.' }) } |
        ConvertTo-Json -Depth 5 -Compress | Set-Content -LiteralPath $chatJson -Encoding ASCII

    # The picture first: it takes the yield and holds the interlude for its whole length.
    $imgJob = Start-Job -ScriptBlock {
        param($base, $jsonFile, $outFile)
        $body = Get-Content -LiteralPath $jsonFile -Raw
        $t0 = Get-Date
        $line = ''
        try {
            $r = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 900
            $line = 'IMAGE HTTP=' + [int]$r.StatusCode + ' seconds=' + [int]((Get-Date) - $t0).TotalSeconds + ' body=' + $r.Content.Substring(0, [Math]::Min(160, $r.Content.Length))
        } catch {
            $line = 'IMAGE HTTP=ERR seconds=' + [int]((Get-Date) - $t0).TotalSeconds + ' msg=' + $_.Exception.Message
        }
        Set-Content -LiteralPath $outFile -Value $line -Encoding ASCII
    } -ArgumentList $base, $imgJson, $imgOut

    Start-Sleep -Seconds 5
    # The chat request lands while that interlude owns the yielded VRAM.
    $chatJob = Start-Job -ScriptBlock {
        param($base, $jsonFile, $outFile)
        $body = Get-Content -LiteralPath $jsonFile -Raw
        $t0 = Get-Date
        $line = ''
        try {
            $r = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 900
            $o = $r.Content | ConvertFrom-Json
            $line = 'CHAT HTTP=' + [int]$r.StatusCode + ' seconds=' + [int]((Get-Date) - $t0).TotalSeconds + ' finish=' + $o.choices[0].finish_reason
        } catch {
            $line = 'CHAT HTTP=ERR seconds=' + [int]((Get-Date) - $t0).TotalSeconds + ' msg=' + $_.Exception.Message
        }
        Set-Content -LiteralPath $outFile -Value $line -Encoding ASCII
    } -ArgumentList $base, $chatJson, $chatOut

    Wait-Job -Timeout 700 -Id $imgJob.Id, $chatJob.Id | Out-Null
    Receive-Job -Id $imgJob.Id, $chatJob.Id -Wait -AutoRemoveJob | ForEach-Object { Log ('JOB| ' + $_) }
    if (Test-Path -LiteralPath $imgOut)  { Log ('IMG| '  + (Get-Content -LiteralPath $imgOut -Raw).Trim()) }
    if (Test-Path -LiteralPath $chatOut) { Log ('CHAT| ' + (Get-Content -LiteralPath $chatOut -Raw).Trim()) }

    $verdict = Get-InterludeVerdict $serveLog
}

Log 'PHASE3 closing the experiment and putting the user server back'
Stop-On-Port
& $gate -Action EndWindow | Out-Null   # marker down, lock released, before the user's start bat runs
Set-Content -LiteralPath $backBat -Value (@('@echo off', ('cd /d "' + $desktopDir + '"'), ('"' + $startBat + '"'))) -Encoding ASCII
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $backBat + '"') | Out-Null
$back = Wait-Healthy 150
& $gate -Action Release | Out-Null
Log ('RESULT repro=' + $verdict + ' user_server_back=' + $back + ' pid=' + (Listener-Pid))
exit 0
