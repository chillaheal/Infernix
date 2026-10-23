param(
    [ValidateSet('Status', 'Check', 'Acquire', 'Release', 'StartWindow', 'EndWindow')] [string]$Action = 'Status',
    [int]$Port = 8888,
    [string]$RequestLog = 'C:\Users\Micke\Desktop\Infernix FlashNext\requests.jsonl',
    [string]$LockFile   = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-window.lock',
    # StartWindow also writes the public marker and opens the visible shield window (503 + notice);
    # EndWindow takes the shield down again, so the port is free before serve is relaunched.
    [string]$MarkerFile = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-window.json',
    [string]$Shield     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\window_shield.ps1',
    [string]$ShieldBat  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_shield.bat',
    [string]$Reason     = '',
    [int]$ShieldMaxMinutes = 90,
    [switch]$NoShield,               # marker + lock but no 503 stub: for a window that binds the port itself
    [int]$QuietSec = 120,          # a request_* event newer than this means a session is mid-turn
    [int]$ProbeSec = 12,           # budget for one admission probe (0 = no probe)
    [int]$MaxWaitSec = 0,          # Acquire: wait for quiet up to this long (0 = one check, then refuse)
    [int]$ExpectedMinutes = 30,    # Acquire: how long this window expects to own the port
    [int]$StaleGraceSec = 90,      # a lock whose owner died frees only after this
    [string]$Owner = '',
    [switch]$Force                 # take the port despite BUSY/LOCKED (the user's own call)
)

# The window gate. A blackout window (cert, bench, interlude repro) has to own the GPU, so it stops
# the serve that answers :8888 -- and every other Claude Code session runs its turns against that
# same port. Measured on 2026-10-23: one window left :8888 down from 10:51 to 10:57:46, and another
# session burned nine ECONNREFUSED retries and lost a turn. A window therefore has to ask first.
#
# A second instance on another port is NOT an option on this machine: the live serve holds about
# 31,5 GiB of the 32,6 GiB VRAM (nvidia-smi showed 690 MiB free) and page-locks 41,5 GiB of host
# RAM through --expert-ram-mib, so a second real instance cannot start next to it, and a cert or a
# bench that does not use the device would measure nothing. This gate is the other half of the same
# guarantee: :8888 only goes down when nothing is waiting on it.
#
# What it reads (no serve change, no restart, no new process):
#   * the serve's OWN request log (--request-log-jsonl, same file for every start bat): the newest
#     request_start / request_done / request_error / request_rejected event of the CURRENT server
#     instance. A request_start with no terminal event after it is a turn in flight on :8888.
#     File mtime is useless here: "throughput" records are written on an interval while idle.
#   * one admission probe (max_tokens=1) when the log says quiet: a picture interlude or a busy lane
#     queues even a trivial request, and the queue is exactly what a window would destroy.
#   * a lock file next to the instance, so two windows never overlap and a running window is
#     visible to everyone ("is :8888 coming back, or is it dead?").
#
# Exit codes: 0 = idle / acquired / released / status; 10 = BUSY (another session has work);
# 11 = LOCKED (a window already holds the port); 2 = bad arguments or a missing path.
#
# A caller that is done with the port releases it; a window that dies without releasing is stale
# once its own process is gone (StaleGraceSec) or when ExpectedMinutes has passed.

$ErrorActionPreference = 'Continue'
$base = 'http://127.0.0.1:' + $Port
$fmtIso = 'yyyy-MM-dd HH:mm:ss'

function Say($m) { Write-Host ('GATE| ' + $m) }
function Verdict($v) { Write-Host ('GATE_VERDICT ' + $v) }

function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    return 0
}

function Read-Tail($path, $maxBytes) {
    # Read the end of a log the server is still appending to: open it with FileShare ReadWrite and
    # seek, so a 190 MB file costs a few MB of parsing instead of a full read.
    if (-not (Test-Path -LiteralPath $path)) { return @() }
    $fs = $null
    try {
        $fs = [System.IO.File]::Open($path, 'Open', 'Read', [System.IO.FileShare]::ReadWrite)
        $len = $fs.Length
        $take = [Math]::Min([int64]$len, [int64]$maxBytes)
        if ($take -le 0) { return @() }
        [void]$fs.Seek($len - $take, 'Begin')
        $buf = New-Object byte[] ([int]$take)
        $got = $fs.Read($buf, 0, [int]$take)
        $text = (New-Object System.Text.UTF8Encoding($false)).GetString($buf, 0, $got)
    } catch {
        Say ('LOG_UNREADABLE ' + $path + ' ' + $_.Exception.Message)
        return @()
    } finally {
        if ($fs) { $fs.Dispose() }
    }
    $nl = $text.IndexOf("`n")                       # the first line may start mid-record
    if ($nl -ge 0) { $text = $text.Substring($nl + 1) } else { return @() }
    return ($text -split "`r?`n")
}

function Get-Traffic($tailBytes) {
    # Newest activity of the server instance that is answering right now, and the requests of that
    # instance that have started but not finished. Request ids only count within one server instance,
    # so records from an earlier instance are ignored by server_instance_id, not by age.
    $res = @{ known = $false; instance = ''; open = 0; age_s = -1; events = 0; started_open = @() }
    $lines = Read-Tail $RequestLog $tailBytes
    if ($lines.Count -eq 0) { return $res }
    $nowMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $inst = ''
    $records = @()
    foreach ($ln in $lines) {
        if ($ln -notmatch '"artifact_type":"infernix_serve_request_log"') { continue }
        try { $o = $ln | ConvertFrom-Json } catch { continue }
        $inst = [string]$o.server_instance_id
        $records += $o
    }
    if ($inst -eq '') { return $res }
    $res.instance = $inst
    $seen = @{}
    $newest = 0
    foreach ($o in $records) {
        if ([string]$o.server_instance_id -ne $inst) { continue }
        $ev = [string]$o.event
        if ($ev -notin @('request_start', 'request_done', 'request_error', 'request_rejected')) { continue }
        $res.events += 1
        $ts = [long]$o.timestamp_unix_ms
        if ($ts -gt $newest) { $newest = $ts }
        # The request key is request.http_request_id (both request_start and its terminal record carry
        # it); request.request_id is nested the same way. A top-level .request_id does not exist, so
        # reading it would collapse every record onto one empty key.
        $id = [string]$o.request.http_request_id
        if ($id -eq '') { $id = [string]$o.request.request_id }
        if ($id -eq '') { $id = 'ts-' + [string]$o.timestamp_unix_ms + '-' + $ev }
        if ($ev -eq 'request_start') { $seen[$id] = 'open' } else { $seen[$id] = 'closed' }
    }
    if ($res.events -eq 0) { return $res }
    $res.known = $true
    foreach ($k in $seen.Keys) { if ($seen[$k] -eq 'open') { $res.open += 1; $res.started_open += $k } }
    $res.age_s = [Math]::Round(($nowMs - $newest) / 1000.0, 1)
    return $res
}

function Invoke-AdmissionProbe($budgetSec) {
    # One token of work. If the engine admits it, the port is genuinely free; if it queues, the
    # picture interlude door is closed or a lane is busy, which is precisely what a window kills.
    $out = @{ ok = $false; why = ''; seconds = 0 }
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $models = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 5).Content | ConvertFrom-Json
        $modelId = [string]($models.data | Select-Object -First 1).id
        if ($modelId -eq '') { $out.why = 'no model id'; $sw.Stop(); return $out }
        $body = @{ model = $modelId; max_tokens = 1; messages = @(@{ role = 'user'; content = 'ok' }) } | ConvertTo-Json -Compress
        $r = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' `
                              -Body $body -UseBasicParsing -TimeoutSec $budgetSec
        $sw.Stop()
        $out.seconds = [Math]::Round($sw.Elapsed.TotalSeconds, 1)
        $out.ok = ([int]$r.StatusCode -eq 200)
        if (-not $out.ok) { $out.why = 'HTTP ' + [int]$r.StatusCode }
    } catch {
        $sw.Stop()
        $out.seconds = [Math]::Round($sw.Elapsed.TotalSeconds, 1)
        $why = $_.Exception.Message
        try { if ($_.Exception.Response) { $why = 'HTTP ' + [int]$_.Exception.Response.StatusCode } } catch {}
        $out.why = $why
    }
    return $out
}

function Get-Lock {
    # none / live (someone holds the port) / stale (holder gone or over its announced time)
    $res = @{ state = 'none'; owner = ''; pid = 0; since = ''; until = ''; age_s = -1 }
    if (-not (Test-Path -LiteralPath $LockFile)) { return $res }
    $j = $null
    try { $j = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json } catch { $res.state = 'stale'; return $res }
    $res.owner = [string]$j.owner
    $res.pid = [int]$j.pid
    $startMs = [long]$j.started_unix_ms
    $expectMs = [long]$j.expected_minutes
    $res.until = ([DateTimeOffset]::FromUnixTimeMilliseconds($startMs).LocalDateTime.AddMinutes($expectMs)).ToString($fmtIso)
    $res.age_s = [Math]::Round(([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() - $startMs) / 1000.0, 1)
    $res.since = ([DateTimeOffset]::FromUnixTimeMilliseconds($startMs).LocalDateTime.ToString($fmtIso))
    $overTime = $res.age_s -gt ($expectMs * 60 + $StaleGraceSec)
    $holderGone = ($res.pid -gt 0) -and -not (Get-Process -Id $res.pid -ErrorAction SilentlyContinue)
    if ($overTime) { $res.state = 'stale'; Say ('lock stale: past its announced end ' + $res.until); return $res }
    if ($holderGone -and $res.age_s -gt $StaleGraceSec) {
        $res.state = 'stale'; Say ('lock stale: owner pid=' + $res.pid + ' is gone'); return $res
    }
    $res.state = 'live'
    return $res
}

function Set-Lock($ownerName) {
    $obj = @{ owner = $ownerName; pid = $PID; started_unix_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds();
              expected_minutes = $ExpectedMinutes; port = $Port } | ConvertTo-Json -Compress
    # Exclusive create so two windows racing on the same second cannot both think they won. The
    # four-arg File.WriteAllText overload (string, string, Encoding, FileMode) does not exist on
    # .NET Framework / Windows PowerShell, so the mode goes through FileStream instead.
    $why = ''
    for ($try = 0; $try -lt 5; $try++) {
        $fs = $null
        try {
            $fs = [System.IO.File]::Open($LockFile, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            $bytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes($obj)
            $fs.Write($bytes, 0, $bytes.Length)
            return $true
        } catch {
            $why = $_.Exception.Message
            Start-Sleep -Milliseconds 400
        } finally {
            if ($fs) { $fs.Dispose() }
        }
    }
    Say ('lock create failed: ' + $why)
    return $false
}

function Evaluate([switch]$WithProbe) {
    # One look at the port: what the server's own log says, then (if quiet) what the engine admits.
    $v = @{ decision = 'IDLE'; reasons = @(); pid = 0; open = 0; age_s = -1; probe = -1; lock = 'none'; lock_owner = '' }
    $v.pid = Listener-Pid
    $lock = Get-Lock
    $v.lock = $lock.state
    $v.lock_owner = $lock.owner
    if ($lock.state -eq 'live') {
        $v.decision = 'LOCKED'
        $v.reasons += ('window in progress: owner=' + $lock.owner + ' pid=' + $lock.pid +
                       ' since=' + $lock.since + ' announced_end=' + $lock.until)
        return $v
    }
    if ($v.pid -eq 0) {
        $v.reasons += 'no listener on :' + $Port + ' (serve is already down; a window may take the port)'
        return $v
    }
    $t = Get-Traffic 6000000
    if ($t.known) {
        $v.open = $t.open
        $v.age_s = $t.age_s
        if ($t.open -gt 0) {
            $v.decision = 'BUSY'
            $v.reasons += ([string]$t.open + ' request(s) in flight on :' + $Port + ' (ids ' + ($t.started_open -join ',') + ')')
            return $v
        }
        if ($t.age_s -ge 0 -and $t.age_s -lt $QuietSec) {
            $v.decision = 'BUSY'
            $v.reasons += ('the last request on :' + $Port + ' finished ' + $t.age_s + ' s ago (< ' + $QuietSec + ' s quiet)')
            return $v
        }
    } else {
        $v.reasons += 'no request records read from ' + $RequestLog + ', relying on the probe'
    }
    if ($WithProbe -and $ProbeSec -gt 0) {
        $p = Invoke-AdmissionProbe $ProbeSec
        $v.probe = $p.seconds
        if (-not $p.ok) {
            $v.decision = 'BUSY'
            $v.reasons += ('admission probe failed after ' + $p.seconds + ' s: ' + $p.why)
        }
    }
    return $v
}

function Show-Verdict($v) {
    Say ('pid=' + $v.pid + ' lock=' + $v.lock + ' lock_owner=' + $v.lock_owner +
         ' open_requests=' + $v.open + ' last_request_age_s=' + $v.age_s + ' probe_s=' + $v.probe)
    foreach ($r in $v.reasons) { Say ('reason: ' + $r) }
    if ($v.decision -eq 'IDLE') { Verdict 'IDLE'; return 0 }
    if ($v.decision -eq 'BUSY') { Verdict 'BUSY'; return 10 }
    if ($v.decision -eq 'LOCKED') { Verdict 'LOCKED'; return 11 }
    Verdict 'IDLE'
    return 0
}

function Start-Shield($ownerName) {
    # The marker is written BEFORE the port goes down, so a session that asks "is it dead or paused?"
    # gets an answer during the grace phase too. The shield window waits for the port, then answers
    # 503 and shows the countdown; explorer keeps it in its own visible window.
    if (-not $Reason) { $Reason = 'fonstret ' + $ownerName }
    $enc = New-Object System.Text.UTF8Encoding($false)
    $endsNow = (Get-Date).AddMinutes($ExpectedMinutes)
    $obj = @{ shield = 'image-spike/window_shield.ps1'; started_by_gate = $ownerName; reason = $Reason;
             port = $Port; started_local = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss');
             ends_local = $endsNow.ToString('yyyy-MM-dd HH:mm:ss');
             answers = 'HTTP 503 with Retry-After (the shield writes its own pid here when it takes over)';
             message = 'infernix cert-window in progress (' + $Reason + ') - :8888 is down on purpose, not a crash';
             lock = $LockFile } | ConvertTo-Json -Compress
    try { [System.IO.File]::WriteAllText($MarkerFile, $obj, $enc) } catch { Say ('marker write failed: ' + $_.Exception.Message) }
    if ($NoShield) { Say ('marker written, no 503 stub because the window binds the port itself: ' + $MarkerFile); return }
    if (-not (Test-Path -LiteralPath $Shield)) { Say ('SHIELD_MISSING ' + $Shield + ' (the window still runs, but without the 503 stub)'); return }
    $lines = @('@echo off',
               ('title INFERNIX 8888 PAUSAS - ' + $ownerName),
               ('powershell -NoProfile -ExecutionPolicy Bypass -File "' + $Shield + '" -Port ' + $Port +
                ' -LockFile "' + $LockFile + '" -MarkerFile "' + $MarkerFile + '" -Reason "' + $Reason +
                '" -MaxMinutes ' + $ShieldMaxMinutes))
    try { [System.IO.File]::WriteAllLines($ShieldBat, $lines, [System.Text.Encoding]::ASCII) } catch { Say ('shield bat write failed: ' + $_.Exception.Message); return }
    Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $ShieldBat + '"') | Out-Null
    Say ('shield window launched (visible): ' + $ShieldBat)
}

function Stop-Shield {
    # Before serve comes back: the shield must let go of the port, or the relaunch cannot bind it.
    $pidToKill = 0
    if (Test-Path -LiteralPath $MarkerFile) {
        try { $j = Get-Content -LiteralPath $MarkerFile -Raw | ConvertFrom-Json; $pidToKill = [int]$j.shield_pid } catch {}
    }
    if ($pidToKill -gt 0 -and (Get-Process -Id $pidToKill -ErrorAction SilentlyContinue)) {
        taskkill /F /T /PID $pidToKill 2>&1 | Out-Null
        Say ('shield pid=' + $pidToKill + ' stopped')
    } else {
        Say 'no shield process recorded as running'
    }
    for ($i = 0; $i -lt 20; $i++) {
        if ((Listener-Pid) -eq 0) { break }
        Start-Sleep -Milliseconds 500
    }
    Say ('port free after shield: listener=' + (Listener-Pid))
    Remove-Item -LiteralPath $MarkerFile -Force -ErrorAction SilentlyContinue
}

if ($Action -eq 'EndWindow') {
    Stop-Shield
    if (Test-Path -LiteralPath $LockFile) {
        Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
        Say ('released ' + $LockFile)
    }
    Verdict 'WINDOW_ENDED'
    exit 0
}

if ($Action -eq 'Release') {
    if (Test-Path -LiteralPath $LockFile) {
        Remove-Item -LiteralPath $LockFile -Force -ErrorAction SilentlyContinue
        Say ('released ' + $LockFile)
    } else { Say 'nothing to release' }
    Verdict 'RELEASED'
    exit 0
}

if ($Action -eq 'Status') {
    # The cheap read-only look, for anyone wondering whether :8888 is coming back or is dead.
    $lock = Get-Lock
    Say ('port=' + $Port + ' listener_pid=' + (Listener-Pid) + ' lock=' + $lock.state +
         ' owner=' + $lock.owner + ' since=' + $lock.since + ' announced_end=' + $lock.until)
    $t = Get-Traffic 6000000
    if ($t.known) { Say ('instance=' + $t.instance + ' open_requests=' + $t.open + ' last_request_age_s=' + $t.age_s) }
    if ($lock.state -eq 'live') { Verdict 'WINDOW_RUNNING'; exit 12 }
    if ((Listener-Pid) -eq 0) { Verdict 'NO_LISTENER'; exit 12 }
    Verdict 'NO_WINDOW'
    exit 0
}

# Check and Acquire both look at the port; only Acquire takes the lock, and only when it is free.
if ($Action -eq 'Check') {
    $v = Evaluate -WithProbe
    exit (Show-Verdict $v)
}

# Acquire: wait out the turns that are running, then take the lock. A probe that times out wrote a
# request_start of its own, so after a failed probe the wait is at least QuietSec long -- otherwise
# the gate's own probe would keep the port forever "busy".
$deadline = (Get-Date).AddSeconds($MaxWaitSec)
for (;;) {
    $v = Evaluate -WithProbe
    if ($v.decision -eq 'IDLE') { break }
    if ($Force) { Say 'FORCE: taking the port anyway, the sessions on it will lose their turns'; break }
    if ((Get-Date) -ge $deadline) {
        Say 'wait budget spent'
        exit (Show-Verdict $v)
    }
    $nap = 15
    if ($v.reasons -join ' ' -match 'probe failed') { $nap = $QuietSec }   # let our own probe age out
    if ($v.decision -eq 'LOCKED') { $nap = 30 }
    if ($deadline - (Get-Date) -lt [TimeSpan]::FromSeconds($nap)) { $nap = [int][Math]::Max(5, (( $deadline - (Get-Date)).TotalSeconds)) }
    Say ('waiting ' + $nap + ' s for :' + $Port + ' (' + ($v.reasons -join '; ') + ')')
    Start-Sleep -Seconds $nap
}

if (-not $Owner) { $Owner = 'pid ' + $PID }
if (-not (Set-Lock $Owner)) {
    Say 'another window took the lock first'
    Verdict 'LOCKED'
    exit 11
}
Say ('ACQUIRED owner=' + $Owner + ' pid=' + $PID + ' expected_minutes=' + $ExpectedMinutes +
     ' announced_end=' + (Get-Date).AddMinutes($ExpectedMinutes).ToString($fmtIso) + ' lock=' + $LockFile)
if ($Action -eq 'StartWindow') { Start-Shield $Owner }
Verdict 'ACQUIRED'
exit 0
