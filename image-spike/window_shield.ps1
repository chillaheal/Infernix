param(
    [int]$Port = 8888,
    [string]$LockFile   = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-window.lock',
    [string]$MarkerFile = 'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-window.json',
    [string]$Reason = 'cert window',
    [int]$MaxMinutes = 90,        # hard stop: the stub must never hold the port longer than this
    [int]$RetryAfterSec = 30,     # the answer when no window end is known at all
    [int]$RetryAfterMin = 15,     # never ask a client to wait less than this (a burst buys nothing)
    [int]$RetryAfterMax = 300,    # nor more than this, however long the window is (see Get-RetryAfterSeconds)
    [switch]$NoWait,              # bind at once (a port the serve does not hold) - used to test
    [switch]$NoWindow             # no banner refresh (when started from another window)
)

# The window shield: one visible window that does two jobs while a blackout window owns the GPU.
#
#  1. A 503 stub on :8888. When infernix-serve is stopped, every other Claude Code session got
#     ECONNREFUSED, which reads as "the server crashed". This answers on the same port with
#     503 + Retry-After + a message saying the port is down ON PURPOSE and about how long is left,
#     so a session (and the person watching it) can tell a pause from a failure.
#  2. The notice itself: this window IS the notice - owner, reason, countdown, and the same text the
#     stub answers with. It also writes the public marker file the countdown comes from.
#
# It binds only once :8888 is actually free (a live serve keeps the port), and it quits when the
# window's lock is released or past its announced end, or at MaxMinutes. A blackout script kills this
# process (its pid is written into the marker) BEFORE it relaunches serve, so the stub can never
# keep the port after the window.
#
# A raw TcpListener, not HttpListener: HttpListener goes through http.sys and wants a URL ACL
# reservation for the port, which a plain user process cannot count on.

$ErrorActionPreference = 'Continue'
# The port is named from $Port, never as a literal: this stub is also run on a spare port to test it, and
# a notice that says :8888 while answering on another port is the kind of text that gets quoted at someone.
$reasonOnly = 'infernix cert-window in progress (' + $Reason + ') - :' + [string]$Port + ' is down on purpose, not a crash'

function Say($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

function Get-WindowInfo {
    $i = @{ live = $false; owner = ''; ends = $null; minutes = 0 }
    if (-not (Test-Path -LiteralPath $LockFile)) { return $i }
    try { $j = Get-Content -LiteralPath $LockFile -Raw | ConvertFrom-Json } catch { return $i }
    $i.owner = [string]$j.owner
    $i.minutes = [int]$j.expected_minutes
    $i.ends = ([DateTimeOffset]::FromUnixTimeMilliseconds([long]$j.started_unix_ms).LocalDateTime).AddMinutes($i.minutes)
    $i.live = (Get-Date) -lt $i.ends
    return $i
}

function Set-Marker($secondsLeft, $ends) {
    # The marker is the public notice anyone can read: who, why, until when, and what the stub says.
    $endsText = ''
    if ($ends) { $endsText = $ends.ToString('yyyy-MM-dd HH:mm:ss') }
    $obj = @{ shield = 'image-spike/window_shield.ps1'; shield_pid = $PID; port = $Port; reason = $Reason;
             started_local = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'); ends_local = $endsText;
             seconds_left = $secondsLeft; answers = 'HTTP 503 with Retry-After ' + [string](Get-RetryAfterSeconds $secondsLeft) + ' s';
             message = $reasonOnly } | ConvertTo-Json -Compress
    try { [System.IO.File]::WriteAllText($MarkerFile, $obj, (New-Object System.Text.UTF8Encoding($false))) } catch { Say ('marker write failed: ' + $_.Exception.Message) }
}

function Listener-OnPort {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    return 0
}

function Get-RemainingSeconds($ends) {
    if (-not $ends) { return [int]($MaxMinutes * 60) }
    $s = [int](($ends - (Get-Date)).TotalSeconds)
    if ($s -lt 0) { return 0 }
    return $s
}

function Get-RetryAfterSeconds($secondsLeft) {
    # Measured in the installed CLI (versions/2.1.233, the bytes behind ~/.local/bin/claude.exe): its
    # retryRequest honours Retry-After EXACTLY and computes its own backoff only when the header is
    # absent, and 503 is retryable (status>=500). The attempt count at that call site is unknown, so a
    # client has a finite number of tries and each one is spent after the delay we name. A constant 30 s
    # therefore fires every attempt long before a 20 min window ends -- the budget runs out while the
    # server is still down. Asking for the time actually left spends the attempts where they can land on
    # a live serve instead, clamped so a short window is not padded and a long one is not a blackout.
    if ($null -eq $secondsLeft) { return $RetryAfterSec }
    $s = [int]$secondsLeft
    if ($s -lt $RetryAfterMin) { return $RetryAfterMin }
    if ($s -gt $RetryAfterMax) { return $RetryAfterMax }
    return $s
}

function Get-StubResponse($secondsLeft) {
    $mins = [Math]::Max(0, [Math]::Ceiling($secondsLeft / 60))
    $msg = $reasonOnly + ', about ' + [string]$mins + ' min left'
    $body = '{"error":{"type":"service_unavailable","code":"cert_window_in_progress","message":"' + $msg + '"}}'
    $enc = New-Object System.Text.UTF8Encoding($false)
    $bodyBytes = $enc.GetBytes($body)
    $head = 'HTTP/1.1 503 Service Unavailable' + "`r`n" +
            'Content-Type: application/json' + "`r`n" +
            'Retry-After: ' + [string](Get-RetryAfterSeconds $secondsLeft) + "`r`n" +
            'Connection: close' + "`r`n" +
            'Content-Length: ' + [string]$bodyBytes.Length + "`r`n`r`n"
    return @{ head = $enc.GetBytes($head); body = $bodyBytes; text = $body }
}

function Serve-One($client, $secondsLeft, $counter) {
    # Read the request head and its body (so a client is not cut off mid-write), then answer 503.
    try {
        $stream = $client.GetStream()
        $stream.ReadTimeout = 5000
        $stream.WriteTimeout = 5000
        $buf = New-Object byte[] 8192
        $text = ''
        for ($i = 0; $i -lt 64; $i++) {
            try { $n = $stream.Read($buf, 0, $buf.Length) } catch { break }
            if ($n -le 0) { break }
            $text += (New-Object System.Text.UTF8Encoding($false)).GetString($buf, 0, $n)
            if ($text.Contains("`r`n`r`n")) { break }
        }
        $cl = 0
        if ($text -match '(?im)^content-length:\s*(\d+)') { $cl = [int]$Matches[1] }
        $headEnd = $text.IndexOf("`r`n`r`n")
        $bodySoFar = 0
        if ($headEnd -ge 0) { $bodySoFar = $text.Length - ($headEnd + 4) }
        $left = $cl - $bodySoFar
        $drained = 0
        while ($left -gt 0 -and $drained -lt 4194304) {
            try { $n = $stream.Read($buf, 0, [Math]::Min($buf.Length, $left)) } catch { break }
            if ($n -le 0) { break }
            $left -= $n; $drained += $n
        }
        $resp = Get-StubResponse $secondsLeft
        $stream.Write($resp.head, 0, $resp.head.Length)
        $stream.Write($resp.body, 0, $resp.body.Length)
        $stream.Flush()
        $counter.n += 1
        $line = ''
        if ($text -match '^(GET|POST|PUT|HEAD|OPTIONS|DELETE) +([^ ]+)') { $line = $Matches[1] + ' ' + $Matches[2] }
        Say ('answered 503 n=' + $counter.n + ' ' + $line)
    } catch { Say ('stub write failed: ' + $_.Exception.Message) }
    finally { try { $client.Close() } catch {} }
}

function Show-Banner($info, $secondsLeft, $answeredTotal) {
    $bar = '=' * 70
    $endsText = 'okand'
    if ($info.ends) { $endsText = $info.ends.ToString('HH:mm:ss') }
    Write-Host ''
    Write-Host $bar
    Write-Host ('  INFERNIX :' + [string]$Port + ' PAUSAS - CERT-/BENCH-FONSTER PAGAR')
    Write-Host ('  agare:    ' + $info.owner)
    Write-Host ('  orsak:    ' + $Reason)
    Write-Host ('  kvar:     ~' + [Math]::Max(0, [int]($secondsLeft / 60)) + ' min (' + $secondsLeft + ' s), upp till ' + $endsText)
    Write-Host ('  servern:  avsiktligt nere. Porten SVARAR 503 (Retry-After ' + [string](Get-RetryAfterSeconds $secondsLeft) + ' s, klamd ' + $RetryAfterMin + '-' + $RetryAfterMax + ' s) i stallet for ECONNREFUSED.')
    Write-Host ('  svarat:   ' + $answeredTotal + ' forfragningar sedan fonstret borjade')
    Write-Host ('  texten andra sessioner far: ' + $reasonOnly)
    Write-Host ('  markor:   ' + $MarkerFile)
    Write-Host $bar
}

$startedAt = Get-Date
$hardEnd = $startedAt.AddMinutes($MaxMinutes)
$counter = @{ n = 0 }

if (-not $NoWait) {
    Say ('waiting for :' + $Port + ' to be released by the live serve')
    for (;;) {
        if ((Listener-OnPort) -eq 0) { break }
        if ((Get-Date) -gt $hardEnd) { Say 'hard stop before the port was free; nothing was bound'; exit 3 }
        $iw = Get-WindowInfo
        if (-not $iw.live) { Say 'the window lock is gone or past its announced end; leaving the port alone'; exit 0 }
        Start-Sleep -Seconds 1
    }
}

$listener = $null
try {
    $listener = New-Object 'System.Net.Sockets.TcpListener' @([System.Net.IPAddress]::Loopback, $Port)
    $listener.Start()
} catch {
    Say ('COULD NOT BIND :' + $Port + ' - ' + $_.Exception.Message + ' (something else owns the port; exiting)')
    exit 4
}
Say ('stub listening on 127.0.0.1:' + $Port + ' pid=' + $PID + ' reason=' + $Reason)

$lastBanner = [DateTime]::MinValue
try {
    for (;;) {
        $info = Get-WindowInfo
        $secondsLeft = Get-RemainingSeconds $info.ends
        if (-not $NoWait) {
            if (-not (Test-Path -LiteralPath $LockFile)) { Say 'window released the lock; closing the stub'; break }
            if ($secondsLeft -le 0) { Say 'the announced window end passed; closing the stub'; break }
            if ((Get-Date) -gt $hardEnd) { Say 'MaxMinutes reached; closing the stub'; break }
        }
        if ($listener.Pending()) {
            $client = $listener.AcceptTcpClient()
            Serve-One $client $secondsLeft $counter
        } elseif (((Get-Date) - $lastBanner).TotalSeconds -ge 10) {
            $lastBanner = Get-Date
            if (-not $NoWindow) { Clear-Host }
            Set-Marker $secondsLeft $info.ends
            Show-Banner $info $secondsLeft $counter.n
        } else {
            Start-Sleep -Milliseconds 150
        }
    }
} finally {
    try { $listener.Stop() } catch {}
    try { Remove-Item -LiteralPath $MarkerFile -Force -ErrorAction SilentlyContinue } catch {}
}
Say ('shield done, answered=' + $counter.n + ' alive=' + [int](((Get-Date) - $startedAt).TotalSeconds) + 's')
exit 0
