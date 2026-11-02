$ErrorActionPreference = 'Continue'

# #83 live check with NO blackout: one plain chat request, no tools field, on the running server.
# The server runs its own hidden generate_image tool inside the request, so the answer must arrive as a
# single assistant message that carries a fetchable artifact URL and shows no tool call to the client.

$base = 'http://127.0.0.1:8888'
$out = 'C:\Users\Micke\Desktop\chat-auto-live.png'

function Log($m) { Write-Output ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

$chatId = ''
$models = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 15).Content | ConvertFrom-Json
foreach ($m in $models.data) { if ([string]$m.id -match 'Flash-Next') { $chatId = [string]$m.id } }
Log ('CHAT_ID ' + $chatId)
if (-not $chatId) { Log 'NO_CHAT_ID'; exit 2 }

$body = @{ model = $chatId; max_tokens = 2048; messages = @(
    @{ role = 'user'; content = 'Draw a small picture of a red fox curled up on pale snow and give me the link to it.' }
) } | ConvertTo-Json -Depth 5 -Compress

$ok = $false
for ($try = 1; $try -le 3; $try++) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $r = Invoke-WebRequest -Uri ($base + '/v1/chat/completions') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 900
        $o = $r.Content | ConvertFrom-Json
        $answer = [string]$o.choices[0].message.content
        $finish = [string]$o.choices[0].finish_reason
        $sw.Stop()
        $url = ''
        if ($answer -match '(http://[^ ""''<>)]+/v1/images/[A-Za-z0-9._-]+)') { $url = $matches[1] }
        $sawTool = ($null -ne $o.choices[0].message.tool_calls)
        Log ('TRY=' + $try + ' HTTP=' + [int]$r.StatusCode + ' SECONDS=' + [int]$sw.Elapsed.TotalSeconds +
             ' finish=' + $finish + ' tool_calls=' + [bool]$sawTool +
             ' tokens=' + $o.usage.prompt_tokens + '/' + $o.usage.completion_tokens)
        Log ('ANSWER| ' + $answer.Substring(0, [Math]::Min(300, $answer.Length)))
        if ($url) {
            $g = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 60
            $b = $g.Content
            $png = ($b.Length -gt 8 -and $b[0] -eq 137 -and $b[1] -eq 80 -and $b[2] -eq 78 -and $b[3] -eq 71)
            [System.IO.File]::WriteAllBytes($out, $b)
            $ok = ($png -and $b.Length -gt 10000 -and -not $sawTool -and ($finish -ne 'tool_calls'))
            Log ('FETCH HTTP=' + [int]$g.StatusCode + ' BYTES=' + $b.Length + ' PNG=' + $png +
                 ' CACHE=' + $g.Headers['Cache-Control'] + ' -> ' + $out)
            break
        }
        Log ('NO_URL finish=' + $finish + ' completion_tokens=' + $o.usage.completion_tokens)
    } catch {
        $sw.Stop()
        $st = -1
        try { $st = [int]$_.Exception.Response.StatusCode } catch {}
        Log ('TRY=' + $try + ' HTTP=' + $st + ' err=' + $_.Exception.Message)
        Start-Sleep -Seconds 20
    }
}

if ($ok) { Log 'RESULT: CHATAUTO PASS - a plain chat request drew a picture the client can fetch'; exit 0 }
Log 'RESULT: CHATAUTO FAIL - no fetchable picture in the answer'
exit 1
