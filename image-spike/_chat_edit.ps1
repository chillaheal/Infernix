# The built-in path, no scripts around it: one chat request, the photo attached, an instruction.
# Nothing here names an endpoint beyond /v1/chat/completions and nothing sends base64 as an image
# argument -- the server's own generate_image tool is called inside the request.
param(
    [string]$BaseUrl = 'http://127.0.0.1:8888',
    [string]$ChatModel = 'Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix',
    [string]$Pic     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\owl_edit\kingfisher.png',
    [string]$Out     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\faceswap\chat_edit.png',
    [string]$Text    = 'Change this bird''s face into an owl face: large round forward-facing orange eyes, a short hooked beak, a feathered facial disc. Keep the body, the branch and the background.'
)
$ErrorActionPreference = 'Continue'

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

if (-not (Test-Path -LiteralPath $Pic)) { Log ('MISSING_PIC ' + $Pic); exit 2 }
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($Pic))
Log ('ATTACH ' + $Pic + ' b64 ' + $b64.Length)

$body = [ordered]@{
    model     = $ChatModel
    messages  = @(
        @{
            role    = 'user'
            content = @(
                @{ type = 'text'; text = $Text },
                @{ type = 'image_url'; image_url = @{ url = ('data:image/png;base64,' + $b64) } }
            )
        }
    )
    max_tokens = 800
}
$json = $body | ConvertTo-Json -Compress -Depth 12
Log ('POST ' + $BaseUrl + '/v1/chat/completions json=' + $json.Length)
$t0 = Get-Date
try {
    $resp = Invoke-WebRequest -Uri ($BaseUrl + '/v1/chat/completions') -Method Post `
            -ContentType 'application/json' -Body $json -UseBasicParsing -TimeoutSec 1800
} catch {
    $detail = $_.Exception.Message
    try {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $detail = $reader.ReadToEnd()
    } catch { }
    Log ('HTTP_FAIL ' + $detail)
    Log ('CHAT_EDIT_RESULT fail=1')
    exit 1
}
Log ('status=' + [int]$resp.StatusCode + ' in ' + [math]::Round(((Get-Date) - $t0).TotalSeconds, 1) + 's')
$obj = $resp.Content | ConvertFrom-Json
$msg = ''
if ($obj.choices -and $obj.choices[0].message) { $msg = [string]$obj.choices[0].message.content }
Log ('ANSWER ' + $msg)
$urls = @([regex]::Matches($resp.Content, 'https?://[^"]*/v1/images/img-[0-9A-Za-z-]+') |
        ForEach-Object { $_.Value } | Select-Object -Unique)
Log ('URLS ' + ($urls -join ' '))
if (-not $urls) { Log 'NO_URL_IN_REPLY'; Log ('CHAT_EDIT_RESULT fail=1'); exit 1 }
$first = $urls[0]
if ($first -match '^https?://[^/]+(/v1/images/.*)$') { $first = $BaseUrl + $Matches[1] }
if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Force }
try {
    Invoke-WebRequest -Uri $first -OutFile $Out -UseBasicParsing -TimeoutSec 120 | Out-Null
    Log ('SAVED ' + $Out + ' ' + (Get-Item -LiteralPath $Out).Length + ' bytes')
} catch {
    Log ('DOWNLOAD_FAIL ' + $_.Exception.Message)
    Log ('CHAT_EDIT_RESULT fail=1'); exit 1
}
Log ('CHAT_EDIT_RESULT fail=0 url=' + $first)
exit 0
