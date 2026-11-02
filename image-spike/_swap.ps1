# Harmonize a coarse head-swap composite through the live edit route (one reference image only).
param(
    [string]$BaseUrl = 'http://127.0.0.1:8888',
    [string]$Model   = 'Qwen-Image-2.1-FP8-v4',
    [string]$Ref     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\faceswap\composite.png',
    [string]$Out     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\faceswap\swap_A.png',
    [string]$Prompt  = 'Photorealistic portrait photograph. Keep this woman''s face exactly as it is: the same facial features, the same long blonde hair, the same blue eyes and the same smile. Remove every trace of the cut and paste: blend the hairline and the neck seamlessly into the shoulders, match the skin tone of the head to the tanned muscular body, add the same sunlight and shadow direction to the face. Keep the pose, the muscles, the arm tattoo, the white hand wraps, the black shorts and the blurred garden background exactly unchanged.',
    [string]$Size    = '512x576',
    [int]$Steps      = 20,
    [int]$Seed       = 7,
    [switch]$Tan
)
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Drawing

# The A prompt asks for BOTH "keep the face exactly as it is" AND "match the skin tone to the body"
# -- a contradiction. B asks only for the tan, so a tanned face in B means A's pale face was the
# prompt, not the checkpoint.
if ($Tan) {
    $Prompt = 'Photorealistic portrait photograph. The head of this woman sits on a tanned muscular bodybuilder body. Give the head, ears and neck the same deep tan as the shoulders and chest so the whole figure has one even skin tone, and match the sunlight and shadow direction on the face to the body. Blend the hairline and the neck seamlessly. Keep her facial features, blue eyes, smile and long blonde hair the same, and keep the pose, the muscles, the arm tattoo, the white hand wraps, the black shorts and the blurred garden background unchanged.'
}

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }

if (-not (Test-Path -LiteralPath $Ref)) { Log ('MISSING_REF ' + $Ref); exit 2 }
$refB64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($Ref))
Log ('REF ' + $Ref + ' ' + (Get-Item -LiteralPath $Ref).Length + ' bytes, b64 ' + $refB64.Length)
Log ('PROMPT ' + $Prompt)

$body = [ordered]@{
    model           = $Model
    prompt          = $Prompt
    n               = 1
    size            = $Size
    steps           = $Steps
    seed            = $Seed
    response_format = 'b64_json'
    image           = $refB64
}
$json = $body | ConvertTo-Json -Compress -Depth 5
if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Force }
Log ('EDIT POST ' + $BaseUrl + '/v1/images/generations size=' + $Size + ' steps=' + $Steps + ' json=' + $json.Length)
$t0 = Get-Date
try {
    $resp = Invoke-WebRequest -Uri ($BaseUrl + '/v1/images/generations') -Method Post `
            -ContentType 'application/json' -Body $json -UseBasicParsing -TimeoutSec 1800
} catch {
    $code = 0; $detail = $_.Exception.Message
    try {
        $code = [int]$_.Exception.Response.StatusCode
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $detail = $reader.ReadToEnd()
    } catch { }
    Log ('HTTP_FAIL status=' + $code + ' body=' + $detail)
    Log ('SWAP_RESULT fail=1')
    exit 1
}
$secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Log ('status=' + [int]$resp.StatusCode + ' in ' + $secs + 's, ' + $resp.RawContentLength + ' bytes')
$obj = $resp.Content | ConvertFrom-Json
if (-not $obj.data -or -not $obj.data[0].b64_json) {
    Log ('NO_IMAGE_IN_RESPONSE ' + $resp.Content.Substring(0, [Math]::Min(400, $resp.Content.Length)))
    Log ('SWAP_RESULT fail=1')
    exit 1
}
[IO.File]::WriteAllBytes($Out, [Convert]::FromBase64String($obj.data[0].b64_json))
$bmp = New-Object System.Drawing.Bitmap $Out
Log ('SAVED ' + $Out + ' ' + (Get-Item -LiteralPath $Out).Length + ' bytes ' + $bmp.Width + 'x' + $bmp.Height)
$bmp.Dispose()
Log ('SWAP_RESULT fail=0 secs=' + $secs)
exit 0
