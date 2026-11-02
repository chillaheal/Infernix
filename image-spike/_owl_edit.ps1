# Instruction-guided edit against the live Infernix serve (same route the M9 behavior gate uses).
param(
    [string]$BaseUrl = 'http://127.0.0.1:8888',
    [string]$Model   = 'Qwen-Image-2.1-FP8-v4',
    [string]$Ref     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\owl_edit\kingfisher.png',
    [string]$Out     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\owl_edit\owl_face_A.png',
    [string]$Prompt  = 'Remove the long straight dagger-like beak completely, it must be gone. The bird has exactly one beak: a short, thick, dark gray owl beak at the center of the face, curving downward to a hook. The head is an owl face: two large round forward-facing orange eyes set in a rounded gray feathered facial disc. Keep the blue speckled crown, the orange chest and belly, the blue wings, the feet, the branch and the blurred green background unchanged.',
    [string]$Size    = '1024x1024',
    [int]$Steps      = 20,
    [int]$Seed       = 7
)
$ErrorActionPreference = 'Continue'
Add-Type -AssemblyName System.Drawing

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
    Log ('OWL_EDIT_RESULT fail=1')
    exit 1
}
$secs = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Log ('status=' + [int]$resp.StatusCode + ' in ' + $secs + 's, ' + $resp.RawContentLength + ' bytes')
$obj = $resp.Content | ConvertFrom-Json
if (-not $obj.data -or -not $obj.data[0].b64_json) {
    Log ('NO_IMAGE_IN_RESPONSE ' + $resp.Content.Substring(0, [Math]::Min(400, $resp.Content.Length)))
    Log ('OWL_EDIT_RESULT fail=1')
    exit 1
}
[IO.File]::WriteAllBytes($Out, [Convert]::FromBase64String($obj.data[0].b64_json))
$bmp = New-Object System.Drawing.Bitmap $Out
Log ('SAVED ' + $Out + ' ' + (Get-Item -LiteralPath $Out).Length + ' bytes ' + $bmp.Width + 'x' + $bmp.Height)
$bmp.Dispose()
Log ('OWL_EDIT_RESULT fail=0 secs=' + $secs)
exit 0
