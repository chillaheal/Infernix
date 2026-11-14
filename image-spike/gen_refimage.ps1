param(
    [string]$RefImage = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\oracle_ref_encode\source.png',
    [string]$Size     = '768x768',
    [string]$Out      = 'C:\Users\Micke\Desktop\Infernix FlashNext\refimage-cert.png'
)
$ErrorActionPreference = 'Continue'
$base = 'http://127.0.0.1:8888'

$id = ''
try {
    $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { $id = $Matches[1] }
} catch {}
Write-Output ('MODEL=' + $id)

$refBytes  = [System.IO.File]::ReadAllBytes($RefImage)
$refB64    = [System.Convert]::ToBase64String($refBytes)
$prompt    = 'Turn the photo into a watercolor painting with soft edges and a warm sunset palette'
$body      = @{ model = $id; prompt = $prompt; size = $Size; n = 1; response_format = 'b64_json';
                image = $refB64 } | ConvertTo-Json -Compress
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 1800
    $obj  = $resp.Content | ConvertFrom-Json
    $b64  = $obj.data[0].b64_json
    $bytes = [System.Convert]::FromBase64String($b64)
    [System.IO.File]::WriteAllBytes($Out, $bytes)
    $sw.Stop()
    Write-Output ('REFIMG in_bytes=' + $refBytes.Length + ' HTTP=200 SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' PNG_BYTES=' + $bytes.Length + ' FILE=' + $Out)
} catch {
    $sw.Stop(); $e = $_.Exception; $st = -1; $bd = ''
    if ($e.Response) {
        try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {}
        try { $st = [int]$e.Response.StatusCode } catch {}
    }
    Write-Output ('REFIMG in_bytes=' + $refBytes.Length + ' HTTP=' + $st + ' SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' BODY=' + $bd)
}
