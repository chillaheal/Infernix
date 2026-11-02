param(
    [string]$Prompt = 'A kingfisher bird, but larger, the stocky build and size of a parrot, with intensely vibrant saturated colors: electric blue, vivid orange-red, emerald and teal iridescent plumage, a long strong parrot-like beak, perched on a branch in a lush green tropical setting, photorealistic, highly detailed, sharp focus, vivid natural light',
    [string]$Size = '768x768',
    [string]$Out  = 'C:\Users\Micke\Desktop\kingfisher_parrot.png'
)
$ErrorActionPreference = 'Continue'
$base = 'http://127.0.0.1:8888'

$id = ''
try {
    $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { $id = $Matches[1] }
} catch {}
Write-Output ('MODEL=' + $id)

$body = @{ model = $id; prompt = $Prompt; size = $Size; n = 1; response_format = 'b64_json' } | ConvertTo-Json -Compress
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 600
    $obj = $resp.Content | ConvertFrom-Json
    $bytes = [System.Convert]::FromBase64String($obj.data[0].b64_json)
    [System.IO.File]::WriteAllBytes($Out, $bytes)
    $sw.Stop()
    Write-Output ('IMG_SIZE=' + $Size + ' HTTP=200 SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' PNG_BYTES=' + $bytes.Length + ' FILE=' + $Out)
} catch {
    $sw.Stop(); $e = $_.Exception; $st = -1; $bd = ''
    if ($e.Response) {
        try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {}
        try { $st = [int]$e.Response.StatusCode } catch {}
    }
    Write-Output ('IMG_SIZE=' + $Size + ' HTTP=' + $st + ' SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' BODY=' + $bd)
}
