param(
    [string]$Size = '768x768',
    [string]$Out  = 'C:\Users\Micke\Desktop\Infernix FlashNext\maine-coon-katt.png'
)
$ErrorActionPreference = 'Continue'
$base = 'http://127.0.0.1:8888'

$c = Get-NetTCPConnection -LocalPort 8888 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$kv = '?'; $img = 'off'; $exe = '?'
if ($c) {
    $cl = (Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess)).CommandLine
    if ($cl -match '--kv-dtype\s+nvfp4') { $kv = 'nvfp4' } elseif ($cl -match '--kv-dtype\s+fp8') { $kv = 'fp8' }
    if ($cl -match '--image-artifact') { $img = 'on' }
    if ($cl -match 'build\\apps') { $exe = 'build' } else { $exe = 'desktop' }
}
Write-Output ('SERVER exe=' + $exe + ' kv=' + $kv + ' images=' + $img)

$id = ''
try {
    $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { $id = $Matches[1] }
} catch {}
Write-Output ('MODEL=' + $id)

$prompt = 'A fluffy white Maine Coon cat with a single small dark spot on its forehead, sitting calmly on a grey stone rock outdoors, soft natural daylight, photorealistic, highly detailed fur'
$body = @{ model = $id; prompt = $prompt; size = $Size; n = 1; response_format = 'b64_json' } | ConvertTo-Json -Compress
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 600
    $obj = $resp.Content | ConvertFrom-Json
    $b64 = $obj.data[0].b64_json
    $bytes = [System.Convert]::FromBase64String($b64)
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
