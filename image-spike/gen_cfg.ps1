param(
    [string]$Size  = '512x512',
    [int]$Steps    = 8,
    [double]$Scale = 3.5,
    [string]$Dir   = 'C:\Users\Micke\Desktop\Infernix FlashNext'
)
$ErrorActionPreference = 'Continue'
$base = 'http://127.0.0.1:8888'

$id = ''
try {
    $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).Content
    if ($m -match '"id"\s*:\s*"([^"]+)"') { $id = $Matches[1] }
} catch {}
Write-Output ('MODEL=' + $id)

$prompt = 'a red apple on a white table, studio light'
$seed   = 12345

function Post-Image($name, $body, $out) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec 1800
        $obj  = $resp.Content | ConvertFrom-Json
        $bytes = [System.Convert]::FromBase64String($obj.data[0].b64_json)
        [System.IO.File]::WriteAllBytes($out, $bytes)
        $sw.Stop()
        $h = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash
        Write-Output ($name + ' HTTP=200 SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' PNG_BYTES=' + $bytes.Length + ' SHA=' + $h + ' FILE=' + $out)
        return $h
    } catch {
        $sw.Stop(); $e = $_.Exception; $st = -1; $bd = ''
        if ($e.Response) {
            try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {}
            try { $st = [int]$e.Response.StatusCode } catch {}
        }
        Write-Output ($name + ' HTTP=' + $st + ' SECONDS=' + [int]$sw.Elapsed.TotalSeconds + ' BODY=' + $bd)
        return ''
    }
}

# A: text-to-image at true_cfg_scale 1 (the single-forward path, the plain regression).
$bodyA = @{ model = $id; prompt = $prompt; size = $Size; n = 1; response_format = 'b64_json';
            seed = $seed; steps = $Steps } | ConvertTo-Json -Compress
$hashA = (Post-Image 'CFGPLAIN' $bodyA ($Dir + '\cfg-plain.png'))

# B: identical seed/prompt, steered by a negative prompt at scale > 1 (second forward per step).
$bodyB = @{ model = $id; prompt = $prompt; size = $Size; n = 1; response_format = 'b64_json';
            seed = $seed; steps = $Steps; negative_prompt = 'blurry, low quality, dark';
            true_cfg_scale = $Scale } | ConvertTo-Json -Compress
$hashB = (Post-Image ('CFGSTEER scale=' + $Scale) $bodyB ($Dir + '\cfg-steered.png'))

if ($hashA -and $hashB) {
    if ($hashA -ne $hashB) { Write-Output 'RESULT: CFG_DIFFERS (steering changes the pixels)' }
    else { Write-Output 'RESULT: CFG_IDENTICAL (steering had NO effect -- defect)' }
} else { Write-Output 'RESULT: INCOMPLETE' }

$healthy = $false
try { $healthy = ([int](Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).StatusCode -eq 200) } catch {}
Write-Output ('SERVER_HEALTHY=' + $healthy)
