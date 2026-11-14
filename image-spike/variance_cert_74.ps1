# Task 74: systematic device cert of the variable image inputs on the LIVE server (:8888).
# Gates: seed reproducibility, seed variance, steps variance, non-square + 1024 sizes (PNG header
# dims), raw-prompt variance, and the steps floor rejection. Plain POSTs, no blackout, self-restoring
# server untouched.
$ErrorActionPreference = 'Stop'
$base = 'http://127.0.0.1:8888'
$root = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$outDir = $root + '\out\var74'
$logFile = $root + '\out\var74.out.log'
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

function Log($s) {
    $l = (Get-Date -Format 'HH:mm:ss') + ' ' + $s
    Write-Host $l
    Add-Content -LiteralPath $logFile $l
}
function Model-Id {
    try {
        $m = (Invoke-WebRequest -Uri ($base + '/v1/models') -UseBasicParsing -TimeoutSec 8).Content
        if ($m -match '"id"\s*:\s*"([^"]+)"') { return $Matches[1] }
    } catch { }
    return ''
}
# One images POST. Returns @{ http=<int>; seconds=<int>; body=<string>; png=<bool> }.
function Image-Post($model, $prompt, $size, $steps, $seed, $out, $tosec) {
    $body = @{ model = $model; prompt = $prompt; size = $size; n = 1; steps = $steps; seed = $seed; response_format = 'b64_json' } | ConvertTo-Json -Compress
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-WebRequest -Uri ($base + '/v1/images/generations') -Method POST -ContentType 'application/json' -Body $body -UseBasicParsing -TimeoutSec $tosec
        $sw.Stop()
        $obj = $resp.Content | ConvertFrom-Json
        $bytes = [System.Convert]::FromBase64String($obj.data[0].b64_json)
        [System.IO.File]::WriteAllBytes($out, $bytes)
        return @{ http = 200; seconds = [int]$sw.Elapsed.TotalSeconds; body = ''; png = $true }
    } catch {
        $sw.Stop(); $e = $_.Exception; $st = -1; $bd = ''
        if ($e.Response) {
            try { $sr = New-Object System.IO.StreamReader($e.Response.GetResponseStream()); $bd = $sr.ReadToEnd(); $sr.Close() } catch {}
            try { $st = [int]$e.Response.StatusCode } catch {}
        }
        return @{ http = $st; seconds = [int]$sw.Elapsed.TotalSeconds; body = $bd; png = $false }
    }
}
function Sha($p) { return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash }
function Png-Dims($p) {
    $b = [System.IO.File]::ReadAllBytes($p)
    $w = ([int]$b[16] * 16777216) + ([int]$b[17] * 65536) + ([int]$b[18] * 256) + [int]$b[19]
    $h = ([int]$b[20] * 16777216) + ([int]$b[21] * 65536) + ([int]$b[22] * 256) + [int]$b[23]
    return ($w.ToString() + 'x' + $h.ToString())
}

Set-Content -LiteralPath $logFile -Value ('VARIANCE74 START ' + (Get-Date -Format 'HH:mm:ss'))
$id = Model-Id
Log ('model=' + $id)
if (-not $id) { Log 'FATAL no model id on :8888'; exit 2 }

$p1 = 'a red cube on a white table'
$p2 = 'a blue sphere on a green field'
$f = { param($n) $outDir + '\' + $n }

$gA = Image-Post $id $p1 '512x512' 2 1 ( & $f 'A.png' ) 900
Log ('A 512x512 steps=2 seed=1 http=' + $gA.http + ' seconds=' + $gA.seconds + ' body=' + $gA.body.Substring(0, [Math]::Min(200, $gA.body.Length)))
$gB = Image-Post $id $p1 '512x512' 2 1 ( & $f 'B.png' ) 900
Log ('B 512x512 steps=2 seed=1 http=' + $gB.http + ' seconds=' + $gB.seconds)
$gC = Image-Post $id $p1 '512x512' 2 2 ( & $f 'C.png' ) 900
Log ('C 512x512 steps=2 seed=2 http=' + $gC.http + ' seconds=' + $gC.seconds)
$gD = Image-Post $id $p1 '512x512' 8 1 ( & $f 'D.png' ) 900
Log ('D 512x512 steps=8 seed=1 http=' + $gD.http + ' seconds=' + $gD.seconds)
$gE = Image-Post $id $p1 '512x768' 2 1 ( & $f 'E.png' ) 900
Log ('E 512x768 steps=2 seed=1 http=' + $gE.http + ' seconds=' + $gE.seconds)
$gF = Image-Post $id $p1 '1024x1024' 2 1 ( & $f 'F.png' ) 900
Log ('F 1024x1024 steps=2 seed=1 http=' + $gF.http + ' seconds=' + $gF.seconds)
$gG = Image-Post $id $p2 '512x512' 2 1 ( & $f 'G.png' ) 900
Log ('G prompt2 512x512 steps=2 seed=1 http=' + $gG.http + ' seconds=' + $gG.seconds)
# The PS 5.1 error-response stream reads back empty. curl captures the body, but PowerShell strips
# the embedded quotes of an inline --data, so the JSON body goes over a file with the @ prefix.
$gj = '{"model":"' + $id + '","prompt":"x","size":"512x512","n":1,"steps":1,"seed":1}'
$gtmp = $outDir + '\guard_body.json'
Set-Content -LiteralPath $gtmp -Value $gj -Encoding ASCII -NoNewline
$gresp = $outDir + '\guard_resp.json'
$gcode = (& curl.exe -s -o $gresp -w '%{http_code}' -X POST -H 'Content-Type: application/json' --data ('@' + $gtmp) ($base + '/v1/images/generations'))
$gbody = if (Test-Path -LiteralPath $gresp) { Get-Content -Raw -LiteralPath $gresp } else { '' }
Log ('STEPS1 http=' + $gcode + ' body=' + $gbody.Substring(0, [Math]::Min(200, $gbody.Length)))

$ok = $true
if ($gA.png -and $gB.png) { $r = ((Sha ( & $f 'A.png' )) -eq (Sha ( & $f 'B.png' ))); Log ('GATE REPRO same-seed identical=' + $r); if (-not $r) { $ok = $false } } else { Log 'GATE REPRO failed (A or B not generated)'; $ok = $false }
if ($gA.png -and $gC.png) { $r = ((Sha ( & $f 'A.png' )) -ne (Sha ( & $f 'C.png' ))); Log ('GATE SEED seed-varies=' + $r); if (-not $r) { $ok = $false } } else { Log 'GATE SEED failed (A or C not generated)'; $ok = $false }
if ($gA.png -and $gD.png) { $r = ((Sha ( & $f 'A.png' )) -ne (Sha ( & $f 'D.png' ))); Log ('GATE STEPS steps-varies=' + $r); if (-not $r) { $ok = $false } } else { Log 'GATE STEPS failed (A or D not generated)'; $ok = $false }
if ($gE.png) { $r = ((Png-Dims ( & $f 'E.png' )) -eq '512x768'); Log ('GATE NONSQUARE dims=' + (Png-Dims ( & $f 'E.png' )) + ' ok=' + $r); if (-not $r) { $ok = $false } } else { Log 'GATE NONSQUARE failed (E not generated)'; $ok = $false }
if ($gF.png) { $r = ((Png-Dims ( & $f 'F.png' )) -eq '1024x1024'); Log ('GATE SIZE1024 dims=' + (Png-Dims ( & $f 'F.png' )) + ' ok=' + $r); if (-not $r) { $ok = $false } } else { Log 'GATE SIZE1024 failed (F not generated)'; $ok = $false }
if ($gA.png -and $gG.png) { $r = ((Sha ( & $f 'A.png' )) -ne (Sha ( & $f 'G.png' ))); Log ('GATE PROMPT prompt-varies=' + $r); if (-not $r) { $ok = $false } } else { Log 'GATE PROMPT failed (A or G not generated)'; $ok = $false }
$r = (($gcode -eq '400') -and ($gbody -match 'image_steps_invalid')); Log ('GATE GUARD steps=1 rejected=' + $r); if (-not $r) { $ok = $false }

if ($ok) { Log 'VARIANCE74 PASS all input-variance gates green' } else { Log 'VARIANCE74 FAIL see gates above' }
exit $(if ($ok) { 0 } else { 1 })
