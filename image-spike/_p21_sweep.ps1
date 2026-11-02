# Does the same composite look better when the model gets the pixel budget it was trained on?
#
# The only things that change from the 23:09 sweep are the canvas and the step count: 512x576 at 20 steps
# became 1024x1152 (same aspect, both sides a multiple of 32) at 40 steps, which is the official default
# regime -- output_resolution 1024 and about 40 denoise steps -- while 20 steps was our own shortcut. The
# prompt is copied byte for byte out of _p20_sweep.ps1 so that a difference in the picture is a difference
# in size or steps and nothing else. The reference is the 525x581 composite lanczos-upscaled by
# _p21_up.py, because the man's photo has no more pixels in it than that.
#
# auto, not true: a true that cannot reserve the rewriter is a 503 by design (measured 23:02: 15788 MiB
# yieldable plus 790 MiB free against 19042 plus 512), and a bigger canvas only makes that tighter.
param(
    [int]$PauseSeconds = 10800,
    [string]$Size = '1024x1152',
    [int]$Steps = 40,
    [ValidateSet('auto', 'true', 'false')]
    [string]$Enhance = 'auto'
)
$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$Here   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\faceswap'
$Swap   = Join-Path $Here '_p20_swap.py'
$Up     = Join-Path $Here '_p21_up.py'
$Comp   = Join-Path $Here '_p20_composite_t1.png'
$Big    = Join-Path $Here ('_p21_composite_' + $Size + '.png')
$Log    = Join-Path $Here '_p21_sweep.log'
$Pause  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\_serve_watch.pause'

function Log($m) {
    $line = (Get-Date).ToString('HH:mm:ss') + ' ' + $m
    Write-Host $line
    Add-Content -LiteralPath $Log -Value $line
}

foreach ($p in @($Python, $Swap, $Up, $Comp)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('MISSING ' + $p); exit 2 }
}

# Seconds, not milliseconds, whatever the field is called: _serve_watch.py:294 compares it straight
# against time.time(), so a value in ms can never expire and an abandoned sweep would leave the card with
# no server and no keepalive. The naming is a bug in the watchdog, recorded for the todo; this writer
# follows the reader that is actually deployed.
$ends = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() + $PauseSeconds)
(@{owner = '_p21_sweep.ps1'; ends_unix_ms = $ends} | ConvertTo-Json -Compress) |
    Set-Content -LiteralPath $Pause -Encoding ascii
Log ('PAUSE wrote ' + $Pause + ' for ' + $PauseSeconds + ' s')

if (Test-Path -LiteralPath $Big) { Remove-Item -LiteralPath $Big -Force }
foreach ($line in (& $Python -I $Up $Comp $Big $Size 2>&1)) { Log ('UP   ' + $line) }
if (-not (Test-Path -LiteralPath $Big)) { Log 'NO_COMPOSITE'; exit 1 }

$PromptSingle = 'On <image1> the woman''s head has already been pasted onto the bodybuilder''s shoulders and it is unfinished: make it one photograph. Let her hair fall naturally with no cut edge, blending the outer strands into the sunlight and the blurred garden behind him. Give the face and the neck the same deep tan and the same sunlight as the chest and shoulders, and let the jaw and the neck join the shoulders without a seam. Keep the body, the pose, the hands, the shorts, the tattoo and the garden as they are.'

$Runs = @(
    @{name = 'c1_s21'; seed = 21},
    @{name = 'c2_s7';  seed = 7},
    @{name = 'c3_s33'; seed = 33}
)

foreach ($r in $Runs) {
    $out = Join-Path $Here ('_p21_' + $r.name + '.png')
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
    $argv = @('-I', $Swap, '--out', $out, '--prompt', $PromptSingle,
              '--size', $Size, '--steps', ([string]$Steps), '--seed', ([string]$r.seed),
              '--enhance', $Enhance, '--ref', $Big)
    Log ('RUN ' + $r.name + ' refs=1 seed=' + $r.seed + ' ' + $Size + ' steps=' + $Steps)
    foreach ($line in (& $Python @argv 2>&1)) { Log ('   ' + $line) }
    if (-not (Test-Path -LiteralPath $out)) { Log ('   NO_OUTPUT ' + $r.name) }
}

Remove-Item -LiteralPath $Pause -Force -ErrorAction SilentlyContinue
Log 'PAUSE cleared, healing is live again'

$toneTargets = @()
foreach ($r in $Runs) {
    $out = Join-Path $Here ('_p21_' + $r.name + '.png')
    if (Test-Path -LiteralPath $out) { $toneTargets += $out }
}
if ($toneTargets.Count) {
    $tone = Join-Path $Here '_tone.py'
    foreach ($line in (& $Python -I $tone @toneTargets 2>&1)) { Log ('TONE ' + $line) }
}
Log 'SWEEP_DONE'
exit 0
