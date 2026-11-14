# Sweep the harmonize step over the composites this round built, and leave one log to judge.
#
# The two things being measured are the ones the user named: the skin tone of the head against the tanned
# body, and the hair, which the 21:59 result cut off. Both are tested on the single-reference route (the
# composite is the whole story) and on two references (her own photo beside it, which the 21:59 run found
# carries identity but has never been tried for hair). The prompt-enhancer is asked for on every run, so
# the log must show prompt_enhance_outcome=applied, and it does not: the watchdog is paused for the sweep
# because a chat probe that queues behind an image job gets the serve killed around 600 s, which is what
# ate the on-case of the 22:20 probe run.
param(
    [string]$BaseUrl = 'http://127.0.0.1:8888',
    [int]$PauseSeconds = 7200,
    # auto, not true: the first sweep at 23:02 was 503 image_vram_insufficient on every run, because the
    # fit check found only 15788 MiB yieldable plus 790 MiB free against the 19042 MiB the rewriter
    # reserves plus 512 MiB. true keeps that refusal by design; auto is allowed to draw without the
    # rewriter, which is what the capability document promises, and the outcome field still says which
    # of the two happened.
    [ValidateSet('auto', 'true', 'false')]
    [string]$Enhance = 'auto'
)
$ErrorActionPreference = 'Continue'
$Python = 'C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe'
$Here   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\faceswap'
$Swap   = Join-Path $Here '_p20_swap.py'
$Comp   = Join-Path $Here '_p20_composite_t1.png'
$Woman  = 'C:\Users\Micke\Desktop\mitt egna ansikte.jpg'
$Log    = Join-Path $Here '_p20_sweep.log'
$Pause  = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\_serve_watch.pause'

function Log($m) {
    $line = (Get-Date).ToString('HH:mm:ss') + ' ' + $m
    Write-Host $line
    Add-Content -LiteralPath $Log -Value $line
}

function Test-File($p) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('MISSING ' + $p); return $false }
    return $true
}

foreach ($p in @($Python, $Swap, $Comp, $Woman)) { if (-not (Test-File $p)) { exit 2 } }

# The keepalive honours this file and lets an abandoned one expire on its own, which is why it carries
# ends_unix_ms rather than being a bare marker.
$ends = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() + ($PauseSeconds * 1000L))
(@{owner = '_p20_sweep.ps1'; ends_unix_ms = $ends} | ConvertTo-Json -Compress) |
    Set-Content -LiteralPath $Pause -Encoding ascii
Log ('PAUSE wrote ' + $Pause + ' for ' + $PauseSeconds + ' s')

$PromptSingle = 'On <image1> the woman''s head has already been pasted onto the bodybuilder''s shoulders and it is unfinished: make it one photograph. Let her hair fall naturally with no cut edge, blending the outer strands into the sunlight and the blurred garden behind him. Give the face and the neck the same deep tan and the same sunlight as the chest and shoulders, and let the jaw and the neck join the shoulders without a seam. Keep the body, the pose, the hands, the shorts, the tattoo and the garden as they are.'

$PromptTwo = '<image1> is the canvas: the woman''s head is already pasted on his shoulders and must be finished into one photograph. Take her face, eyes, smile and hair from <image2> so she stays the same person. Let the hair fall naturally with no cut edge, blend the outer strands into the sunlight and the garden, give the face and the neck the same deep tan and the same sunlight as the chest and shoulders of <image1>, and join the jaw and the neck into the shoulders without a seam. Keep the body, the pose, the hands, the shorts, the tattoo and the garden of <image1> as they are.'

$Runs = @(
    @{name = 'a1_s7';  refs = @($Comp);           seed = 7},
    @{name = 'a2_s21'; refs = @($Comp);           seed = 21},
    @{name = 'a3_s33'; refs = @($Comp);           seed = 33},
    @{name = 'b1_s7';  refs = @($Comp, $Woman);   seed = 7},
    @{name = 'b2_s21'; refs = @($Comp, $Woman);   seed = 21}
)

foreach ($r in $Runs) {
    $out = Join-Path $Here ('_p20_' + $r.name + '.png')
    if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }
    $prompt = if ($r.refs.Count -gt 1) { $PromptTwo } else { $PromptSingle }
    $argv = @('-I', $Swap, '--out', $out, '--prompt', $prompt,
              '--size', '512x576', '--steps', '20', '--seed', ([string]$r.seed), '--enhance', $Enhance)
    foreach ($ref in $r.refs) { $argv += @('--ref', $ref) }
    Log ('RUN ' + $r.name + ' refs=' + $r.refs.Count + ' seed=' + $r.seed)
    $output = & $Python @argv 2>&1
    foreach ($line in $output) { Log ('   ' + $line) }
    if (-not (Test-Path -LiteralPath $out)) { Log ('   NO_OUTPUT ' + $r.name) }
}

Remove-Item -LiteralPath $Pause -Force -ErrorAction SilentlyContinue
Log 'PAUSE cleared, healing is live again'

# The tone metric is the number the user's complaint is about: face minus chest, per channel.
$toneTargets = @()
foreach ($r in $Runs) {
    $out = Join-Path $Here ('_p20_' + $r.name + '.png')
    if (Test-Path -LiteralPath $out) { $toneTargets += $out }
}
if ($toneTargets.Count) {
    $tone = Join-Path $Here '_tone.py'
    foreach ($line in (& $Python -I $tone @toneTargets 2>&1)) { Log ('TONE ' + $line) }
}
Log 'SWEEP_DONE'
exit 0
