# Swap the shipped executables in the user's desktop folder for the ones built from the merged tree, then
# bring the live server back up -- inside a coordinated blackout window, because the resident server holds
# the file lock on infernix-serve.exe and ~31 GiB of VRAM.
#
# Nothing is invented here. The command line the server comes back with is READ from the user's own launcher
# bat ('start flash next local.bat') by taking the line that invokes infernix-serve.exe; the only edit is the
# --image-artifact path, v2 -> v4, because M0 (image-edit/NOTES-M0.md) found the vision weights absent from
# v2 and present in v3/v4, and every edit gate ran against v4. The user's bat is not modified.
#
# The two exes are copied (not moved) to a timestamped backup folder first, and if the new server does not
# come back healthy the old exes are restored and the user's own bat is relaunched.

param(
    [int]$GateMaxWaitSec = 900,
    [int]$GateExpectedMinutes = 30,
    [int]$GraceSec = 60,
    [int]$HealthyIterations = 240,
    [string]$ImageArtifactOverride = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix',
    [string]$OldImageArtifact = 'Qwen-Image-2.1-FP8-v2.infernix'
)

$ErrorActionPreference = 'Continue'

$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$userBat    = Join-Path $desktopDir 'start flash next local.bat'
$buildDir   = Join-Path $repo 'build\apps\Release'
$exes       = @('infernix-serve.exe', 'infernix.exe')
$port       = 8888
$runBat     = Join-Path $repo 'image-spike\_serve_swap_run.bat'
$serveLog   = Join-Path $repo 'image-spike\serve_swap_server.out.log'
$serveErr   = Join-Path $repo 'image-spike\serve_swap_server.err.log'
$stamp      = (Get-Date).ToString('yyyyMMddTHHmmss')
$backupDir  = Join-Path $desktopDir ('_pre-swap-' + $stamp)

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    $r = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
         Where-Object { [int]$_.RemotePort -eq 0 } | Select-Object -First 1
    if ($r) { return [int]$r.OwningProcess }
    return 0
}
function Gpu-UsedMib {
    $v = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
    if ($v) { return [int]$v }
    return -1
}
function Launch-Bat($bat) { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $bat + '"') | Out-Null }
function Wait-Healthy($iters) {
    for ($i = 0; $i -lt $iters; $i++) {
        Start-Sleep -Seconds 5
        try {
            $h = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/health' -UseBasicParsing -TimeoutSec 4
            if ([int]$h.StatusCode -eq 200) {
                $m = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 8
                if ([int]$m.StatusCode -eq 200 -and ($m.Content -match 'qwen3\.8')) { return $true }
            }
        } catch { }
    }
    return $false
}

foreach ($p in @($userBat, $buildDir, $ImageArtifactOverride)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}
foreach ($e in $exes) {
    $src = Join-Path $buildDir $e
    if (-not (Test-Path -LiteralPath $src)) { Log ('FATAL missing built exe: ' + $src); exit 2 }
    Log ('SRC ' + $e + ' ' + (Get-Item -LiteralPath $src).Length + ' ' + (Get-Item -LiteralPath $src).LastWriteTime)
}
$freeGib = [math]::Round(((Get-PSDrive C).Free / 1GB), 1)
Log ('DISK free on C: ' + $freeGib + ' GiB')
if ($freeGib -lt 3) { Log 'FATAL less than 3 GiB free, the backups would not fit'; exit 2 }

# Build the launch line from the user's own bat: the one line that runs infernix-serve.exe.
$userLines = Get-Content -LiteralPath $userBat
$serveLine = $null
foreach ($l in $userLines) {
    if ($l -match 'infernix-serve\.exe') { if ($serveLine) { Log 'FATAL the launcher has more than one infernix-serve.exe line'; exit 2 }; $serveLine = $l }
}
if (-not $serveLine) { Log 'FATAL no infernix-serve.exe line in the launcher'; exit 2 }
Log ('LINE captured from the launcher, length=' + $serveLine.Length)
$serveLine = $serveLine.Trim()
if ($serveLine.StartsWith('start ', [StringComparison]::OrdinalIgnoreCase)) {
    Log 'FATAL the captured line starts with start -- refusing to guess how to unwrap it'; exit 2
}

# Replace the WHOLE quoted path, not the file name inside it. The launcher spells the artifact as
#   --image-artifact "C:\...\models\Qwen-Image-2.1-FP8-v2.infernix"
# and replacing only the file name left the models directory in front of the absolute override, which
# the server then opened as "...models\C:\...\v4.infernix" -- CreateFileW 500 for t2i and, because the
# artifact never loaded, 503 image_generation_not_configured for a reference.
$v2Pattern = '"[^"]*' + [regex]::Escape($OldImageArtifact) + '"'
$v2Replacement = '"' + $ImageArtifactOverride + '"'
$newLine = $serveLine -replace $v2Pattern, $v2Replacement
if ($newLine -eq $serveLine) {
    Log ('FATAL the artifact override matched nothing -- the launcher does not carry ' + $OldImageArtifact); exit 2
}

$gate = Join-Path $repo 'image-spike\window_gate.ps1'
if (-not (Test-Path -LiteralPath $gate)) { Log ('FATAL missing gate: ' + $gate); exit 2 }
& $gate -Action StartWindow -Owner (Split-Path -Leaf $MyInvocation.MyCommand.Path) -Reason (Split-Path -Leaf $MyInvocation.MyCommand.Path) -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, nothing was copied'; exit 10 }

function Release-Window { & $gate -Action EndWindow | Out-Null; & $gate -Action Release | Out-Null }

Log ('PHASE0 grace ' + $GraceSec + 's')
Start-Sleep -Seconds $GraceSec

$livePid = Listener-Pid
Log ('PHASE1 live pid=' + $livePid)
if ($livePid -gt 0) { taskkill /F /T /PID $livePid 2>&1 | Out-Null }
# A sibling that lost the port race or exited blocked in a CUDA call is not the listener and keeps
# the exe's image section mapped, which makes Copy-Item fail with "user-mapped section open".
taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
taskkill /F /IM infernix.exe 2>&1 | Out-Null
for ($i = 0; $i -lt 90; $i++) {
    if (Get-Process infernix-serve -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 1; continue }
    break
}
for ($i = 0; $i -lt 40; $i++) {
    $u = Gpu-UsedMib
    if ($u -ge 0 -and $u -lt 2000) { break }
    Start-Sleep -Seconds 3
}
Log ('PHASE1 drained gpu_used_mib=' + (Gpu-UsedMib))

# PHASE2: back up, then copy the merged-tree exes over the desktop copies.
New-Item -ItemType Directory -Path $backupDir | Out-Null
foreach ($e in $exes) {
    $dst = Join-Path $desktopDir $e
    if (Test-Path -LiteralPath $dst) {
        $srcItem = Get-Item -LiteralPath $dst
        Copy-Item -LiteralPath $dst -Destination (Join-Path $backupDir $e) -Force
        Log ('BACKUP ' + $e + ' ' + $srcItem.Length + ' ' + $srcItem.LastWriteTime + ' -> ' + $backupDir)
    } else {
        Log ('BACKUP ' + $e + ' was not present in the desktop folder')
    }
}
# Windows refuses to overwrite a file whose image section is still mapped by any process (a serve
# that exited blocked in a CUDA call keeps it pinned after taskkill /F), but renaming it is allowed.
# So the old exe is moved into the backup folder under a .mapped name and the new one copied in fresh.
function Move-Aside($path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $leaf = Split-Path -Leaf $path
    $tag  = Join-Path $backupDir ($leaf + '.mapped-' + $stamp)
    try {
        Move-Item -LiteralPath $path -Destination $tag -Force
        Log ('MOVED-ASIDE ' + $leaf + ' -> ' + $tag)
    } catch {
        Log ('MOVE_ASIDE_FAILED ' + $path + ': ' + $_.Exception.Message)
    }
}
foreach ($e in $exes) {
    $src = Join-Path $buildDir $e
    $dst = Join-Path $desktopDir $e
    Move-Aside $dst
    Copy-Item -LiteralPath $src -Destination $dst -Force
    $got = Get-Item -LiteralPath $dst
    if ($got.Length -ne (Get-Item -LiteralPath $src).Length) {
        Log ('COPY_VERIFY_FAILED ' + $e + ' ' + $got.Length + ' != ' + (Get-Item -LiteralPath $src).Length)
        foreach ($b in $exes) {
            $bp = Join-Path $backupDir $b
            if (Test-Path -LiteralPath $bp) {
                Move-Aside (Join-Path $desktopDir $b)
                Copy-Item -LiteralPath $bp -Destination (Join-Path $desktopDir $b) -Force
            }
        }
        Log 'restored the backup exes; launching the user bat unchanged'
        Launch-Bat $userBat
        $h = Wait-Healthy $HealthyIterations
        Log ('ROLLED_BACK healthy=' + $h)
        Release-Window
        exit 7
    }
    Log ('SWAPPED ' + $e + ' ' + $got.Length + ' ' + $got.LastWriteTime)
}

# PHASE3: write the run bat (the copied line, artifact swapped, output redirected so the startup log is
# readable) and read it back before launching. The guards below must all pass.
# The whole launcher is copied, not just the serve line: the line needs %ART%, which the dir loop above it
# sets. Only the serve line is touched -- artifact swapped, output redirected so the startup log is readable.
$outLines = @()
foreach ($l in $userLines) {
    if ($l -match 'infernix-serve\.exe') {
        $outLines += (($l -replace $v2Pattern, $v2Replacement) + ' > "' + $serveLog + '" 2> "' + $serveErr + '"')
    } else {
        $outLines += $l
    }
}
[System.IO.File]::WriteAllLines($runBat, $outLines, [System.Text.Encoding]::ASCII)
$text = Get-Content -LiteralPath $runBat -Raw
$need = @('infernix-serve.exe', $v2Replacement, '--port 8888', '--image-text-processor', '--draft-model', '--prefix-cache-file')
$bad  = @($OldImageArtifact)
$missing = @()
foreach ($n in $need) { if (-not $text.Contains($n)) { $missing += $n } }
$presentBad = @()
foreach ($b in $bad) { if ($text.Contains($b)) { $presentBad += $b } }
# The path bug this guard exists for: a directory glued in front of an absolute path.
if ($text -match 'models\\C:\\') { $presentBad += 'a models directory in front of the absolute artifact path' }
if ($missing.Count -gt 0 -or $presentBad.Count -gt 0) {
    Log ('RUNBAT_REFUSED missing=[' + ($missing -join ',') + '] unwanted=[' + ($presentBad -join ',') + ']')
    Log 'the desktop exes are already swapped; the user bat is the way back'
    Launch-Bat $userBat
    $h = Wait-Healthy $HealthyIterations
    Log ('RUNBAT_REFUSED relaunched the user bat healthy=' + $h)
    Release-Window
    exit 6
}
Log ('RUNBAT ok, ' + $outLines.Count + ' lines, copied from ' + $userBat)

Log 'PHASE3 launching the swapped server'
if (Test-Path -LiteralPath $serveLog) { Remove-Item -LiteralPath $serveLog -Force }
if (Test-Path -LiteralPath $serveErr) { Remove-Item -LiteralPath $serveErr -Force }
Launch-Bat $runBat
$healthy = Wait-Healthy $HealthyIterations
Log ('HEALTHY=' + $healthy + ' gpu_used_mib=' + (Gpu-UsedMib))

if ($healthy) {
    try {
        $cap = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/capabilities' -UseBasicParsing -TimeoutSec 8
        Log ('CAPABILITIES status=' + [int]$cap.StatusCode)
        $cap.Content | ForEach-Object { Log ('CAP| ' + $_.Substring(0, [Math]::Min(900, $_.Length))) }
    } catch { Log ('CAPABILITIES failed: ' + $_.Exception.Message) }
    if (Test-Path -LiteralPath $serveLog) {
        Log '--- serve stdout, the lines that name the image route, the vision tower or a refusal ---'
        Select-String -LiteralPath $serveLog -Pattern 'image|vision|reference|artifact' -CaseSensitive:$false |
            Select-Object -Last 40 | ForEach-Object { Log ('SRV| ' + $_.Line) }
    }
    if (Test-Path -LiteralPath $serveErr) {
        Get-Content -LiteralPath $serveErr -Tail 20 | ForEach-Object { Log ('ERR| ' + $_) }
    }
} else {
    Log 'NOT HEALTHY: restoring the backup exes and the user bat'
    foreach ($b in $exes) {
        $bp = Join-Path $backupDir $b
        if (Test-Path -LiteralPath $bp) {
            Move-Aside (Join-Path $desktopDir $b)
            Copy-Item -LiteralPath $bp -Destination (Join-Path $desktopDir $b) -Force
        }
    }
    $k = Listener-Pid
    if ($k -gt 0) { taskkill /F /T /PID $k 2>&1 | Out-Null }
    Start-Sleep -Seconds 5
    Launch-Bat $userBat
    $h2 = Wait-Healthy $HealthyIterations
    Log ('ROLLED_BACK healthy=' + $h2)
    if (Test-Path -LiteralPath $serveLog) { Get-Content -LiteralPath $serveLog -Tail 40 | ForEach-Object { Log ('SRV| ' + $_) } }
    if (Test-Path -LiteralPath $serveErr) { Get-Content -LiteralPath $serveErr -Tail 40 | ForEach-Object { Log ('ERR| ' + $_) } }
}

Release-Window
Log ('RESULT healthy=' + $healthy + ' backup=' + $backupDir)
exit 0
