param(
    [int]$GateMaxWaitSec = 0,        # 0 = one look at :8888 and refuse if it is busy; N = wait up to N s
    [int]$GateExpectedMinutes = 10   # a rollback restart is a short window
)

$ErrorActionPreference = 'Continue'

# Put the binary that served before the C task 6 deploy back on the Desktop instance and bring the
# server up again. The deployed build is being rolled back because an image request takes the whole
# process down with it (rmsnorm.cu:177 CUDA_CHECK out of memory), so it must not stay in service.

$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$desktopExe = Join-Path $desktopDir 'infernix-serve.exe'
$backupExe  = Join-Path $desktopDir 'infernix-serve.exe.pre-task6.bak'
$serveBat   = Join-Path $desktopDir 'start flash next.bat'
$port       = 8888

function Log($m) { Write-Host ((Get-Date).ToString('HH:mm:ss') + ' ' + $m) }
function Listener-Pid {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return [int]$c.OwningProcess }
    return 0
}

foreach ($p in @($backupExe, $serveBat)) {
    if (-not (Test-Path -LiteralPath $p)) { Log ('FATAL missing path: ' + $p); exit 2 }
}

# The window gate: :8888 carries every other Claude Code session's turns, so it is only taken when
# it is quiet. This script has no $repo of its own, so the gate path is spelled out.
$gate = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\window_gate.ps1'
if (-not (Test-Path -LiteralPath $gate)) { Log ('FATAL missing gate: ' + $gate); exit 2 }
& $gate -Action StartWindow -Owner (Split-Path -Leaf $MyInvocation.MyCommand.Path) -Reason (Split-Path -Leaf $MyInvocation.MyCommand.Path) -MaxWaitSec $GateMaxWaitSec -ExpectedMinutes $GateExpectedMinutes
if ($LASTEXITCODE -ne 0) { Log 'WINDOW_REFUSED_BY_GATE: nothing was stopped, :8888 left as it was'; exit 10 }

$p = Listener-Pid
Log ('PHASE1 stopping the serve on :' + $port + ' pid=' + $p)
if ($p -gt 0) { taskkill /F /T /PID $p 2>&1 | Out-Null }
taskkill /F /IM infernix-serve.exe 2>&1 | Out-Null
for ($i = 0; $i -lt 30; $i++) {
    if ((Listener-Pid) -eq 0 -and -not (Get-Process infernix-serve -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Seconds 2
}

Copy-Item -LiteralPath $backupExe -Destination $desktopExe -Force
Log ('PHASE2 restored exe size=' + (Get-Item -LiteralPath $desktopExe).Length)

& $gate -Action EndWindow | Out-Null   # the 503 stub has to release :8888 before serve binds it again
Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $serveBat + '"') | Out-Null
Log 'PHASE2 the start bat was launched (detached, visible window)'

$healthy = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 5
    try {
        $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8888/v1/models' -UseBasicParsing -TimeoutSec 5
        if ([int]$r.StatusCode -eq 200 -and ($r.Content -match 'Qwen3\.8-Flash-Next')) { $healthy = $true; break }
    } catch { }
}
& $gate -Action Release | Out-Null
Log ('RESULT rollback healthy=' + $healthy + ' pid=' + (Listener-Pid))
exit 0
