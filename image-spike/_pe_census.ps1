# PE census the user asked for after every serve launch ("det var 3 prompt enchancers igang samtidigt").
# One file so no quoting of a WQL filter through bash is needed. Prints every python process whose command
# line names pe_worker, with the port it should hold (8899) and whether that port answers at all.
$ErrorActionPreference = 'Continue'
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'pe_worker' }
Write-Output ("pe_worker_count=" + @($procs).Count)
foreach ($p in $procs) {
    Write-Output ("pid " + $p.ProcessId + " started " + $p.CreationDate + " ws " +
        [math]::Round((Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue).WorkingSet64 / 1MB) + " MB")
    Write-Output ("   " + $p.CommandLine)
}
$listening = Get-NetTCPConnection -LocalPort 8899 -State Listen -ErrorAction SilentlyContinue
if ($listening) { Write-Output ("port_8899_listening_by_pid=" + ($listening.OwningProcess -join ',')) }
else { Write-Output 'port_8899_NOT_LISTENING' }
try {
    $h = (Invoke-WebRequest -Uri 'http://127.0.0.1:8899/health' -UseBasicParsing -TimeoutSec 8).Content
    Write-Output ("sidecar_health=" + $h)
} catch { Write-Output ('sidecar_health_error=' + $_.Exception.Message) }
Write-Output 'CENSUS_DONE'
