$ErrorActionPreference = 'Continue'
# Read-only: did infernix-serve.exe actually crash (Windows error reporting / .NET runtime), and when?
$since = (Get-Date).AddDays(-4)
$found = 0
$events = Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $since } -ErrorAction SilentlyContinue
foreach ($e in $events) {
    if ($e.Message -notmatch 'infernix') { continue }
    $found++
    $head = ($e.Message -split "`r?`n" | Select-Object -First 6) -join ' | '
    Write-Output ($e.TimeCreated.ToString('yyyy-MM-dd HH:mm:ss') + '  id=' + $e.Id + '  lvl=' + $e.Level + '  ' + $head.Substring(0, [Math]::Min(300, $head.Length)))
}
Write-Output ('INFERNIX_APPLICATION_EVENTS=' + $found)
