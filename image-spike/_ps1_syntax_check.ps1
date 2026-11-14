param([Parameter(Mandatory = $true)][string]$Path)

$errs = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$errs)
if ($errs -and $errs.Count -gt 0) {
    $errs | ForEach-Object { Write-Output ('PARSE_ERROR line ' + $_.Extent.StartLineNumber + ': ' + $_.Message) }
    exit 1
}
Write-Output ('PARSE_OK ' + $Path)
exit 0
