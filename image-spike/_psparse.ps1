param([Parameter(Mandatory = $true)][string]$Path)
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$null, [ref]$errors)
if ($errors.Count -eq 0) { Write-Output ('PS_PARSE_OK ' + $Path); exit 0 }
foreach ($error in $errors) { Write-Output ('PS_ERR line=' + $error.Extent.StartLineNumber + ' ' + $error.Message) }
exit 1
