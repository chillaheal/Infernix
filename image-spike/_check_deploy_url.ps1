param([string]$Target = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\blackout_deploy_url.ps1')
$text = Get-Content -LiteralPath $Target -Raw
$errs = $null
[System.Management.Automation.PSParser]::Tokenize($text, [ref]$errs) | Out-Null
Write-Output ('PARSE_ERRORS=' + @($errs).Count)
foreach ($e in $errs) { Write-Output ('  ' + $e.Message + ' @line ' + $e.Token.StartLine) }
foreach ($m in [regex]::Matches($text, "'([A-Za-z]:\\\\?[^']*?)'")) {
    $p = $m.Groups[1].Value
    if ($p -match '\$') { continue }
    if ($p -match '\.(png|log|bat)$') { Write-Output ('WRITABLE(exists=' + [bool](Test-Path -LiteralPath $p) + ') ' + $p); continue }
    Write-Output ('' + [bool](Test-Path -LiteralPath $p) + ' ' + $p)
}
