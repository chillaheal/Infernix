# Dry check of the artifact override in blackout_serve_swap.ps1: apply the same regex to the user's own
# launcher line and print the --image-artifact token that would be launched. Nothing is stopped, copied
# or launched, and the server is not touched.
$ErrorActionPreference = 'Continue'
$repo       = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$desktopDir = 'C:\Users\Micke\Desktop\Infernix FlashNext'
$userBat    = Join-Path $desktopDir 'start flash next local.bat'
$ImageArtifactOverride = Join-Path $repo 'out\Qwen-Image-2.1-FP8-v4.infernix'
$OldImageArtifact      = 'Qwen-Image-2.1-FP8-v2.infernix'

$errors = $null
$ps1 = Join-Path $repo 'image-spike\blackout_serve_swap.ps1'
[void][System.Management.Automation.Language.Parser]::ParseFile($ps1, [ref]$null, [ref]$errors)
if ($errors -and $errors.Count -gt 0) {
    foreach ($e in $errors) { Write-Host ('PS1_PARSE_ERROR ' + $e.Extent.StartLineNumber + ' ' + $e.Message) }
    Write-Host 'PS1_PARSE=FAIL'; exit 1
}
Write-Host 'PS1_PARSE=OK'

$serveLine = $null
foreach ($l in (Get-Content -LiteralPath $userBat)) { if ($l -match 'infernix-serve\.exe') { $serveLine = $l } }
if (-not $serveLine) { Write-Host 'FATAL no serve line'; exit 2 }

$v2Pattern = '"[^"]*' + [regex]::Escape($OldImageArtifact) + '"'
$v2Replacement = '"' + $ImageArtifactOverride + '"'
$new = $serveLine -replace $v2Pattern, $v2Replacement
if ($new -eq $serveLine) { Write-Host 'REPLACE=NO_MATCH'; exit 2 }

$m = [regex]::Match($new, '--image-artifact\s+"([^"]*)"')
if (-not $m.Success) { Write-Host 'ARTIFACT_TOKEN=NOT_FOUND'; exit 2 }
$token = $m.Groups[1].Value
Write-Host ('ARTIFACT_TOKEN=' + $token)
Write-Host ('ARTIFACT_EXACT=' + ($token -eq $ImageArtifactOverride))
Write-Host ('ARTIFACT_EXISTS=' + (Test-Path -LiteralPath $token))
Write-Host ('CONCAT_BUG=' + ($new -match 'models\\C:\\'))
Write-Host ('V2_STILL_PRESENT=' + ($new.Contains($OldImageArtifact)))
