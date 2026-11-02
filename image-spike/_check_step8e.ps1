$ErrorActionPreference = 'Continue'
$ps1 = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\blackout_step8e_rollout.ps1'
$e = $null
$src = Get-Content -Raw $ps1
[void][System.Management.Automation.PSParser]::Tokenize($src, [ref]$e)
if ($e.Count -eq 0) { 'PARSE_OK' } else { 'PARSE_ERR ' + $e.Count; $e | ForEach-Object { Write-Output ($_.Token.StartLine.ToString() + ': ' + $_.Message) } }
if ($src -match '[^\t\n\r -~]') { 'NONASCII_FOUND' } else { 'ASCII_OK' }
$paths = @(
  'C:\Users\Micke\Desktop\Infernix FlashNext\infernix-serve.exe',
  'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v2.infernix',
  'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor',
  'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\run_blackout_8e.bat'
)
foreach ($p in $paths) {
  if (Test-Path -LiteralPath $p) { 'OK ' + $p } else { 'MISSING ' + $p }
}
