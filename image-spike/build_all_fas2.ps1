# Bundle B fas 2 build: the three bundle-A cert targets (Debug, what blackout_bundle_a.ps1 runs) and
# the stage bench (Release). Narrow targets only -- never the infernix-serve.exe app target.
$ErrorActionPreference = 'Continue'
$repo = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
Set-Location $repo
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'

$log = Join-Path $repo 'image-spike\build_fas2.log'
"=== Debug cert targets $(Get-Date) ===" | Out-File -FilePath $log -Encoding ascii

& cmake --build build --config Debug -j 8 --target infernix_image_need_ledger_real_test infernix_qwen4_exp_image_interlude_prod_real_test infernix_qwen4_exp_image_interlude_failures_real_test *>> $log
"DEBUG_CERT_RC=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding ascii

"=== Release stage bench $(Get-Date) ===" | Out-File -FilePath $log -Append -Encoding ascii
& cmake --build build --config Release -j 8 --target infernix_image_stage_bench *>> $log
"RELEASE_BENCH_RC=$LASTEXITCODE" | Out-File -FilePath $log -Append -Encoding ascii
"DONE $(Get-Date)" | Out-File -FilePath $log -Append -Encoding ascii
