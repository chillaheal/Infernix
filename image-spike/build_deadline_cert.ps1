# Build the Debug cert target for the admission-queue deadline test. Visible window; writes a log.
$ErrorActionPreference = 'Stop'
$root   = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$build  = Join-Path $root 'build'
$log    = Join-Path $root 'image-spike\build_deadline_cert.log'
$target = 'infernix_qwen4_exp_image_queue_deadline_real_test'

if (-not (Test-Path -LiteralPath $build)) { Write-Output ('FATAL no build dir: ' + $build); exit 2 }
if (-not (Test-Path -LiteralPath 'C:\vcpkg')) { Write-Output 'FATAL no C:\vcpkg'; exit 2 }
$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
Write-Output ('CMAKE=' + $cmake)
$env:VCPKG_ROOT           = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'

& $cmake --build $build --config Debug --target $target -j 2>&1 | Tee-Object -FilePath $log
Write-Output ('BUILD_EXIT=' + $LASTEXITCODE)
