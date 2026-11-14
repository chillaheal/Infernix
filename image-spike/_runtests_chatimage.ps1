# Head copied from Infernix\.agent\_check_serve_host_tests.bat: rebuild ONE test target and run it.
# Narrow on purpose -- infernix_serve is a static lib, so this never relinks the running serve exe.
# Same shape as _runtests.ps1, only the target and its exe named differently: the chat image tool's
# own gate, which is where the stringified-list fix lands.
$root = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
if (-not (Test-Path -LiteralPath $root)) { Write-Output "MISSING $root"; exit 2 }
Set-Location -LiteralPath $root
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars)) { Write-Output "MISSING $vcvars"; exit 2 }
$env:PATH = $env:PATH + ';C:\vcpkg\installed\x64-windows\bin'
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
& cmd.exe /c "`"$vcvars`" >nul && cmake --build build --config Release --target infernix_chat_image_tool_test -j"
Write-Output "BUILD_EXIT=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { exit 1 }
& cmd.exe /c "`"$vcvars`" >nul && build\tests\Release\infernix_chat_image_tool_test.exe"
Write-Output "TESTEXIT_chat_image_tool=$LASTEXITCODE"
