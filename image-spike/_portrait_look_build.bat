@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_portrait_look_build.log

REM Release test target only. Nothing here links infernix-serve.exe, so the resident server on :8888 is
REM untouched; the render itself runs inside blackout_portrait_look.ps1, not here. No new files, no new
REM CMake target, so no reconfigure.
echo === BUILD infernix_image_edit_behavior_test === > %LOG%
cmake --build build --config Release -j 8 --target infernix_image_edit_behavior_test >> %LOG% 2>&1
if errorlevel 1 goto :fail
echo PORTRAIT_BUILD_DONE
exit /b 0

:fail
echo PORTRAIT_BUILD_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
