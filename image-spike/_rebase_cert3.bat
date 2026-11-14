@echo off
setlocal
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_rebase_cert3.log

echo === BUILD THE TWO REMAINING CERT TARGETS (log %LOG%) ===
cmake --build build --config Debug -j 8 --target infernix_qwen4_exp_image_interlude_dit_real_test infernix_qwen4_exp_image_interlude_full_real_test > %LOG% 2>&1
if errorlevel 1 goto :fail
echo CERT3_DONE
exit /b 0

:fail
echo CERT3_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
