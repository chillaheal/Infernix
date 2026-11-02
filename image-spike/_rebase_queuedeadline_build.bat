@echo off
setlocal
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_rebase_queuedeadline_build.log
set EXE=build\tests\Debug\infernix_qwen4_exp_image_queue_deadline_real_test.exe

REM D2 in that test failed in the window. The test is made deterministic first (the lane holder must be
REM resident before the second request is submitted) and now needs its Debug exe rebuilt. Debug test
REM target only: it does not link infernix-serve.exe and does not touch :8888.
echo before:
for %%F in ("%EXE%") do echo   size=%%~zF mtime=%%~tF
cmake --build build --config Debug -j 8 --target infernix_qwen4_exp_image_queue_deadline_real_test > %LOG% 2>&1
if errorlevel 1 goto :fail
echo after:
for %%F in ("%EXE%") do echo   size=%%~zF mtime=%%~tF
echo QUEUEDEADLINE_BUILD_DONE
exit /b 0

:fail
echo QUEUEDEADLINE_BUILD_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
