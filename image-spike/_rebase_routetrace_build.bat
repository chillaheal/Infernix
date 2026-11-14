@echo off
setlocal
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_rebase_routetrace_build.log
set OLD=image-spike\route_trace_premerge_2026-10-21.exe
set EXE=build\tests\Debug\infernix_qwen4_exp_route_trace_real_test.exe

REM PHASE4 of blackout_rebase_cert.ps1 runs the Debug route-trace exe as the AFTER build. That exe is still
REM the 2026-10-21 pre-merge binary, so the window would compare a pre-merge exe with itself. Save it aside
REM (it becomes -ExactOldExe) and rebuild the target from the merged tree. Debug test target only: it does
REM not link infernix-serve.exe and does not touch :8888.
if not exist "%OLD%" copy /y "%EXE%" "%OLD%" >nul && echo OLD_SAVED
for %%F in ("%OLD%") do echo   fore-merge exe: size=%%~zF mtime=%%~tF
cmake --build build --config Debug -j 8 --target infernix_qwen4_exp_route_trace_real_test > %LOG% 2>&1
if errorlevel 1 goto :fail
for %%F in ("%EXE%") do echo   efter-merge exe: size=%%~zF mtime=%%~tF
echo ROUTETRACE_BUILD_DONE
exit /b 0

:fail
echo ROUTETRACE_BUILD_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
