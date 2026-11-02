@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_pr28_build_test.log
set BIN=build\tests\Debug
set "PATH=C:\vcpkg\installed\x64-windows\bin;C:\vcpkg\installed\x64-windows\debug\bin;%PATH%"

REM sources.cmake gained expert_profile.cpp, so reconfigure first. Debug libraries and one test exe only:
REM nothing here links infernix-serve.exe, so the server on :8888 keeps running.
echo === RECONFIGURE === > %LOG%
cmake -S . -B build -DBUILD_TESTING=ON >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === BUILD ENGINE + EXPERT CACHE TEST (Debug) ===
cmake --build build --config Debug -j 8 --target infernix_engine infernix_qwen4_exp_expert_cache_test >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === RUN THE EXPERT CACHE TEST ===
"%BIN%\infernix_qwen4_exp_expert_cache_test.exe" > image-spike\_pr28_expert_cache.out 2>&1
echo RESULT infernix_qwen4_exp_expert_cache_test exit=!errorlevel!
echo PR28_DONE
exit /b 0

:fail
echo PR28_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
