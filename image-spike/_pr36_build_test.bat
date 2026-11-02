@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_pr36_build_test.log
set BIN=build\tests\Debug
set "PATH=C:\vcpkg\installed\x64-windows\bin;C:\vcpkg\installed\x64-windows\debug\bin;%PATH%"

REM PR 36 adds metrics_json.cpp/h and a new test target, so CMake has to reconfigure before the new
REM target can be built. Debug test targets only: nothing here links infernix-serve.exe and :8888 stays up.
echo === RECONFIGURE === > %LOG%
cmake -S . -B build -DBUILD_TESTING=ON >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === BUILD THE AFFECTED HOST TESTS (Debug) ===
cmake --build build --config Debug -j 8 --target infernix_metrics_contract_test infernix_metrics_test infernix_openai_images_schema_test infernix_image_generation_service_test >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === RUN THEM ===
for %%T in (infernix_metrics_contract_test infernix_metrics_test infernix_openai_images_schema_test infernix_image_generation_service_test) do (
    "%BIN%\%%T.exe" > "image-spike\_pr36_%%T.out" 2>&1
    echo RESULT %%T exit=!errorlevel!
)
echo PR36_DONE
exit /b 0

:fail
echo PR36_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
