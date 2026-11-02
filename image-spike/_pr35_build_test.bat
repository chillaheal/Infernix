@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_pr35_build_test.log
set BIN=build\tests\Debug
set "PATH=C:\vcpkg\installed\x64-windows\bin;C:\vcpkg\installed\x64-windows\debug\bin;%PATH%"

REM Debug host tests only - nothing here links infernix-serve.exe, so :8888 keeps serving.
echo === RECONFIGURE === > %LOG%
cmake -S . -B build -DBUILD_TESTING=ON >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === BUILD THE HOST TESTS PR 35 TOUCHES (Debug) ===
cmake --build build --config Debug -j 8 --target infernix_qwen3_5_frontend_test infernix_request_log_test infernix_openai_schema_test infernix_anthropic_schema_test infernix_engine >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === RUN THEM ===
for %%T in (infernix_qwen3_5_frontend_test infernix_request_log_test infernix_openai_schema_test infernix_anthropic_schema_test) do (
    "%BIN%\%%T.exe" > "image-spike\_pr35_%%T.out" 2>&1
    echo RESULT %%T exit=!errorlevel!
)
echo === RUN THE PYTHON SIDE ===
python -m pytest tests/convert/test_resources.py tests/test_request_log_schema.py -q > image-spike\_pr35_pytest.out 2>&1
echo RESULT pytest exit=!errorlevel!
echo PR35_DONE
exit /b 0

:fail
echo PR35_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
