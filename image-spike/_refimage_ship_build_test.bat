@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_refimage_ship_build_test.log
set BIN=build\tests\Release
set "PATH=C:\vcpkg\installed\x64-windows\bin;%PATH%"

REM The reference_image capability bit lives in src/serve/http_server.cpp, and the test that reads the
REM capability payload is infernix_http_routes_test. Release host test only: nothing here links
REM infernix-serve.exe, so :8888 stays up and the resident LLM is untouched. No new files, so no
REM reconfigure.
echo === BUILD === > %LOG%
cmake --build build --config Release -j 8 --target infernix_http_routes_test >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === RUN ===
"%BIN%\infernix_http_routes_test.exe" > "image-spike\_refimage_ship_http_routes.out" 2>&1
echo RESULT http_routes exit=!errorlevel!
type "image-spike\_refimage_ship_http_routes.out"
echo SHIPBIT_DONE
exit /b 0

:fail
echo SHIPBIT_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
