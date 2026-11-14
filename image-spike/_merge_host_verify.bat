@echo off
set LOG=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_merge_host_verify.log
set REPO=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix

REM The merged tree, verified with the host tests that cover the files the merge touched. No app target and
REM no CUDA test target, so nothing here needs the GPU and nothing links infernix-serve.exe: the resident
REM server on :8888 is untouched. Absolute LOG everywhere -- an earlier revision of this file did "cd build"
REM and then redirected to a RELATIVE log, so ctest never ran and its output went nowhere. vcvars64.bat is
REM not called: this run has no vswhere, so it left VCPKG_ROOT on the VS-embedded tree (no ffmpeg) and the
REM configure step died on it. cmake finds its generator from the cache.
cd /d "%REPO%" || goto :fail
if not exist "%REPO%\build\CMakeCache.txt" goto :fail
set "PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin"

echo === BUILD the four host targets === > "%LOG%"
cmake --build build --config Release -j 8 --target infernix_openai_images_schema_test infernix_image_generation_service_test infernix_chat_image_tool_test infernix_http_routes_test >> "%LOG%" 2>&1
if %errorlevel% neq 0 goto :fail
findstr /c:"Configuring incomplete" "%LOG%" >nul && goto :fail
findstr /c:"CMake Error" "%LOG%" >nul && goto :fail

echo === CTEST the same four === >> "%LOG%"
ctest --test-dir build -C Release -R "^(infernix_openai_images_schema_test|infernix_image_generation_service_test|infernix_chat_image_tool_test|infernix_http_routes_test)$" --output-on-failure >> "%LOG%" 2>&1
if %errorlevel% neq 0 goto :fail
findstr /c:"100%% tests passed, 0 tests failed out of 4" "%LOG%" >nul || goto :fail

echo MERGE_HOST_VERIFY_OK
exit /b 0

:fail
echo MERGE_HOST_VERIFY_FAILED
findstr /i /c:"error" /c:"Failed" /c:"Configuring incomplete" "%LOG%"
exit /b 1
