@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --target infernix_image_pipeline_gate_test -j 6 > out\gate_build.log 2>&1
if errorlevel 1 (echo BUILD_FAIL & exit /b 1)
set INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v2.infernix
build\tests\Debug\infernix_image_pipeline_gate_test.exe
if errorlevel 1 (echo GATE_FAIL) else (echo GATE_EXIT=%errorlevel%)
