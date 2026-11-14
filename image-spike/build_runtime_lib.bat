@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --target infernix_qwen_image_runtime -j 6 > out\runtime_lib_build.log 2>&1
if errorlevel 1 (echo BUILD_FAIL) else (echo BUILD_OK)
