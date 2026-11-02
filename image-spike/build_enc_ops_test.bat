@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --target infernix_image_vae_encode_ops_test -j 6 > out\enc_ops_build.log 2>&1
if errorlevel 1 (echo BUILD_FAIL) else (echo BUILD_OK)
