@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --target infernix_openai_images_schema_test -j 6 > out\schema_test_build.log 2>&1
if errorlevel 1 (echo SCHEMA_BUILD_FAIL & exit /b 1)
build\tests\Debug\infernix_openai_images_schema_test.exe
if errorlevel 1 (echo SCHEMA_TEST_FAIL) else (echo SCHEMA_TEST_DONE)
