@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_edit_wire_build.log

REM Build after wiring the edit route into serving: the serve library (image_generation_service.cpp/.h,
REM image_text_tower.h), the two app executables the user ships, and the host contract test that grades
REM the route's seams. No CMakeLists touched, so no reconfigure and no CUDA rebuild.
REM VCPKG_ROOT is set explicitly and vcvars64.bat is NOT called: this host has no vswhere, so calling it
REM leaves VCPKG_ROOT on the VS-embedded tree (no ffmpeg) and dies the configure step.
echo === BUILD serve lib + apps + image_generation_service test === > "%LOG%"
cmake --build build --config Release -j 8 --target infernix_serve infernix_image_generation_service_test infernix infernix-serve >> "%LOG%" 2>&1
if %errorlevel% neq 0 goto :fail
findstr /c:"CMake Error" "%LOG%" >nul && goto :fail
findstr /i /c:"error C" /c:"LNK1" /c:"fatal error" "%LOG%" >nul && goto :fail
echo EDIT_WIRE_BUILD_DONE
exit /b 0

:fail
echo EDIT_WIRE_BUILD_FAILED
findstr /i /c:"error" /c:"CMake Error" "%LOG%"
exit /b 1
