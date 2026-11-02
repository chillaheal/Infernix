@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\_serve_build.log

REM The two app executables the user ships in the desktop folder, built from the merged tree (23091f5).
REM Only the two app targets, no CMakeLists touched so no reconfigure, and nothing here touches the
REM resident server on :8888: the exe is linked into build\apps\Release, the desktop copy is swapped
REM separately inside the blackout window. VCPKG_ROOT is set explicitly and vcvars64.bat is NOT called:
REM this host has no vswhere, so calling it left VCPKG_ROOT on the VS-embedded tree (no ffmpeg) and died
REM the configure step.
echo === BUILD infernix + infernix-serve (merged tree) === > "%LOG%"
cmake --build build --config Release -j 8 --target infernix infernix-serve >> "%LOG%" 2>&1
if %errorlevel% neq 0 goto :fail
findstr /c:"CMake Error" "%LOG%" >nul && goto :fail
findstr /i /c:"error C" /c:"LNK1" /c:"fatal error" "%LOG%" >nul && goto :fail
echo SERVE_BUILD_DONE
exit /b 0

:fail
echo SERVE_BUILD_FAILED
findstr /i /c:"error" /c:"CMake Error" "%LOG%" | more +0
exit /b 1
