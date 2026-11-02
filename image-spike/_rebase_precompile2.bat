@echo off
setlocal
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_rebase_precompile2.log
set SERVE_EXE=build\apps\Release\infernix-serve.exe

REM infernix-serve links ONLY these three libraries (apps/CMakeLists.txt:41), and they are static
REM libraries, so building them cannot relink the running exe. This is what actually shortens the
REM window: inside it, only the final link of infernix-serve.exe should be left.
echo === BUILD THE THREE LIBRARIES infernix-serve LINKS (Release) ===
echo before:
for %%F in ("%SERVE_EXE%") do echo   size=%%~zF mtime=%%~tF
cmake --build build --config Release -j 8 --target infernix_serve infernix_product_config infernix_product_logging > %LOG% 2>&1
if errorlevel 1 goto :fail

echo after:
for %%F in ("%SERVE_EXE%") do echo   size=%%~zF mtime=%%~tF
findstr /c:"infernix-serve.exe" %LOG% >nul && echo WARNING_SERVE_EXE_MENTIONED_IN_LOG || echo SERVE_EXE_NOT_MENTIONED
echo PRECOMPILE2_DONE
exit /b 0

:fail
echo PRECOMPILE2_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
