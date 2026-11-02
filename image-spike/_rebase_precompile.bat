@echo off
setlocal
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_rebase_precompile.log
set SERVE_EXE=build\apps\Release\infernix-serve.exe

echo === PRECOMPILE ONLY (no link): infernix-serve and its dependencies ===
echo The serve exe before:
for %%F in ("%SERVE_EXE%") do echo   %%~nxF size=%%~zF mtime=%%~tF
echo This step must NOT link infernix-serve.exe while :8888 is live. Log: %LOG%

cmake --build build --config Release -j 8 --target infernix-serve -- -t:ClCompile > %LOG% 2>&1
if errorlevel 1 goto :fail

echo === exe after (must be the same size and mtime as above) ===
for %%F in ("%SERVE_EXE%") do echo   %%~nxF size=%%~zF mtime=%%~tF
findstr /c:"linking" /c:"-> infernix-serve.exe" %LOG% | find /c /v "" | findstr /r "^[1-9]" >nul && echo WARNING_LINK_LINE_IN_LOG || echo NO_LINK_OF_SERVE_EXE
echo PRECOMPILE_DONE
exit /b 0

:fail
echo PRECOMPILE_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
