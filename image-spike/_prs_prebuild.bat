@echo off
setlocal EnableDelayedExpansion
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set LOG=image-spike\_prs_prebuild.log
set TRACE=build\tests\Debug\infernix_qwen4_exp_route_trace_real_test.exe
set OLD=image-spike\route_trace_pre_prs_2026-10-23.exe

REM The route-trace exe from 16:43 is post-rebase but PRE the three merged PRs, so it is the literal
REM "before" for the exact route across #36/#28/#35. Save it before anything rebuilds it.
echo === SAVE THE PRE-PR ROUTE TRACE EXE ===
if not exist "%OLD%" copy /y "%TRACE%" "%OLD%" >nul && echo OLD_SAVED
for %%F in ("%OLD%") do echo   fore-PR exen: size=%%~zF mtime=%%~tF

REM Both trees need the merged headers (types.h gained ChatTemplateRoleSupport): the Release libraries
REM serve links, and the Debug cert exes. Nothing here links infernix-serve.exe, so :8888 keeps serving.
echo === RELEASE LIBRARIES === > %LOG%
cmake --build build --config Release -j 8 --target infernix_serve infernix_product_config infernix_product_logging >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo === DEBUG CERT AND TRACE TARGETS === >> %LOG%
cmake --build build --config Debug -j 8 --target infernix_qwen4_exp_image_yield_real_test infernix_qwen4_exp_image_interlude_op_real_test infernix_qwen4_exp_image_interlude_failures_real_test infernix_qwen4_exp_image_interlude_prod_real_test infernix_qwen4_exp_image_interlude_dit_real_test infernix_qwen4_exp_image_interlude_full_real_test infernix_qwen4_exp_image_queue_deadline_real_test infernix_qwen4_exp_expert_fault_real_test infernix_qwen4_exp_route_trace_real_test >> %LOG% 2>&1
if errorlevel 1 goto :fail

echo after:
for %%F in ("%TRACE%") do echo   efter-PR exen: size=%%~zF mtime=%%~tF
for %%F in ("build\apps\Release\infernix-serve.exe") do echo   serve exe: size=%%~zF mtime=%%~tF
findstr /c:"infernix-serve.exe" %LOG% >nul && echo WARNING_SERVE_EXE_MENTIONED_IN_LOG || echo SERVE_EXE_NOT_MENTIONED
echo PRS_PREBUILD_DONE
exit /b 0

:fail
echo PRS_PREBUILD_FAILED
findstr /i /c:"error" %LOG%
exit /b 1
