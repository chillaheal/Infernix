@echo off
setlocal
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set VCPKG_ROOT=C:\vcpkg
set VCPKG_TARGET_TRIPLET=x64-windows
set PATH=C:\vcpkg\installed\x64-windows\debug\bin;%PATH%
set LOG=image-spike\_rebase_cert2.log

echo === BUILD DEBUG CERT TARGETS (log %LOG%) ===
cmake --build build --config Debug -j 8 --target infernix_qwen4_exp_memory_plan_test infernix_openai_images_schema_test infernix_image_generation_service_test infernix_qwen4_exp_image_yield_real_test infernix_qwen4_exp_image_interlude_op_real_test infernix_qwen4_exp_image_interlude_failures_real_test infernix_qwen4_exp_image_queue_deadline_real_test infernix_qwen4_exp_image_interlude_prod_real_test infernix_qwen4_exp_expert_fault_real_test > %LOG% 2>&1
if errorlevel 1 goto :fail

echo === RUN HOST CERT TESTS ===
ctest --test-dir build -C Debug --output-on-failure -R "infernix_qwen4_exp_memory_plan_test|infernix_openai_images_schema_test|infernix_image_generation_service_test" > image-spike\_rebase_cert2_tests.log 2>&1
if errorlevel 1 goto :fail

echo CERT2_DONE
exit /b 0

:fail
echo CERT2_FAILED
echo --- error lines ---
findstr /i /c:"error" %LOG%
echo --- test error lines ---
findstr /i /c:"error" image-spike\_rebase_cert2_tests.log
exit /b 1
