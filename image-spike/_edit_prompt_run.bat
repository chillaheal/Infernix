@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin"
set "INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix"
set "INFERNIX_IMAGE_ORACLE=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle"
set "INFERNIX_VISION_STAGE_DIR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_stages_engine"
set "INFERNIX_QWENVL_PROCESSOR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor"
set TD=9
set TE1=9
set TE2=9
set TE3=9
set TL=9
set TB=9
set TR=9
call "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\run_dump_edit_step_gpu.bat" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_step_dump.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_step_dump.err.log"
set TD=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_vision_prod_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_blackout12.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_blackout12.err.log"
set TE1=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_prompt_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.err.log"
set TE2=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_pipeline_prod_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_pipeline_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_pipeline_blackout.err.log"
set TE3=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_need_ledger_real_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ledger_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ledger_blackout.err.log"
set TL=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_behavior_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\behavior_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\behavior_blackout.err.log"
set TB=%errorlevel%
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\ref_edit_behavior.py" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.err.log"
set TR=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_step_dump.out.log" echo TESTEXIT=%TD%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_blackout12.out.log" echo TESTEXIT=%TE1%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.out.log" echo TESTEXIT=%TE2%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_pipeline_blackout.out.log" echo TESTEXIT=%TE3%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ledger_blackout.out.log" echo TESTEXIT=%TL%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\behavior_blackout.out.log" echo TESTEXIT=%TB%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.out.log" echo TESTEXIT=%TR%
