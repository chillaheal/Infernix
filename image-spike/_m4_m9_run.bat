@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin"
set "INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix"
set "INFERNIX_IMAGE_ORACLE=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle"
set "INFERNIX_IMAGE_ORACLE_F32FED=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle_fp32fed_tower"
set "INFERNIX_IMAGE_ORACLE_VISION_F32=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle_fp32_vision"
set "INFERNIX_REF_ENCODE_ORACLE=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\oracles"
set "INFERNIX_VISION_STAGE_DIR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_stages_engine"
set "INFERNIX_QWENVL_PROCESSOR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor"
set TE=9
set TB=9
set TP=9
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\dump_text_fp32fed_probe.py" portrait > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\fp32fed_portrait_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\fp32fed_portrait_blackout.err.log"
set TP=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\fp32fed_portrait_blackout.out.log" echo TESTEXIT=%TP%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_prompt_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.err.log"
set TE=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_behavior_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_behavior_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_behavior_blackout.err.log"
set TB=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.out.log" echo TESTEXIT=%TE%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_behavior_blackout.out.log" echo TESTEXIT=%TB%
set TS1=9
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_vae_encode_prod_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vae_encode_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vae_encode_blackout.err.log"
set TS1=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vae_encode_blackout.out.log" echo TESTEXIT=%TS1%
set TS2=9
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_need_ledger_real_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ledger_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ledger_blackout.err.log"
set TS2=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ledger_blackout.out.log" echo TESTEXIT=%TS2%
set TS3=9
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_pipeline_prod_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\pipeline_textonly_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\pipeline_textonly_blackout.err.log"
set TS3=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\pipeline_textonly_blackout.out.log" echo TESTEXIT=%TS3%
