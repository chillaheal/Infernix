@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin"
set "INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix"
set "INFERNIX_IMAGE_ORACLE=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle"
set "INFERNIX_VISION_STAGE_DIR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_stages_engine"
set "INFERNIX_QWENVL_PROCESSOR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor"
set "INFERNIX_TEXT_CURVE=1"
set TC=9
set TE=9
set TR=9
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\dump_edit_text_oracle.py" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\text_curve_dump.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\text_curve_dump.err.log"
set TC=%errorlevel%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_prompt_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.err.log"
set TE=%errorlevel%
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\ref_edit_behavior.py" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.err.log"
set TR=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\text_curve_dump.out.log" echo TESTEXIT=%TC%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\edit_prompt_blackout.out.log" echo TESTEXIT=%TE%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.out.log" echo TESTEXIT=%TR%
