@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin"
set "INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix"
set "INFERNIX_IMAGE_ORACLE=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\oracle"
set "INFERNIX_VISION_STAGE_DIR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_stages_engine"
set TE=9
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_vision_prod_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_blackout.err.log"
set TE=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\vision_blackout.out.log" echo TESTEXIT=%TE%
