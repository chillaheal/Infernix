@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin"
set "INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix"
set "INFERNIX_QWENVL_PROCESSOR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-spike\artifact\processor"
set "INFERNIX_EDIT_PORTRAIT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\evidence\source_portrait.png"
set "INFERNIX_EDIT_OUTDIR=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\portrait_look"
set "INFERNIX_EDIT_PROMPT=Change the background to solid green. Keep the woman and her navy blazer unchanged."
set TB=9
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Release\infernix_image_edit_behavior_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\portrait_look_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\portrait_look_blackout.err.log"
set TB=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\portrait_look_blackout.out.log" echo TESTEXIT=%TB%
