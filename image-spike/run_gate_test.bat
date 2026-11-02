@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v2.infernix
build\tests\Debug\infernix_image_pipeline_gate_test.exe
if errorlevel 1 (echo GATE_FAIL) else (echo GATE_DONE code=%errorlevel%)
