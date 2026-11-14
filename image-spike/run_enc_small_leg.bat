@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Debug"
set INFERNIX_TEST_ARTIFACT=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v2.infernix
set INFERNIX_REF_ENCODE_ORACLE=C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out
set INFERNIX_REF_ENCODE_LEG=oracle_ref_encode_small
set INFERNIX_VAE_STAGE_STATS=1
echo CWD=%CD%
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\tests\Debug\infernix_image_vae_encode_prod_cuda_test.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\enc_small_leg.log" 2>&1
if errorlevel 1 (echo LEG_FAIL) else (echo LEG_OK)
