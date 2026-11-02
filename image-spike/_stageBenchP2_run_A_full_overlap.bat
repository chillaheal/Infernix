@echo off
cd /d "C:\Users\Micke\Desktop\Infernix FlashNext"
"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\bench\Release\infernix_image_stage_bench.exe" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\stage_bench_A_full_overlap.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\stage_bench_A_full_overlap.err.log"
set TE=0
if errorlevel 1 set TE=1
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\stage_bench_A_full_overlap.out.log" echo BENCHEXIT=%TE%
