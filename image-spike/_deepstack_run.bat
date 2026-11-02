@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next"
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\deepstack_ablate.py" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\deepstack_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\deepstack_blackout.err.log"
set TE=0
if errorlevel 1 set TE=1
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\deepstack_blackout.out.log" echo TESTEXIT=%TE%
