@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next"
set "PYTHONUNBUFFERED=1"
set TR=9
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\ref_edit_behavior.py" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.err.log"
set TR=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\ref_behave_blackout.out.log" echo TESTEXIT=%TR%
