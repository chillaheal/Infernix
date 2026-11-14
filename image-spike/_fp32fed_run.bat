@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
set TP=9
"C:\Users\Micke\AppData\Local\Programs\Python\Python313\python.exe" -I "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\image-edit\dump_text_fp32fed_probe.py" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\text_fp32fed.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\text_fp32fed.err.log"
set TP=%errorlevel%
>>"C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\text_fp32fed.out.log" echo TESTEXIT=%TP%
