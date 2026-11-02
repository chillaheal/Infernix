@echo off
title INFERNIX 8888 PAUSAS - blackout_edit_prompt.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\window_shield.ps1" -Port 8888 -LockFile "C:\Users\Micke\Desktop\Infernix FlashNext\infernix-window.lock" -MarkerFile "C:\Users\Micke\Desktop\Infernix FlashNext\infernix-window.json" -Reason "blackout_edit_prompt.ps1" -MaxMinutes 90
