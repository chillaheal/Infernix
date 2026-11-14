@echo off
cd /d "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix"
echo queue-deadline cert window: waits for :8888 to go quiet, stops serve, runs the one cert exe, restores serve
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\image-spike\blackout_queue_deadline.ps1' -GateMaxWaitSec 900 -GateExpectedMinutes 6 -GraceSec 30 *>&1 | Tee-Object -FilePath '.\image-spike\_rebase_qdeadline_window.log'"
echo.
echo RUN DONE - log: image-spike\_rebase_qdeadline_window.log
pause
