@echo off
cd /d "C:\Users\Micke\Desktop\Infernix FlashNext"
set ARTDIR=models
set ART=
for /f "delims=" %%f in ('dir /b /a-d "%ARTDIR%\*-00001-of-*.infernix" 2^>nul') do set ART=%ARTDIR%\%%f
if not defined ART (
  echo Artifact not found in %ARTDIR% - run tools.convert first.
  exit /b 1
)
infernix-serve.exe "%ART%" --ngram-volume "%ARTDIR%\Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram" --chat-template "C:\Users\Micke\Desktop\Infernix FlashNext\qwen3_8.jinja" --host 127.0.0.1 --port 8888 --max-context 384000 --rope-yarn-factor 1.5 --kv-capacity auto --kv-dtype nvfp4 --default-max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --frequency-penalty 0.0 --default-thinking-budget 8192 --thinking-budget-policy clamp --vision --vision-offload on --spec dflash2 --draft-model "C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-DFlash2-Infernix.infernix" --draft-tokens auto --lm-head-draft --preserve-thinking --request-log-jsonl requests.jsonl --max-concurrency 2 --thinking-budget-message "Considering the limited time available to the user, I must stop thinking now. Time to act:" --expert-a4-activations --prefill-chunk 4096 --log-level info --prefix-cache-file "C:\Users\Micke\Desktop\Infernix FlashNext\prefix-cache-flash-next.bin" --expert-ram-mib 46000 --ngram-draft-tokens 15 --ngram-min-match 11 --log-stats-panel on --host-context-mib 4096 --image-artifact "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\out\Qwen-Image-2.1-FP8-v4.infernix" --image-text-processor "C:\Users\Micke\Desktop\Infernix FlashNext\models\image-processor" > "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\serve_swap_server.out.log" 2> "C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\serve_swap_server.err.log"
rem --expert-a4-activations is ON: routed experts run with 4-bit NVFP4 activations (W4A4) at the
rem   activation scales NVIDIA calibrated. Measured on the 1M corpus (4K ctx, BF16 KV): ppl 3.4865
rem   -> 3.5024, KL vs the official FP8 0.042 -> 0.057; tg256 110 -> 135 plain, 150-162 -> 216 with
rem   MTP. Delete the flag above for the 16-bit default NVIDIA serves the checkpoint with.
echo EXIT=%ERRORLEVEL%
