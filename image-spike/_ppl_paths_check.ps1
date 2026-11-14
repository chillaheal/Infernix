$paths = @(
    'C:\vcpkg\installed\x64-windows\bin',
    'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat',
    'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\image-spike\window_gate.ps1',
    'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\build\apps\Release\infernix-perplexity.exe',
    'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix\eval\corpora\perplexity-1m\manifest.json',
    'C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix-00001-of-00003.infernix',
    'C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-NVIDIA-FP4-Local-Infernix.ngram',
    'C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix',
    'C:\Users\Micke\Desktop\Infernix FlashNext\models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram'
)
foreach ($p in $paths) {
    Write-Output (('{0,-6}' -f (Test-Path -LiteralPath $p)) + $p)
}
