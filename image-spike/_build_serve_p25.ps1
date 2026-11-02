# Rebuild of ONE target after the auto-skip fix: the serve executable itself. Narrow on purpose -- the
# rule this session lives by is that no full `cmake --build build` runs while :8888 answers, and a
# single target rebuilds only infernix_serve and the exe that links it. The running process is the
# Desktop copy, which no build writes, so this relink is free to happen beside a live serve.
# Env + vcvars block copied verbatim from _build_bench_p24.ps1 (which is copied from
# _runtests_chatimage.ps1, proven this session).
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
if (-not (Test-Path -LiteralPath $root)) { Write-Output "MISSING $root"; exit 2 }
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path -LiteralPath $vcvars)) { Write-Output "MISSING $vcvars"; exit 2 }
Set-Location -LiteralPath $root
$env:PATH = $env:PATH + ';C:\vcpkg\installed\x64-windows\bin'
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
& cmd.exe /c "`"$vcvars`" >nul && cmake --build build --config Release --target infernix-serve -j"
Write-Output "BUILD_EXIT=$LASTEXITCODE"
$exe = Join-Path $root 'build\apps\Release\infernix-serve.exe'
if (Test-Path -LiteralPath $exe) {
    Write-Output ("exe " + (Get-Item $exe).LastWriteTime + " " + (Get-Item $exe).Length + " bytes")
} else { Write-Output 'EXE_ABSENT' }
Write-Output 'SERVE_BUILD_DONE'
