# Build the Release serve binary BEFORE the blackout window, so the server is down only for the copy,
# the start and the live gates. Writes a log; run in a visible window.
$ErrorActionPreference = 'Stop'
$repo     = 'C:\Users\Micke\Documents\Kodprojekt\Qwen Flash Next\Infernix'
$buildDir = Join-Path $repo 'build'
$buildExe = Join-Path $repo 'build\apps\Release\infernix-serve.exe'
$log      = Join-Path $repo 'image-spike\deploy_serve_build.log'
$cmake    = 'C:\Users\Micke\AppData\Local\Microsoft\WinGet\Packages\Kitware.CMake_Microsoft.Winget.Source_8wekyb3d8bbwe\cmake-4.4.3-windows-x86_64\bin\cmake.exe'

foreach ($p in @($buildDir, $cmake)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Output ('FATAL missing path: ' + $p); exit 2 }
}
$env:VCPKG_ROOT = 'C:\vcpkg'
$env:VCPKG_TARGET_TRIPLET = 'x64-windows'
& $cmake --build $buildDir --target infernix-serve --config Release -j 6 2>&1 | Tee-Object -FilePath $log | Select-Object -Last 8
Write-Output ('BUILD_RC=' + $LASTEXITCODE)
if (Test-Path -LiteralPath $buildExe) {
    $i = Get-Item -LiteralPath $buildExe
    Write-Output ('EXE size=' + $i.Length + ' mtime=' + $i.LastWriteTime)
} else {
    Write-Output ('EXE_MISSING ' + $buildExe)
}
