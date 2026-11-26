<#
.SYNOPSIS
  Configure-and-audit drift dry-run of the UPSTREAM Infernix tree, in a throwaway checkout with
  its own build directory. Host-only targets. Never card-running.

.DESCRIPTION
  What this is for
  ----------------
  Our image lane lives on top of a fork with no merge-base with upstream (see
  docs/maintainer/image-integration-points.md). Before committing to a port onto a new upstream
  ref, we need to know whether that ref still configures and still builds the host-side libraries
  on THIS machine, with THIS toolchain -- because a ref that cannot configure is a different, much
  more expensive kind of bad news than a ref whose seams moved. This script answers that question
  as a diagnostic. It is not a gate: it exits 0 when it completes, whatever it found, and exits 2
  only when it refuses to attempt a configure that is already doomed.

  Never, ever points at our own build tree
  ----------------------------------------
  Our checkout builds with the Visual Studio generator into
  "<repo>\build-windows" (image-edit\_p40_configure.bat:15-21). Upstream's own launcher
  (build_native.bat, f9c2e42c) configures with "Ninja Multi-Config" into that SAME directory name
  and carries this guard, verbatim at f9c2e42c:build_native.bat lines 38-42:

      :configure
      if exist build-windows\CMakeCache.txt if not exist build-windows\build.ninja (
        echo build-windows was configured with another generator: deleting it for the Ninja build
        rmdir /s /q build-windows
        if exist build-windows exit /b 1
      )

  That is a WRECKING BALL aimed at our build directory, and our tree is exactly the shape it
  triggers on: our build-windows has a CMakeCache.txt (written by the VS generator) and has NO
  build.ninja, so the guard's two conditions are both true and it would rmdir /s /q the whole
  tree. What that directory holds on this machine is not scratch: it is where the Release binaries
  we run are produced, including the infernix-serve.exe the live server was launched from, plus
  every object file and the CUDA CMake state. Deleting it costs a full rebuild (the audit measured
  a full native build at about 12.6 minutes) and, if it happened while a serve was started from
  those outputs, it takes the deployed exe with it.

  Two consequences, both enforced below:
    1. This script configures upstream into ".upstream-dryrun\build-<ref>" OUTSIDE the repository,
       never into any directory named build-windows.
    2. Before the first cmake invocation, Assert-SafeDirs proves the resolved source dir and build
       dir are both under .upstream-dryrun and that neither is the repo's build-windows. If that
       assertion fails the script stops; a doomed configure is a bug in this file, not a finding.

  Why the checkout is throwaway
  -----------------------------
  Upstream is only reachable here as objects in our own object database (there is no merge-base and
  no upstream working tree). Two primitives are supported:
    -Method worktree : git worktree add --detach <ref> (a real git tree, so upstream's
                       cmake/GenerateBuildId.cmake records a real build id -- without a git tree it
                       only warns and stamps "build unknown", which is fine to build but hides the
                       identity we want in the log)
    -Method archive  : git archive <ref> | Expand-Archive (leaves .git entirely untouched, but the
                       tree has no git metadata, so the build id becomes "unknown")
  Both write only under .upstream-dryrun, and both are removed at the end unless -Keep is passed.

  Toolchain environment, copied verbatim
  --------------------------------------
  The vcvars/vcpkg/CUDA/PATH lines below are copied verbatim from the proven launcher in this
  workspace, image-edit\_p40_build_only.bat:11-19 (and the generator/flag lines from
  image-edit\_p40_configure.bat:15-21). They were not reconstructed. That file's own header records
  why the env has to live in a .bat and not in an inline "cmd /c" from PowerShell: inline nested
  quotes get mangled, vcvars64.bat then dies with "'vswhere.exe' is not recognized as an internal
  or external command" while the outer cmd line still exits 0, and the caller sees a green build
  that produced nothing. So this script generates two throwaway .bat files under
  .upstream-dryrun\scripts\ (never inside image-edit\, which is not ours to edit) and runs those.
  The only deliberate deviation from _p40_configure.bat is -S/-B: it must point at the throwaway
  checkout and the throwaway build dir, for the reason documented above.

  Host-only targets, and why nothing here touches the GPU
  -------------------------------------------------------
  Every target in the default list is an add_library(... STATIC ...) whose declared sources contain
  no .cu file (Audit-Targets re-derives that from the checked-out CMakeLists files and DROPS any
  target that fails it, with the reason printed). A static host library compiles with the host
  compiler only: no nvcc, no kernel, no CUDA context, no device memory. Nothing in the list is an
  executable, so there is no link against a CUDA runtime and no process that could open a card.
  Explicitly NOT done, and that is the point:
    - ctest is never run (many upstream test entries load real artifacts and take the device; the
      suite-wide scheduling block in upstream's tests/CMakeLists.txt marks them RUN_SERIAL "gpu;real")
    - infernix-serve, the launcher, the PE worker and the CLI are never started; only compiled if
      you add them to -Targets yourself, which you should not
    - no process is ever stopped, restarted, paused or killed except our own cmake child process if
      it exceeds the stage deadline
    - the live server on 127.0.0.1:8888 is never consulted. Nothing in this script issues an HTTP
      request and nothing judges health, so it is safe to run while the serve and the PE worker are
      up. The serve does not need to be paused, and the image-edit pause file is not touched.

  Pre-flight, before any configure is attempted
  ---------------------------------------------
  The known ways this machine has failed a foreign configure, each checked below and each an exit 2
  with the missing item printed rather than a 20-minute CMake failure:
    (a) ffmpeg: upstream's cmake/Dependencies.cmake calls find_package(FFMPEG REQUIRED) on Windows
        and resolves it from VCPKG_ROOT/VCPKG_TARGET_TRIPLET through cmake/FindFFMPEG.cmake. That
        module is byte-identical to upstream (blob e86ffff16ec391929d0f83a914ee739561d2da17) and has
        zero backslashes, so it is not the suspect; the real historical abort was av*.lib and the
        headers simply not being installed under the triplet tree, from a triplet mix-up (x6-windows
        instead of x64-windows). We therefore require avformat.lib, avcodec.lib, avutil.lib,
        swscale.lib plus include\libavformat\avformat.h and include\libavcodec\avcodec.h.
    (b) CUDA >= 13.4: upstream hard-gates it at f9c2e42c:CMakeLists.txt:21-27 with a FATAL_ERROR
        ("CUDA 13.2 miscompiles sm_120a code"), and it also pins CMAKE_CUDA_ARCHITECTURES=120a at
        :9-13. nvcc's own reported release is parsed; an older toolkit is a guaranteed configure
        failure, so it is reported before we start.
    (c) triplet is exactly x64-windows, matching the tree we just audited (and the same value the
        proven launcher sets).
  A missing cuda DLL directory or vcvars path is likewise reported before the run.

  Output contract
  ---------------
  One "DRYRUN_STAGE <name> <PASS|FAIL|SKIP> <detail>" line per step, logs kept under
  .upstream-dryrun\logs\, then a final "DRYRUN_FAILED_COUNT=<n>" and a one-line verdict. Read it as
  a report, not a merge blocker.

.PARAMETER Ref
  Upstream ref to dry-run. Default origin/main. Accepts any rev parseable by git in this repo.

.PARAMETER Method
  worktree (default) or archive. See above.

.PARAMETER SkipBuild
  Configure and audit only; do not compile anything.

.PARAMETER Keep
  Leave the throwaway checkout, build dir and worktree registration in place for inspection.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\image_upstream_dryrun.ps1 -Ref f9c2e42c
  Full dry-run: pre-flight, checkout, audit, configure, host-only builds, report.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File tools\image_upstream_dryrun.ps1 -SkipBuild
  Pre-flight and configure only.
#>
[CmdletBinding()]
param(
  [string]   $Ref    = 'origin/main',
  [ValidateSet('worktree','archive')] [string] $Method = 'worktree',
  [string]   $RepoRoot = 'C:\Users\Micke\Documents\Kodprojekt\Infernix - Flash next\Infernix',
  [string]   $Root     = 'C:\Users\Micke\Documents\Kodprojekt\Infernix - Flash next\.upstream-dryrun',
  [string[]] $Targets  = @('infernix_text','infernix_grammar','infernix_model_loading',
                           'infernix_artifact','infernix_product_config','infernix_product_logging',
                           'infernix_media_decode','infernix_media_pdf'),
  [int]      $Jobs = 8,
  [int]      $ConfigureTimeoutSec = 1800,
  [int]      $BuildTimeoutSec = 1200,
  [switch]   $SkipBuild,
  [switch]   $Keep
)

$ErrorActionPreference = 'Stop'
$script:Failed = 0
$script:Stages = New-Object System.Collections.ArrayList

# ---------------------------------------------------------------------------
# Toolchain constants, copied verbatim from image-edit\_p40_build_only.bat:11-16
# and image-edit\_p40_configure.bat:15-21. Do not "improve" these; the values
# are what this machine is actually provisioned with and what the proven
# launcher uses.
# ---------------------------------------------------------------------------
$VcvarsBat  = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$VcpkgRoot  = 'C:\vcpkg'
$Triplet    = 'x64-windows'
$CudaPath   = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4'
$MinCuda    = '13.4'
$Generator  = 'Visual Studio 18 2026'
$CudaArch   = '120a'

function Emit([string]$name, [string]$status, [string]$detail) {
  $null = $script:Stages.Add([pscustomobject]@{ Name = $name; Status = $status; Detail = $detail })
  Write-Host ("DRYRUN_STAGE {0} {1} {2}" -f $name.ToUpper().Replace(' ','_'), $status, $detail)
}
function Fail([string]$name, [string]$detail) {
  $script:Failed++
  Emit $name 'FAIL' $detail
}
function Info([string]$text) { Write-Host ("       {0}" -f $text) }

function Get-GitText([string[]]$gitArgs, [string]$cwd) {
  # argv as a list, never a shell string, so no quoting ambiguity around paths with spaces.
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  $out = & git -C $cwd @gitArgs 2>&1
  $code = $LASTEXITCODE
  $ErrorActionPreference = $prev
  return [pscustomobject]@{ Code = $code; Text = ($out | Out-String).TrimEnd() }
}

function Invoke-Batch([string]$batPath, [string]$logPath, [int]$timeoutSec) {
  # Runs the generated .bat in cmd.exe (see the header for why it must be a .bat), with our
  # own child pid tracked so a hang cannot wedge the diagnostic. Only this pid is ever stopped.
  if (Test-Path $logPath) { Remove-Item $logPath -Force }
  if (Test-Path ($logPath + '.err')) { Remove-Item ($logPath + '.err') -Force }
  $exitFile = $logPath + '.exit'
  if (Test-Path $exitFile) { Remove-Item $exitFile -Force }
  # A single argument string, passed to cmd verbatim: the bat path contains spaces, and the
  # documented form for that is  cmd /d /c ""<path>"  . An array here gets re-quoted by
  # Start-Process and cmd then fails on the outer quotes.
  $argLine = '/d /c ""{0}""' -f $batPath
  $proc = Start-Process -FilePath $env:ComSpec -ArgumentList $argLine `
                        -RedirectStandardOutput $logPath -RedirectStandardError ($logPath + '.err') `
                        -NoNewWindow -PassThru
  if (-not $proc.WaitForExit($timeoutSec * 1000)) {
    try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch { }
    return [pscustomobject]@{ Code = 124; Text = 'TIMEOUT'; TimedOut = $true }
  }
  $text = ''
  if (Test-Path $logPath) { $text += (Get-Content $logPath -Raw -ErrorAction SilentlyContinue) }
  $errFile = $logPath + '.err'
  if (Test-Path $errFile) { $text += (Get-Content $errFile -Raw -ErrorAction SilentlyContinue) }
  # The exit code is read from the sentinel the generated .bat writes itself, NOT from
  # $proc.ExitCode: with -NoNewWindow plus redirected streams, Start-Process -PassThru has been
  # observed to return a process object whose ExitCode is $null even after WaitForExit returned
  # true, and $null -ne 0 turned a SUCCESSFUL configure into a reported FAIL. The sentinel is
  # written by cmd's own %errorlevel%, so it is cmake's code and nothing else. $proc.ExitCode is
  # only a fallback, and an unknown status is reported as such (125) rather than guessed.
  $code = $null
  if (Test-Path $exitFile) {
    $raw = (Get-Content $exitFile -Raw -ErrorAction SilentlyContinue)
    if ($raw -and ([int]::TryParse($raw.Trim(), [ref]$null))) { $code = [int]$raw.Trim() }
  }
  if ($null -eq $code) {
    try { if ($null -ne $proc.ExitCode) { $code = $proc.ExitCode } } catch { }
  }
  if ($null -eq $code) { $code = 125 }
  return [pscustomobject]@{ Code = $code; Text = ($text -replace "`r",'').TrimEnd(); TimedOut = $false }
}

function Show-LogExcerpt([string]$text) {
  # Print the lines that name a real problem, then always the tail: a filtered-only view once
  # hid a successful configure behind two benign "not found" probe lines.
  $lines = @($text -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
  $bad = @($lines | Where-Object { $_ -match 'CMake Error|CMake Warning \(unused|FATAL|fatal error|FAILED|Cannot find|Could NOT find|error [A-Z][0-9]+' } |
                 Where-Object { $_ -notmatch 'Looking for|Performing Test' } | Select-Object -First 15)
  foreach ($t in $bad) { Info ("! {0}" -f $t) }
  Info '--- last 15 log lines ---'
  foreach ($t in ($lines | Select-Object -Last 15)) { Info ("  {0}" -f $t) }
}

function Write-BatchFile([string]$path, [string[]]$lines) {
  # ASCII-only, CRLF, no BOM: cmd.exe reads UTF-8-with-BOM batch files badly.
  $enc = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($path, (($lines -join "`r`n") + "`r`n"), $enc)
}

function Assert-SafeDirs([string]$srcDir, [string]$buildDir) {
  # The guard against the rmdir /s /q wrecking ball documented in the header.
  $rootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
  foreach ($candidate in @($srcDir, $buildDir)) {
    $full = [System.IO.Path]::GetFullPath($candidate)
    if (-not $full.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
      throw "refusing to use '$full': it is not under the throwaway root '$rootFull'"
    }
  }
  $forbidden = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot 'build-windows'))
  if (([System.IO.Path]::GetFullPath($buildDir)) -eq $forbidden) {
    throw "refusing to configure into the repo's own build dir '$forbidden'"
  }
  if ([System.IO.Path]::GetFileName($buildDir).ToLowerInvariant() -eq 'build-windows') {
    throw "refusing a build dir named 'build-windows' (upstream's build_native.bat:38-42 rmdir target)"
  }
}

# ---------------------------------------------------------------------------
# Stage 0: locate everything. Nothing here touches the GPU or the network.
# ---------------------------------------------------------------------------
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "== image upstream dry-run =="
Write-Host ("   started      : {0}" -f $stamp)
Write-Host ("   repo (ours)  : {0}" -f $RepoRoot)
Write-Host ("   throwaway    : {0}" -f $Root)
Write-Host ("   ref          : {0}" -f $Ref)
Write-Host ("   method       : {0}" -f $Method)
Write-Host "   live server  : not consulted (no HTTP to 127.0.0.1:8888, serve and PE worker untouched)"

foreach ($need in @($RepoRoot, $VcvarsBat, (Join-Path $VcpkgRoot 'installed'), $CudaPath)) {
  if (-not (Test-Path -LiteralPath $need)) {
    Fail 'preflight-path' ("missing required path: {0}" -f $need)
    Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
    Write-Host 'VERDICT: ABORTED - a required toolchain path does not exist; nothing was configured'
    exit 2
  }
}

$gitCheck = Get-GitText @('--version') (Split-Path -Parent $RepoRoot)
if ($gitCheck.Code -ne 0) {
  Fail 'preflight-git' 'git is not runnable'
  Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
  Write-Host 'VERDICT: ABORTED - git unavailable'
  exit 2
}
Emit 'preflight-git' 'PASS' $gitCheck.Text

$repoCheck = Get-GitText @('rev-parse','--is-inside-work-tree') $RepoRoot
if ($repoCheck.Code -ne 0 -or $repoCheck.Text -ne 'true') {
  Fail 'preflight-repo' ("not a git work tree: {0}" -f $RepoRoot)
  Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
  Write-Host 'VERDICT: ABORTED - repo root is not a git work tree'
  exit 2
}
Emit 'preflight-repo' 'PASS' $RepoRoot

$resolve = Get-GitText @('rev-parse','--verify','--quiet',($Ref + '^{commit}')) $RepoRoot
if ($resolve.Code -ne 0 -or -not $resolve.Text) {
  Fail 'preflight-ref' ("ref '{0}' is not present in this object database" -f $Ref)
  Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
  Write-Host 'VERDICT: ABORTED - upstream ref unresolvable (nothing was configured)'
  exit 2
}
$Sha = $resolve.Text.Trim()
$Short = $Sha.Substring(0,12)
Emit 'preflight-ref' 'PASS' ("{0} -> {1}" -f $Ref, $Short)

$null = New-Item -ItemType Directory -Force -Path $Root | Out-Null
$scriptsDir = Join-Path $Root 'scripts'
$logsDir = Join-Path $Root 'logs'
$null = New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null
$null = New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

# ---------------------------------------------------------------------------
# Stage 1: pre-flight. (a) ffmpeg, (b) CUDA >= 13.4, (c) triplet.
# Each failure prints exactly what is missing and exits 2 BEFORE any configure.
# ---------------------------------------------------------------------------
$tripletTree = Join-Path $VcpkgRoot ("installed\{0}" -f $Triplet)
if ((Split-Path -Leaf $Triplet) -ne 'x64-windows') {
  Fail 'preflight-c-triplet' ("triplet must be exactly 'x64-windows', got '{0}'" -f $Triplet)
} elseif (-not (Test-Path -LiteralPath $tripletTree)) {
  Fail 'preflight-c-triplet' ("triplet tree absent: {0}" -f $tripletTree)
} else {
  $strays = @(Get-ChildItem -LiteralPath (Join-Path $VcpkgRoot 'installed') -Directory |
              Where-Object { $_.Name -notmatch '^(x64-windows|vcpkg)$' } | ForEach-Object { $_.Name })
  $detail = "triplet tree present: $tripletTree"
  if ($strays.Count) { $detail = $detail + " (other installed trees: " + ($strays -join ', ') + ")" }
  Emit 'preflight-c-triplet' 'PASS' $detail
}

$ffmpegLibs = @('avformat.lib','avcodec.lib','avutil.lib','avdevice.lib','avfilter.lib','swscale.lib','swresample.lib')
$ffmpegHeads = @('libavformat\avformat.h','libavcodec\avcodec.h','libavutil\avutil.h','libswscale\swscale.h')
$missing = @()
foreach ($lib in $ffmpegLibs) {
  if (-not (Test-Path -LiteralPath (Join-Path $tripletTree ("lib\{0}" -f $lib)))) { $missing += ("lib\{0}" -f $lib) }
}
foreach ($head in $ffmpegHeads) {
  if (-not (Test-Path -LiteralPath (Join-Path $tripletTree ("include\{0}" -f $head)))) { $missing += ("include\{0}" -f $head) }
}
if ($missing.Count) {
  Fail 'preflight-a-ffmpeg' ("upstream find_package(FFMPEG REQUIRED) would abort; missing under {0}: {1}" -f $tripletTree, ($missing -join ', '))
  Write-Host '         fix: vcpkg install ffmpeg:x64-windows (the historical failure was a triplet'
  Write-Host '              typo, so the libs landed under a tree this build never searches)'
} else {
  Emit 'preflight-a-ffmpeg' 'PASS' ("all {0} import libs and {1} headers under {2}" -f $ffmpegLibs.Count, $ffmpegHeads.Count, $tripletTree)
}

$nvcc = Join-Path $CudaPath 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $nvcc)) {
  Fail 'preflight-b-cuda' ("nvcc not found: {0}" -f $nvcc)
} else {
  $prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
  $nvccOut = (& $nvcc --version 2>&1) -join "`n"
  $nvccCode = $LASTEXITCODE
  $ErrorActionPreference = $prev
  $match = [regex]::Match($nvccOut, 'release\s+(\d+)\.(\d+)')
  if ($nvccCode -ne 0 -or -not $match.Success) {
    Fail 'preflight-b-cuda' ("could not read nvcc version from '{0}': {1}" -f $nvcc, $nvccOut)
  } else {
    $found = [version]("{0}.{1}" -f $match.Groups[1].Value, $match.Groups[2].Value)
    $needed = [version]$MinCuda
    if ($found -lt $needed) {
      # Upstream's FATAL_ERROR at f9c2e42c:CMakeLists.txt:21-27 makes this a certain configure death.
      Fail 'preflight-b-cuda' ("CUDA {0} is older than upstream's hard minimum {1} (CMakeLists.txt:21-27)" -f $found, $MinCuda)
    } else {
      Emit 'preflight-b-cuda' 'PASS' ("CUDA {0} >= {1} required by upstream CMakeLists.txt:21-27" -f $found, $MinCuda)
    }
  }
}

if ($script:Failed -gt 0) {
  Write-Host ''
  Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
  Write-Host 'VERDICT: ABORTED IN PRE-FLIGHT - no checkout was created and no configure was attempted'
  exit 2
}
Emit 'preflight' 'PASS' 'all prerequisites present, configure is worth attempting'

# ---------------------------------------------------------------------------
# Stage 2: throwaway checkout. Our working tree and our build dir are never
# entered; only .upstream-dryrun and git's own worktree metadata are written.
# ---------------------------------------------------------------------------
$slug = ($Sha.Substring(0,8))
$srcDir = Join-Path $Root ("src-{0}" -f $slug)
$buildDir = Join-Path $Root ("build-{0}" -f $slug)
$wtRegistered = $false

try {
  if (Test-Path -LiteralPath $srcDir) {
    if ($Method -eq 'worktree') {
      $null = Get-GitText @('worktree','remove','--force',$srcDir) $RepoRoot
    }
    Remove-Item -LiteralPath $srcDir -Recurse -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path -LiteralPath $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force -ErrorAction SilentlyContinue
  }
  $null = New-Item -ItemType Directory -Force -Path $srcDir | Out-Null

  if ($Method -eq 'worktree') {
    $wt = Get-GitText @('worktree','add','--detach',$srcDir,$Sha) $RepoRoot
    if ($wt.Code -ne 0) {
      Fail 'checkout-worktree' ($wt.Text -replace "`r`n", ' | ')
      throw 'worktree creation failed'
    }
    $wtRegistered = $true
    Emit 'checkout-worktree' 'PASS' ("git worktree add --detach -> {0}" -f $srcDir)
  } else {
    $zip = Join-Path $Root ("archive-{0}.zip" -f $slug)
    if (Test-Path -LiteralPath $zip) { Remove-Item $zip -Force }
    $arch = Get-GitText @('archive',('--format=zip'),('--output=' + $zip),$Sha) $RepoRoot
    if ($arch.Code -ne 0) {
      Fail 'checkout-archive' ($arch.Text -replace "`r`n", ' | ')
      throw 'git archive failed'
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $srcDir -Force
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    Emit 'checkout-archive' 'PASS' ("git archive -> {0} (no .git here, so the build id will read 'unknown')" -f $srcDir)
  }

  $fileCount = @(Get-ChildItem -LiteralPath $srcDir -Recurse -File -ErrorAction SilentlyContinue).Count
  if ($fileCount -lt 50) {
    Fail 'checkout-contents' ("only {0} files in the checkout" -f $fileCount)
    throw 'checkout looks empty'
  }
  Emit 'checkout-contents' 'PASS' ("{0} files under {1}" -f $fileCount, $srcDir)
  Assert-SafeDirs $srcDir $buildDir
  Emit 'checkout-safety' 'PASS' ("build dir is outside the repo and not named build-windows: {0}" -f $buildDir)
} catch {
  Fail 'checkout' $_.Exception.Message
  Write-Host ''
  Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
  Write-Host 'VERDICT: ABORTED IN CHECKOUT - no configure was attempted'
  if ($Keep) { } elseif ($wtRegistered) {
    $null = Get-GitText @('worktree','remove','--force',$srcDir) $RepoRoot
  }
  exit 2
}

# ---------------------------------------------------------------------------
# Stage 3: audit the target list against the checked-out upstream files. A
# target is only allowed if it is an add_library(... STATIC ...) with no .cu in
# its declared sources -- that is the proof it needs no CUDA compiler and no
# card. Anything else is DROPPED with the reason printed, never silently.
# ---------------------------------------------------------------------------
$allowed = New-Object System.Collections.ArrayList
foreach ($target in $Targets) {
  $found = $null
  $files = @(Get-ChildItem -LiteralPath $srcDir -Recurse -Filter 'CMakeLists.txt' -File -ErrorAction SilentlyContinue)
  foreach ($f in $files) {
    $text = Get-Content -LiteralPath $f.FullName -Raw -ErrorAction SilentlyContinue
    if (-not $text) { continue }
    $m = [regex]::Match($text, ('(?is)add_library\(\s*{0}\b(.*?)\)\s*\r?\n' -f [regex]::Escape($target)))
    if ($m.Success) { $found = @{ File = $f.FullName; Block = $m.Groups[1].Value }; break }
  }
  if (-not $found) {
    Fail 'audit-targets' ("{0}: no add_library({0} ...) anywhere in the upstream checkout - dropped" -f $target)
    continue
  }
  $block = $found.Block
  $rel = $found.File.Substring($srcDir.Length).TrimStart('\')
  if ($block -notmatch '(?i)\bSTATIC\b') {
    Fail 'audit-targets' ("{0} in {1}: not declared STATIC - dropped (a shared/executable target would link the CUDA runtime)" -f $target, $rel)
    continue
  }
  if ($block -match '(?i)\.cu\b') {
    Fail 'audit-targets' ("{0} in {1}: declares a .cu source - dropped (would need nvcc and a card)" -f $target, $rel)
    continue
  }
  if ($block -match '(?i)\bCUDA\b') {
    Fail 'audit-targets' ("{0} in {1}: mentions CUDA in its own declaration - dropped (host-only list must be CUDA-free)" -f $target, $rel)
    continue
  }
  $cuOnDisk = @(Get-ChildItem -LiteralPath (Split-Path -Parent $found.File) -Recurse -Filter '*.cu' -File -ErrorAction SilentlyContinue)
  $null = $allowed.Add($target)
  Emit 'audit-targets' 'PASS' ("{0}: STATIC, no .cu in its declaration ({1}) - host compile only, no card" -f $target, $rel)
}
if ($allowed.Count -eq 0) {
  Fail 'audit-targets' 'no host-only target survived the audit'
  Write-Host ''
  Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
  Write-Host 'VERDICT: ABORTED IN AUDIT - no host-only target was safe to build, so nothing was configured or compiled'
  exit 2
}
Info ("building {0} of {1} requested targets: {2}" -f $allowed.Count, $Targets.Count, ($allowed -join ', '))

# ---------------------------------------------------------------------------
# Stage 4: configure. Generated .bat with the proven env, VS generator, and the
# throwaway -S/-B. This is the whole point of the run: does upstream still
# configure on this machine with this toolchain?
# ---------------------------------------------------------------------------
$cfgBat = Join-Path $scriptsDir ("configure-{0}.bat" -f $slug)
$cfgLog = Join-Path $logsDir ("configure-{0}.log" -f $slug)
$cfgLines = @(
  '@echo off',
  'setlocal',
  ('REM Env copied verbatim from image-edit\_p40_build_only.bat:11-16 and the cmake flags from'),
  ('REM image-edit\_p40_configure.bat:15-21. Only -S/-B deviate: throwaway dirs, never build-windows.'),
  ('call "{0}"' -f $VcvarsBat),
  'if errorlevel 1 (',
  ('  > "{0}" echo 1' -f ($cfgLog + '.exit')),
  '  exit /b 1',
  ')',
  ('set "VCPKG_ROOT={0}"' -f $VcpkgRoot),
  ('set "VCPKG_TARGET_TRIPLET={0}"' -f $Triplet),
  ('set "CUDA_PATH={0}"' -f $CudaPath),
  ('set "PATH=%PATH%;{0}\installed\{1}\bin"' -f $VcpkgRoot, $Triplet),
  ('cd /d "{0}"' -f $srcDir),
  'if errorlevel 1 (',
  ('  > "{0}" echo 2' -f ($cfgLog + '.exit')),
  '  exit /b 2',
  ')',
  ('cmake -S . -B "{0}" -G "{1}" -A x64 ^' -f $buildDir, $Generator),
  ('  -T "cuda=%CUDA_PATH%" ^'),
  ('  -DVCPKG_TARGET_TRIPLET={0} ^' -f $Triplet),
  ('  -DCMAKE_PREFIX_PATH="{0}\installed\{1}" ^' -f $VcpkgRoot, $Triplet),
  ('  -DCMAKE_CUDA_ARCHITECTURES={0} ^' -f $CudaArch),
  '  -DINFERNIX_BUILD_APPS=ON ^',
  '  -DBUILD_TESTING=ON',
  ('> "{0}" echo %errorlevel%' -f ($cfgLog + '.exit')),
  'exit /b %errorlevel%'
)
Write-BatchFile $cfgBat $cfgLines
Emit 'configure-batch' 'PASS' ("generated {0}" -f $cfgBat)

$cfg = Invoke-Batch $cfgBat $cfgLog $ConfigureTimeoutSec
$cfgLogText = Get-Content $cfgLog -Raw -ErrorAction SilentlyContinue
$configured = ($cfg.Code -eq 0) -and ($cfgLogText -match 'Generating done')
if ($configured) {
  Emit 'configure' 'PASS' ("cmake configured {0} (log: {1})" -f $buildDir, $cfgLog)
  $cfgTail = ($cfg.Text -split "`n" | Where-Object { $_ -match 'Configuring done|Generating done|Build files have been written|CMake Warning' } | Select-Object -Last 8)
  foreach ($t in $cfgTail) { Info $t.Trim() }
} else {
  Fail 'configure' ("cmake exited {0} (log: {1})" -f $cfg.Code, $cfgLog)
  Show-LogExcerpt $cfg.Text
}

# ---------------------------------------------------------------------------
# Stage 5: build, one host-only target at a time. Never ctest, never an app,
# never anything that could open a CUDA context.
# ---------------------------------------------------------------------------
if ($SkipBuild) {
  Emit 'build' 'SKIP' '-SkipBuild was given (configure-and-audit only)'
} elseif (-not $configured) {
  Emit 'build' 'SKIP' 'configure failed, so there is no build system to compile against'
} else {
  foreach ($target in $allowed) {
    $bat = Join-Path $scriptsDir ("build-{0}-{1}.bat" -f $slug, $target)
    $log = Join-Path $logsDir ("build-{0}-{1}.log" -f $slug, $target)
    $lines = @(
      '@echo off',
      'setlocal',
      ('REM Copied verbatim from image-edit\_p40_build_only.bat:11-19; the build dir is the'),
      ('REM throwaway one under .upstream-dryrun, never the repo build-windows.'),
      ('call "{0}"' -f $VcvarsBat),
      'if errorlevel 1 (',
      ('  > "{0}" echo 1' -f ($log + '.exit')),
      '  exit /b 1',
      ')',
      ('set "VCPKG_ROOT={0}"' -f $VcpkgRoot),
      ('set "VCPKG_TARGET_TRIPLET={0}"' -f $Triplet),
      ('set "CUDA_PATH={0}"' -f $CudaPath),
      ('set "PATH=%PATH%;{0}\installed\{1}\bin"' -f $VcpkgRoot, $Triplet),
      ('cd /d "{0}"' -f $srcDir),
      'if errorlevel 1 (',
      ('  > "{0}" echo 2' -f ($log + '.exit')),
      '  exit /b 2',
      ')',
      ('cmake --build "{0}" --config Release --target {1} -j {2}' -f $buildDir, $target, $Jobs),
      ('> "{0}" echo %errorlevel%' -f ($log + '.exit')),
      'exit /b %errorlevel%'
    )
    Write-BatchFile $bat $lines
    $r = Invoke-Batch $bat $log $BuildTimeoutSec
    # An exit code of 0 is not the intended behaviour; the artifact has to exist. A target that
    # "built" without producing its library is reported as a failure, not a pass.
    $lib = @(Get-ChildItem -LiteralPath $buildDir -Recurse -Filter ($target + '.lib') -File -ErrorAction SilentlyContinue)
    if ($r.Code -eq 0 -and $lib.Count -gt 0) {
      $sizeKB = [math]::Round(($lib[0].Length / 1KB), 1)
      Emit ("build-{0}" -f $target) 'PASS' ("{0} built: {1} ({2} KB), log {3}" -f $target, $lib[0].FullName, $sizeKB, $log)
    } elseif ($r.Code -eq 0) {
      Fail ("build-{0}" -f $target) ("cmake --build exited 0 but produced no {0}.lib under {1} (log: {2})" -f $target, $buildDir, $log)
      Show-LogExcerpt $r.Text
    } else {
      Fail ("build-{0}" -f $target) ("cmake --build --target {0} exited {1} (log: {2})" -f $target, $r.Code, $log)
      Show-LogExcerpt $r.Text
    }
  }
}

# ---------------------------------------------------------------------------
# Stage 6: cleanup and report.
# ---------------------------------------------------------------------------
if ($Keep) {
  $keepDetail = '-Keep: left in place: ' + $srcDir + ' ; ' + $buildDir
  if ($wtRegistered) { $keepDetail = $keepDetail + ' ; worktree still registered' }
  Emit 'cleanup' 'SKIP' $keepDetail
} else {
  if ($wtRegistered) {
    $null = Get-GitText @('worktree','remove','--force',$srcDir) $RepoRoot
    $null = Get-GitText @('worktree','prune') $RepoRoot
  }
  foreach ($d in @($srcDir, $buildDir)) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction SilentlyContinue }
  }
  $left = @(Get-ChildItem -LiteralPath $Root -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
  Emit 'cleanup' 'PASS' ("checkout and build dir removed; kept logs and scripts (dirs left: {0})" -f ($left -join ', '))
}

Write-Host ''
Write-Host '---- stage summary ----'
foreach ($s in $script:Stages) {
  Write-Host ("  {0,-6} {1}" -f $s.Status, $s.Name)
}
Write-Host ''
Write-Host ("DRYRUN_FAILED_COUNT={0}" -f $script:Failed)
if ($script:Failed -eq 0) {
  Write-Host ("VERDICT: upstream {0} configures and builds the audited host-only targets on this machine - a seam failure from tools/check_image_seams.py is the only thing that would need re-derivation, not the build" -f $Short)
} else {
  Write-Host ("VERDICT: upstream {0} did NOT dry-run clean ({1} failing stage(s)) - read the logs above before budgeting a port; this is a diagnostic, not a gate" -f $Short, $script:Failed)
}
exit 0
