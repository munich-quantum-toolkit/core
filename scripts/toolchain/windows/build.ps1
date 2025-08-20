param(
  [Parameter(Mandatory=$true)][string]$Ref,
  [Parameter(Mandatory=$true)][string]$Prefix,
  [string]$Targets = "auto"
)
#
# Windows LLVM/MLIR optimized toolchain builder (PGO + ThinLTO)
#
# Description:
#   Builds a multi-stage optimized LLVM/MLIR toolchain on Windows using:
#   - Stage0: clang + compiler-rt (profile runtime)
#   - Stage1: instrumented build + lit tests to collect profiles
#   - Stage2: final PGO + ThinLTO toolchain
#   Uses ccache (LLVM_CCACHE_BUILD=ON) and emits a .tar.zst archive via tar | zstd.
#
# Usage:
#   scripts/toolchain/windows/build.ps1 -Ref <ref> -Prefix <install_prefix> [-Targets <targets>]
#     -Ref            Git ref/tag/commit (e.g., llvmorg-20.1.8 or SHA)
#     -Prefix         Install directory for the final toolchain
#     -Targets        LLVM_TARGETS_TO_BUILD (default: "X86;AArch64")
#
# Environment:
#   TOOLCHAIN_CLEAN=1           Clean before building
#   TOOLCHAIN_STAGE_FROM/TO     Limit stages (e.g., 2 and 2 for Stage2 only)
#   TOOLCHAIN_HOST_TRIPLE       Override computed host triple
#   TOOLCHAIN_CPU_FLAGS         Extra flags (e.g., -march=haswell)
#   LLVM_DIR/MLIR_DIR           Exported by the composite action consuming this script
#   PATH                        Updated by the composite action to include installed bin
#
# Outputs:
#   - Installs into -Prefix
#   - Creates llvm-mlir_<ref>_windows_<arch>_<targets>_opt.tar.zst next to the repo root
#
# Example:
#   pwsh -File scripts/toolchain/windows/build.ps1 -Ref llvmorg-20.1.8 -Prefix "$PWD/llvm-install" -Targets X86
#
$ErrorActionPreference = "Stop"

# Incremental controls
$stageFrom = [int](${env:TOOLCHAIN_STAGE_FROM} ?? 0)
$stageTo = [int](${env:TOOLCHAIN_STAGE_TO} ?? 2)

# Ensure directories (incremental unless TOOLCHAIN_CLEAN=1)
$work = Get-Location
$clean = $env:TOOLCHAIN_CLEAN
if ($clean -eq '1') {
  Remove-Item -Recurse -Force llvm-project, build_stage0, build_stage1, build_stage2, stage0-install, stage1-install, pgoprof -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force -Path (Join-Path $work 'pgoprof\raw') | Out-Null

# Host triple and CPU flags
$arch = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Architecture)
# 9 is x64, 12 is ARM64 per docs
if ($arch -eq 9) {
  $hostTripleComputed = 'x86_64-pc-windows-msvc'
  $hostTarget = 'X86'
} elseif ($arch -eq 12) {
  $hostTripleComputed = 'aarch64-pc-windows-msvc'
  $hostTarget = 'AArch64'
} else {
  $hostTripleComputed = 'x86_64-pc-windows-msvc'
  $hostTarget = 'X86'
}
$hostTriple = ${env:TOOLCHAIN_HOST_TRIPLE}
if (-not $hostTriple) { $hostTriple = $hostTripleComputed }
if (-not $Targets -or $Targets -eq 'auto') { $Targets = $hostTarget }
$cpuFlags = ${env:TOOLCHAIN_CPU_FLAGS}
if (-not $cpuFlags) { if ($arch -eq 9) { $cpuFlags = '-march=haswell -mtune=haswell' } else { $cpuFlags = '' } }

# Clone or update llvm-project
if (Test-Path 'llvm-project/.git') {
  Push-Location llvm-project
  git remote set-url origin https://github.com/llvm/llvm-project.git | Out-Null
  try { git fetch --depth 1 origin $Ref | Out-Null } catch {}
  try { git checkout -f FETCH_HEAD | Out-Null } catch { git checkout -f $Ref | Out-Null }
  Pop-Location
} else {
  if ($Ref -match '^llvmorg-') {
    git clone --depth 1 --branch $Ref https://github.com/llvm/llvm-project.git
  } else {
    git clone --depth 1 https://github.com/llvm/llvm-project.git
    Push-Location llvm-project
    git fetch origin $Ref --depth 1
    git checkout -f FETCH_HEAD
    Pop-Location
  }
}

# Install uv and lit
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  $uvInstall = "iwr -useb https://astral.sh/uv/install.ps1 | iex"
  Invoke-Expression $uvInstall
}
$uvBin = Join-Path $env:USERPROFILE ".local\bin"
$env:PATH = "$uvBin;$env:PATH"
uv tool install lit | Out-Null
$LIT = (Get-Command lit).Source

# Optional ccache
$launcherArgs = @()
if (Get-Command ccache -ErrorAction SilentlyContinue) {
  $launcherArgs += @('-DCMAKE_C_COMPILER_LAUNCHER=ccache','-DCMAKE_CXX_COMPILER_LAUNCHER=ccache','-DLLVM_CCACHE_BUILD=ON')
}

function Invoke-CMake($src, $build, $args) { cmake -S $src -B $build @args }
function Build-Install($build, $target='install') { cmake --build $build --config Release --target $target }

# Common args
$commonArgs = @(
  '-G','Ninja',
  '-DCMAKE_BUILD_TYPE=Release',
  '-DLLVM_INCLUDE_TESTS=OFF','-DLLVM_BUILD_TESTS=OFF',
  '-DLLVM_INCLUDE_EXAMPLES=OFF',
  '-DLLVM_ENABLE_ASSERTIONS=OFF',
  ("-DLLVM_TARGETS_TO_BUILD=$Targets"),
  '-DLLVM_ENABLE_LTO=Thin',
  '-DLLVM_ENABLE_ZSTD=ON',
  '-DLLVM_INSTALL_UTILS=ON',
  ("-DLLVM_HOST_TRIPLE=$hostTriple")
) + $launcherArgs

# Stage0: build clang (+compiler-rt profile)
if ($stageFrom -le 0 -and 0 -le $stageTo) {
  $stage0 = Join-Path $work 'stage0-install'
  $args0 = $commonArgs + @(
    '-DLLVM_ENABLE_PROJECTS=clang',
    '-DLLVM_ENABLE_RUNTIMES=compiler-rt',
    '-DCOMPILER_RT_BUILD_PROFILE=ON',
    '-DCOMPILER_RT_BUILD_SANITIZERS=OFF',
    '-DCOMPILER_RT_BUILD_XRAY=OFF',
    '-DCOMPILER_RT_BUILD_MEMPROF=OFF',
    ("-DCMAKE_INSTALL_PREFIX=$stage0")
  )
  Invoke-CMake 'llvm-project/llvm' 'build_stage0' $args0
  Build-Install 'build_stage0' 'install'
}

$env:CC = Join-Path (Join-Path $work 'stage0-install') 'bin/clang.exe'
$env:CXX = Join-Path (Join-Path $work 'stage0-install') 'bin/clang++.exe'

# Stage1: instrumented build with tests enabled to collect profiles
$rawDir = Join-Path $work 'pgoprof\raw'
if ($stageFrom -le 1 -and 1 -le $stageTo) {
  $env:LLVM_PROFILE_FILE = (Join-Path $rawDir '%p-%m.profraw')
  $commonFlags = ("-fprofile-instr-generate $cpuFlags").Trim()
  $args1 = $commonArgs + @(
    '-DLLVM_INCLUDE_TESTS=ON','-DLLVM_BUILD_TESTS=ON',
    '-DLLVM_ENABLE_PROJECTS=mlir',
    ("-DCMAKE_C_COMPILER=$env:CC"),("-DCMAKE_CXX_COMPILER=$env:CXX"),("-DCMAKE_ASM_COMPILER=$env:CC"),
    ("-DCMAKE_C_FLAGS=$commonFlags"),("-DCMAKE_CXX_FLAGS=$commonFlags"),
    ("-DLLVM_EXTERNAL_LIT=$LIT"),
    ("-DCMAKE_INSTALL_PREFIX=$work/stage1-install")
  )
  Invoke-CMake 'llvm-project/llvm' 'build_stage1' $args1
  Build-Install 'build_stage1' 'install'
  # Run tests to produce .profraw
  Build-Install 'build_stage1' 'check-mlir'
}

# Merge profiles
$profdata = Join-Path $work 'pgoprof\merged.profdata'
$profrawFiles = Get-ChildItem -Path $rawDir -Filter *.profraw -File -ErrorAction SilentlyContinue
if (-not $profrawFiles) {
  "Warning: no .profraw collected; proceeding with empty profile" | Write-Host
  New-Item -ItemType File -Force -Path $profdata | Out-Null
} else {
  & (Join-Path (Join-Path $work 'stage0-install') 'bin/llvm-profdata.exe') merge -output=$profdata $rawDir\*.profraw | Out-Null
}

# Stage2: final PGO+ThinLTO build
if ($stageFrom -le 2 -and 2 -le $stageTo) {
  $useFlags = ("-fprofile-use=$profdata -Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date $cpuFlags").Trim()
  $args2 = $commonArgs + @(
    '-DLLVM_ENABLE_PROJECTS=mlir',
    ("-DCMAKE_C_COMPILER=$env:CC"),("-DCMAKE_CXX_COMPILER=$env:CXX"),("-DCMAKE_ASM_COMPILER=$env:CC"),
    ("-DCMAKE_C_FLAGS=$useFlags"),("-DCMAKE_CXX_FLAGS=$useFlags"),
    ("-DLLVM_EXTERNAL_LIT=$LIT"),
    ("-DCMAKE_INSTALL_PREFIX=$Prefix")
  )
  Invoke-CMake 'llvm-project/llvm' 'build_stage2' $args2
  Build-Install 'build_stage2' 'install'
}

# Emit compressed archive (.tar.zst) using tar | zstd
$archiveName = "llvm-mlir_${Ref}_windows_${env:PROCESSOR_ARCHITECTURE}_$($Targets -replace ';','_')_opt.tar.zst"
$archivePath = Join-Path $work $archiveName
if (-not (Get-Command zstd -ErrorAction SilentlyContinue)) { Write-Error "zstd not found. Please install zstd." }
Push-Location $Prefix
try {
  if (Test-Path $archivePath) { Remove-Item -Force $archivePath }
  tar -cf - . | zstd -T0 -19 -o $archivePath
} finally { Pop-Location }

"Windows build completed at $Prefix (incremental, Zstd, HOST_TRIPLE=$hostTriple, cache=ccache)" | Write-Host
"Archive: $archivePath" | Write-Host
