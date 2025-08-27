# Cross-platform (Windows) setup for LLVM/MLIR by building from source when needed.
# Usage: pwsh .github/scripts/setup-mlir.ps1 -MajorVersion 20 [-InstallPrefix C:\path\to\llvm-install] [-Tag 20.1.0]
param(
  [Parameter(Mandatory = $true)][int]$MajorVersion,
  [string]$InstallPrefix = "$env:GITHUB_WORKSPACE\llvm-install",
  [string]$Tag
)

$ErrorActionPreference = 'Stop'

function Append-Env([string]$Key, [string]$Value) {
  if (-not $env:GITHUB_ENV) {
    throw "GITHUB_ENV not set. Are you running in GitHub Actions?"
  }
  "${Key}=${Value}" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}

function Resolve-LatestTag {
  param([int]$Major)
  Write-Host "Resolving latest llvmorg-$Major.* tag from upstream..."
  $tagLine = git ls-remote --tags https://github.com/llvm/llvm-project.git "llvmorg-$Major.*" |
      Where-Object { $_ -match 'llvmorg-\d+\.\d+\.\d+' } |
      Sort-Object -Property {
        if ($_ -match 'llvmorg-(\d+)\.(\d+)\.(\d+)') {
          [int]$Matches[1]*10000 + [int]$Matches[2]*100 + [int]$Matches[3]
        }
      } |
      Select-Object -Last 1

  $resolved = ($tagLine | ForEach-Object {
      if ($_ -match 'refs/tags/llvmorg-(\d+\.\d+\.\d+)') { $Matches[1] }
    }) | Where-Object { $_ -ne $null }

  if (-not $resolved) { throw "Failed to resolve latest llvmorg-$Major tag." }
  return $resolved
}

function Append-DirsToEnv {
  param([string]$Prefix)
  $llvmDir = Join-Path $Prefix 'lib/cmake/llvm'
  $mlirDir = Join-Path $Prefix 'lib/cmake/mlir'
  Append-Env -Key 'LLVM_DIR' -Value $llvmDir
  Append-Env -Key 'MLIR_DIR' -Value $mlirDir
}

function Ensure-LLVM {
  param(
    [int]$Major,
    [string]$Prefix,
    [string]$ResolvedTag
  )

  $llvmDir = Join-Path $Prefix 'lib/cmake/llvm'
  $mlirDir = Join-Path $Prefix 'lib/cmake/mlir'

  if (Test-Path $llvmDir -and Test-Path $mlirDir) {
    Write-Host "Found existing LLVM/MLIR install at $Prefix. Skipping build."
    Append-DirsToEnv -Prefix $Prefix
    return
  }

  if (-not $ResolvedTag) {
    $ResolvedTag = Resolve-LatestTag -Major $Major
  }

  Write-Host "Building LLVM/MLIR $ResolvedTag into $Prefix..."
  $repoDir = Join-Path $PWD 'llvm-project'
  if (Test-Path $repoDir) { Remove-Item -Recurse -Force $repoDir }
  git clone --depth 1 https://github.com/llvm/llvm-project.git --branch "llvmorg-$ResolvedTag"
  Push-Location $repoDir
  try {
    $buildDir = 'build_llvm'
    # Use argument array to avoid PowerShell line-continuation quirks on Windows
    $cmakeArgs = @(
      '-S','llvm',
      '-B',$buildDir,
      '-DLLVM_ENABLE_PROJECTS=mlir',
      '-DLLVM_BUILD_EXAMPLES=OFF',
      '-DLLVM_TARGETS_TO_BUILD=Native',
      '-DCMAKE_BUILD_TYPE=Release',
      '-DLLVM_BUILD_TESTS=OFF',
      '-DLLVM_INCLUDE_TESTS=OFF',
      '-DLLVM_INCLUDE_EXAMPLES=OFF',
      '-DLLVM_ENABLE_ASSERTIONS=ON',
      '-DLLVM_INSTALL_UTILS=ON',
      "-DCMAKE_INSTALL_PREFIX=$Prefix"
    )
    cmake @cmakeArgs

    cmake --build $buildDir --target install --config Release
  }
  finally {
    Pop-Location
  }

  Append-DirsToEnv -Prefix $Prefix
}

Ensure-LLVM -Major $MajorVersion -Prefix $InstallPrefix -ResolvedTag $Tag
Write-Host "LLVM/MLIR setup complete. LLVM_DIR and MLIR_DIR exported."
