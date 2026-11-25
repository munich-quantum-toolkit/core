param(
  [Parameter(Mandatory)][ValidateSet('X64', 'ARM64')][string]$Architecture,
  [Parameter(Mandatory)][string]$InstallRoot,
  [string]$ThinLtoCache
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$llvmVersion = '23.1.6'
$tripleArch, $archiveSha, $zstdSha = if ($Architecture -eq 'ARM64') {
  'aarch64', 'd3f65e879cd4b1524a0c85c871e0fc26164644b28e618fc341fac7850d48ea38', 'ea1a61e8b1be2e01e67aa2f706a16087ae46ae4099adcc5a166fbd90a6dfd237'
} else {
  'x86_64', '7623f582089983c1cb51030cf1996514b19464e5669099571b399e8543770155', '7a8ddad468d6750bc5d532cac7efd3efd532125f2c098e2c9e9a14305d746002'
}
$archiveRoot = "clang+llvm-$llvmVersion-$tripleArch-pc-windows-msvc"
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$InstallRoot = (Resolve-Path $InstallRoot).Path
$requiredTools = @('clang-cl.exe', 'lld-link.exe', 'llvm-lib.exe', 'llvm-nm.exe')

# Reuse an exact native installation with all of the tools needed by CMake.
$clangBin = $null
$existing = Get-Command clang-cl.exe -ErrorAction SilentlyContinue
if ($existing) {
  $candidateBin = Split-Path $existing.Source
  $candidateVersion = (& $existing.Source --version | Out-String)
  if (($candidateVersion -match "clang version $([regex]::Escape($llvmVersion))\b") -and
      ($candidateVersion -match "Target: $tripleArch-.*-windows-msvc") -and
      (@($requiredTools | Where-Object { !(Test-Path (Join-Path $candidateBin $_)) }).Count -eq 0) -and
      ((& (Join-Path $candidateBin 'lld-link.exe') --version | Out-String) -match
       "\b$([regex]::Escape($llvmVersion))\b")) {
    $clangBin = $candidateBin
  }
}

function Get-VerifiedArchive([string]$URL, [string]$Path, [string]$SHA256) {
  if (!(Test-Path $Path)) {
    & curl.exe --fail --silent --show-error --location --retry 3 --output "$Path.partial" $URL
    Move-Item -LiteralPath "$Path.partial" -Destination $Path
  }
  if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $SHA256) {
    throw "Unexpected checksum for $Path"
  }
}

if (!$clangBin) {
  $archive = Join-Path $InstallRoot "$archiveRoot.tar.zst"
  $urlName = "$($archiveRoot.Replace('+', '%2B')).tar.zst"
  $url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-$llvmVersion/$urlName"
  Get-VerifiedArchive $url $archive $archiveSha
  $zstdName = "zstd-1.6.0_${tripleArch}-pc-windows-msvc.tar.gz"
  $zstdArchive = Join-Path $InstallRoot $zstdName
  Get-VerifiedArchive "https://github.com/munich-quantum-software/portable-mlir-toolchain/releases/download/2026.11.25/$zstdName" $zstdArchive $zstdSha
  $windowsTar = Join-Path ([Environment]::SystemDirectory) 'tar.exe'
  & $windowsTar -xzf $zstdArchive -C $InstallRoot zstd.exe
  $zstd = Join-Path $InstallRoot 'zstd.exe'
  # The command line tools and resource headers suffice for the MSVC C++ ABI.
  $extractTools = $requiredTools + @('clang.exe', 'llvm-ar.exe', 'llvm-ranlib.exe', 'llvm-mt.exe', 'llvm-rc.exe')
  $entries = @($extractTools | ForEach-Object { "$archiveRoot/bin/$_" }) +
    @("$archiveRoot/lib/clang/23")
  & $zstd -d $archive --long=31 --stdout | & $windowsTar -xf - -C $InstallRoot @entries
  $clangBin = Join-Path $InstallRoot "$archiveRoot/bin"
}

$compiler = Join-Path $clangBin 'clang-cl.exe'
$linker = Join-Path $clangBin 'lld-link.exe'
$nm = Join-Path $clangBin 'llvm-nm.exe'
$clangVersion = (& $compiler --version | Out-String)
if ($clangVersion -notmatch "clang version $([regex]::Escape($llvmVersion))\b" -or
    $clangVersion -notmatch "Target: $tripleArch-.*-windows-msvc") {
  throw "Unexpected clang-cl version or target: $clangVersion"
}
if ((& $linker --version | Tee-Object -Variable linkerVersion) -notmatch "\b$([regex]::Escape($llvmVersion))\b") {
  throw "Unexpected lld-link version: $linkerVersion"
}
Write-Host $clangVersion.TrimEnd()
Write-Host $linkerVersion
& $nm --version

$env:PATH = "$clangBin;$env:PATH"
$clangCMakeArgs = @(
  "-DCMAKE_C_COMPILER:FILEPATH=$($compiler.Replace('\', '/'))"
  "-DCMAKE_CXX_COMPILER:FILEPATH=$($compiler.Replace('\', '/'))"
  '-DCMAKE_LINKER_TYPE=LLD'
  "-DCMAKE_LINKER_LLD:FILEPATH=$($linker.Replace('\', '/'))"
  "-DCMAKE_NM:FILEPATH=$($nm.Replace('\', '/'))"
  '-DENABLE_IPO=ON'
)

if ($ThinLtoCache) {
  $cache = [IO.Path]::GetFullPath($ThinLtoCache).Replace('\', '/')
  foreach ($kind in 'EXE', 'SHARED', 'MODULE') {
    $clangCMakeArgs += "-DCMAKE_${kind}_LINKER_FLAGS_INIT=/lldltocache:`"$cache`""
  }
}
