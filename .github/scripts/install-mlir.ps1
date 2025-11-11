$ErrorActionPreference = "Stop"

# Detect architecture
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture

# Set asset name and URL
$tag = "test-release"

switch ($arch) {
    x64 {
        $asset_name = "windows-2025-archive.zip"
        $archive_name = "llvm-mlir_21.1.5_windows_x86_64_X86_opt.tar.zst"
    }
    arm64 {
        $asset_name = "windows-11-arm-archive.zip"
        $archive_name = "llvm-mlir_21.1.5_windows_arm64_AArch64_opt.tar.zst"
    }
    default {
        Write-Error "Unsupported architecture: $arch"; exit 1
    }
}

$release_url = "https://github.com/burgholzer/portable-mlir-toolchain/releases/download/$tag/$asset_name"

# Download asset
Write-Host "Downloading $asset_name from $release_url..."
Invoke-WebRequest -Uri $release_url -OutFile $asset_name

# Unzip asset
Write-Host "Unzipping $asset_name..."
Expand-Archive -Path $asset_name -DestinationPath . -Force

# Check for zstd
if (-not (Get-Command zstd -ErrorAction SilentlyContinue)) {
    Write-Error "zstd not found. Please install zstd (e.g., via Chocolatey: choco install zstd)."
    exit 1
}

# Check for tar
if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
    Write-Error "tar not found. Please install tar (e.g., via Chocolatey: choco install git)."
    exit 1
}

# Unpack archive
Write-Host "Extracting $archive_name..."
& zstd -d $archive_name --output-dir-flat .
& tar -xf ($archive_name -replace '.zst$', '')

Write-Host "Done. Archive extracted."
