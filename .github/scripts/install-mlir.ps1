# Usage: & install-mlir.ps1 -tag <tag> -installation_dir <installation directory>
param(
    [Parameter(Mandatory=$true)]
    [string]$tag,
    [Parameter(Mandatory=$true)]
    [string]$install_prefix
)

$ErrorActionPreference = "Stop"

# Change to installation directory
Set-Location -Path $install_prefix

# Detect architecture
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture

# Set asset name
switch ($arch) {
    x64 {
        $asset_name = "windows-2022-archive.zip"
    }
    arm64 {
        $asset_name = "windows-11-arm-archive.zip"
    }
    default {
        Write-Error "Unsupported architecture: $arch"; exit 1
    }
}

# Set asset URL
$asset_url = "https://github.com/burgholzer/portable-mlir-toolchain/releases/download/$tag/$asset_name"

# Download asset
Write-Host "Downloading $asset_name from $asset_url..."
Invoke-WebRequest -Uri $asset_url -OutFile $asset_name

# Unzip asset
Write-Host "Unzipping $asset_name..."
Expand-Archive -Path $asset_name -DestinationPath . -Force

# Find archive after unzip
$archive_path = Get-ChildItem -Recurse -File -Filter "*.tar.zst" | Select-Object -First 1
if (-not $archive_path) {
    Write-Error "No archive found after unzip of $asset_name."
    exit 1
}

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
& zstd -d $archive_path --output-dir-flat .
& tar -xf ($archive_path -replace '.zst$', '')

# Output instructions
Write-Host "MLIR toolchain has been installed"
Write-Host "Run the following commands to set up your environment:"
Write-Host "  \$env:LLVM_DIR = \"$PWD\lib\cmake\llvm\""
Write-Host "  \$env:MLIR_DIR = \"$PWD\lib\cmake\mlir\""
Write-Host "  \$env:PATH = \"$PWD\bin;\" + \$env:PATH"
