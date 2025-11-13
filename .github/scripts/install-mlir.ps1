# Usage: install-mlir.ps1 -tag <tag> -install_prefix <installation directory>
param(
    [Parameter(Mandatory=$true)]
    [string]$tag,
    [Parameter(Mandatory=$true)]
    [string]$install_prefix,
    [string]$token
)

$ErrorActionPreference = "Stop"

# Change to installation directory
pushd $install_prefix > $null

# Detect architecture
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture

# Determine download URL
$headers = @{
    "Accept" = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}
if ($token) {
    $headers["Authorization"] = "Bearer $token"
}

$release_url = "https://api.github.com/repos/burgholzer/portable-mlir-toolchain/releases/tags/$tag"
$release_json = Invoke-RestMethod -Uri $release_url -Headers $headers

$assets_url = $release_json.assets_url
$assets_json = Invoke-RestMethod -Uri $assets_url -Headers $headers

$download_urls = $assets_json | ForEach-Object { $_.browser_download_url }

switch ($arch) {
    x64 {
        $download_url = $download_urls | Where-Object { $_ -match '.*_windows_.*_X86\.tar\.zst' }
    }
    arm64 {
        $download_url = $download_urls | Where-Object { $_ -match '.*_windows_.*_AArch64\.tar\.zst' }
    }
    default {
        Write-Error "Unsupported architecture: $arch"; exit 1
    }
}

# Download asset
Write-Host "Downloading asset from $download_url ..."
Invoke-WebRequest -Uri $download_url -OutFile "asset.tar.zst"

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
Write-Host "Extracting archive..."
& zstd -d "asset.tar.zst" --output-dir-flat .
& tar -xf "asset.tar"

# Return to original directory
popd > $null

# Output instructions
Write-Host "MLIR toolchain has been installed"
Write-Host "Run the following commands to set up your environment:"
Write-Host "  `$env:LLVM_DIR = '$(Get-Location -PSProvider FileSystem | Select-Object -ExpandProperty Path)\lib\cmake\llvm'"
Write-Host "  `$env:MLIR_DIR = '$(Get-Location -PSProvider FileSystem | Select-Object -ExpandProperty Path)\lib\cmake\mlir'"
Write-Host "  `$env:Path = '$(Get-Location -PSProvider FileSystem | Select-Object -ExpandProperty Path)\bin;`$env:Path'"
