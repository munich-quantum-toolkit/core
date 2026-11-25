param([Parameter(Mandatory)][ValidateSet('clang','msvc')][string]$Mode)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$root = $env:GITHUB_WORKSPACE
$source = Join-Path $root 'source'
$results = Join-Path $root 'results'
$env:STUDY_MODE = $Mode
$env:STUDY_RESULT = Join-Path $results $Mode
$wheelDir = Join-Path $env:STUDY_RESULT 'wheels'
New-Item -Type Directory -Force $wheelDir | Out-Null
Set-Location $source
if ((git rev-parse HEAD) -ne $env:SOURCE_SHA) { throw 'Unexpected source revision' }
if (Test-Path 'build') { Remove-Item -LiteralPath 'build' -Recurse -Force }
if (!(Test-Path "$env:MLIR_DIR/MLIRConfig.cmake")) { throw 'Missing common SDK' }
$tag = if ($env:RUNNER_ARCH -eq 'ARM64') { 'win_arm64' } else { 'win_amd64' }
$env:CIBW_BUILD = "cp311-$tag cp315t-$tag"
$env:CMAKE_GENERATOR = 'Ninja'
$env:TEMP = $env:RUNNER_TEMP
$env:TMP = $env:RUNNER_TEMP
$hook = (Join-Path $PSScriptRoot 'measure.py').Replace('\','/')
$env:CIBW_BEFORE_BUILD_WINDOWS = "python `"$hook`" start `"{project}`""
$env:CIBW_REPAIR_WHEEL_COMMAND_WINDOWS = "python `"$hook`" repair `"{project}`" `"{wheel}`" `"{dest_dir}`""
# Keep the first ABI cold on every runner and retain the build tree for the second.
$env:CMAKE_ARGS = '-DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER='
if ($Mode -eq 'msvc') {
  # Override the wheel preset to select the developer shell's default MSVC compiler.
  $env:SKBUILD_CMAKE_ARGS = '-DENABLE_IPO=ON'
} else {
  $compiler = Join-Path $env:VSINSTALLDIR "VC/Tools/Llvm/$env:VSCMD_ARG_TGT_ARCH/bin/clang-cl.exe"
  & $compiler --version | Tee-Object "$wheelDir/../clang-version.txt"
  & (Join-Path (Split-Path $compiler) 'lld-link.exe') --version | Tee-Object "$wheelDir/../lld-version.txt"
}
& cibuildwheel . --output-dir $wheelDir
$wheels = @(Get-ChildItem -Path $wheelDir -Filter '*.whl')
if ($wheels.Count -ne 2) { throw "Expected two repaired wheels for $Mode" }
$standard = @($wheels | Where-Object Name -like '*-cp311-abi3-*')
if ($standard.Count -ne 1) { throw 'Expected a cp311-abi3 wheel' }
$venv = Join-Path $env:RUNNER_TEMP "installed-$Mode"
uv venv --python 3.13 --managed-python $venv
uv pip install --python "$venv/Scripts/python.exe" --no-deps $standard[0].FullName
$cli = Resolve-Path "$venv/Lib/site-packages/mqt/core/bin/mqt-core.exe"
Add-Content -Path $env:GITHUB_ENV -Value ("{0}_CLI={1}" -f $Mode.ToUpperInvariant(), $cli.Path)
git diff --exit-code
