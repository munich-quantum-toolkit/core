#!/bin/bash
set -e

# Usage: ./build-and-setup-mlir-linux.sh -t 21.1.0 [-p /path/to/llvm-install]

# Default values
INSTALL_PREFIX="${GITHUB_WORKSPACE}/llvm-install"

# Parse arguments
while getopts "t:p:" opt; do
  case $opt in
    t) TAG="$OPTARG"
    ;;
    p) INSTALL_PREFIX="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

# Check for required tag argument
if [ -z "$TAG" ]; then
  echo "Error: Tag (-t) is required"
  echo "Usage: $0 -t <tag> [-p /path/to/llvm-install]"
  exit 1
fi

# Function to append to GITHUB_ENV
append_env() {
  if [ -z "$GITHUB_ENV" ]; then
    echo "GITHUB_ENV not set. Are you running in GitHub Actions?"
    exit 1
  fi
  echo "$1=$2" >> "$GITHUB_ENV"
}

# Function to append LLVM/MLIR directories to environment
append_dirs_to_env() {
  local prefix=$1
  local llvm_dir="$prefix/lib/cmake/llvm"
  local mlir_dir="$prefix/lib/cmake/mlir"
  append_env "LLVM_DIR" "$llvm_dir"
  append_env "MLIR_DIR" "$mlir_dir"
}

# Get number of CPU cores for parallel build
if [ -f /proc/cpuinfo ]; then
  # Linux
  NCORES=$(grep -c ^processor /proc/cpuinfo)
else
  # Fallback
  NCORES=4
fi

# Main LLVM setup function
ensure_llvm() {
  local tag=$1
  local prefix=$2

  local llvm_dir="$prefix/lib/cmake/llvm"
  local mlir_dir="$prefix/lib/cmake/mlir"

  # Check if LLVM is already installed
  if [ -d "$llvm_dir" ] && [ -d "$mlir_dir" ]; then
    echo "Found existing LLVM/MLIR install at $prefix. Skipping build."
    append_dirs_to_env "$prefix"
    return
  fi

  echo "Building LLVM/MLIR $tag into $prefix..."

  # Clone LLVM project
  rm -rf "$prefix"
  mkdir -p "$prefix"
  git clone --depth 1 https://github.com/llvm/llvm-project.git --branch "llvmorg-$tag" "$prefix/llvm-project"

  pushd "$prefix/llvm-project" > /dev/null

  # Build LLVM
  build_dir="build_llvm"
  cmake -S llvm -B "$build_dir" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD=Native \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_BUILD_TESTS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_INSTALL_PREFIX="$prefix"

  # Use all available cores for the build on Linux
  cmake --build "$build_dir" --target install --config Release -j${NCORES}

  popd > /dev/null

  append_dirs_to_env "$prefix"
}

# Execute main function
ensure_llvm "$TAG" "$INSTALL_PREFIX"
echo "LLVM/MLIR setup complete. LLVM_DIR and MLIR_DIR exported."
