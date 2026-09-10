#!/usr/bin/env bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

project=$(cd "$(dirname "$0")/../.." && pwd)
cd "$project"
: "${STUDY_ROOT:?}" "${STUDY_OPERATION:?}" "${STUDY_SDK_LTO:?}"
mkdir -p "$STUDY_ROOT/source" "$STUDY_ROOT/input-sdk"
if [[ $(uname -s) == Linux ]]; then
  /opt/python/cp314-cp314/bin/python3 -m pip install uv==0.12.5
  export PATH="/opt/python/cp314-cp314/bin:$PATH"
  manylinux-install-clang -v 22.1.8.1 -c 8b399744aeb49c70048b379b9b3ffc651d86fde808551c8cc4138c4fadc5308e
  export CC=/opt/clang/bin/clang CXX=/opt/clang/bin/clang++
  export AR=/opt/clang/bin/llvm-ar RANLIB=/opt/clang/bin/llvm-ranlib
  export LLVM_PROFDATA=/opt/clang/bin/llvm-profdata
  export PATH="/opt/clang/bin:$PATH"
  git config --global --add safe.directory "$project"
else
  export CC CXX AR RANLIB LLVM_PROFDATA SDKROOT
  CC=$(xcrun --find clang)
  CXX=$(xcrun --find clang++)
  AR=$(xcrun --find ar)
  RANLIB=$(xcrun --find ranlib)
  LLVM_PROFDATA=$(xcrun --find llvm-profdata)
  SDKROOT=$(xcrun --show-sdk-path)
  xcodebuild -version > "$STUDY_ROOT/xcode.txt"
  xcrun --show-sdk-build-version >> "$STUDY_ROOT/xcode.txt"
fi

uv venv --python 3.14.7 "$STUDY_ROOT/venv"
uv export --frozen --no-default-groups --group build --group test-base --no-emit-project --no-hashes > "$STUDY_ROOT/build-requirements.txt"
uv pip sync --python "$STUDY_ROOT/venv/bin/python" "$STUDY_ROOT/build-requirements.txt"
uv pip install --python "$STUDY_ROOT/venv/bin/python" wheel==0.45.1 "delocate==0.13.0; sys_platform == 'darwin'"
export PATH="$STUDY_ROOT/venv/bin:$PATH"

llvm_commit=ea7d852a70e8bdfaf601d6626a760f9771b2c4b4
curl --fail --location --retry 3 "https://github.com/llvm/llvm-project/archive/$llvm_commit.tar.gz" -o "$STUDY_ROOT/source.tar.gz"
tar -xf "$STUDY_ROOT/source.tar.gz" -C "$STUDY_ROOT/source" --strip-components=1
python -c 'import hashlib,sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").hexdigest())' "$STUDY_ROOT/source.tar.gz" > "$STUDY_ROOT/source.sha256"
rm "$STUDY_ROOT/source.tar.gz"
archive=$(find "$STUDY_ROOT/download" -name '*.tar.zst' -print -quit)
test -n "$archive"
python -m tarfile --extract "$archive" "$STUDY_ROOT/input-sdk"

python scripts/optimization_study.py "$STUDY_OPERATION" \
  --root "$STUDY_ROOT/output" --sdk "$STUDY_ROOT/input-sdk" \
  --llvm-source "$STUDY_ROOT/source" --llvm-source-id "$llvm_commit" \
  --toolchain-repo "$project/toolchain" --sdk-lto "$STUDY_SDK_LTO" \
  --core-lto "${STUDY_CORE_LTO:-OFF}" --pgo "${STUDY_PGO:-none}" \
  --jobs "${STUDY_JOBS:-4}" --lto-workers "${STUDY_LTO_WORKERS:-1}"
