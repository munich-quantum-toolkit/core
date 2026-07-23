#!/usr/bin/env sh
# Copyright (c) 2026 Chair for Design Automation, TUM
# Copyright (c) 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Clear only the mutable tool state belonging to this worktree.
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "${script_directory}/.." && pwd)
cache_root="${repository_root}/.cache"
uv_cache="${cache_root}/uv"
uv_python="${cache_root}/uv-python"
uv_tool_bin="${cache_root}/uv-bin"
uv_tools="${cache_root}/uv-tools"
ccache_cache="${cache_root}/ccache"
sccache_cache="${cache_root}/sccache"
sccache_socket="${cache_root}/sccache.sock"

if [ -d "${uv_cache}" ]; then
  if command -v uv >/dev/null 2>&1; then
    UV_CACHE_DIR="${uv_cache}" uv cache clean
  else
    rm -rf -- "${uv_cache}"
  fi
fi

rm -rf -- "${uv_python}" "${uv_tool_bin}" "${uv_tools}"

if [ -d "${ccache_cache}" ]; then
  if command -v ccache >/dev/null 2>&1; then
    CCACHE_DIR="${ccache_cache}" ccache --clear
  else
    rm -rf -- "${ccache_cache}"
  fi
fi

if [ -S "${sccache_socket}" ]; then
  SCCACHE_DIR="${sccache_cache}" SCCACHE_SERVER_UDS="${sccache_socket}" \
    sccache --stop-server
fi

rm -rf -- "${cache_root}"

echo "Cleared worktree-local caches under ${cache_root}"
