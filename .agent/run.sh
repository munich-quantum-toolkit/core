#!/usr/bin/env sh
# Copyright (c) 2026 Chair for Design Automation, TUM
# Copyright (c) 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Run a command with all mutable tool state kept inside this worktree.
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "${script_directory}/.." && pwd)
cache_root="${repository_root}/.cache"

export XDG_CACHE_HOME="${cache_root}/xdg"
export PREK_HOME="${cache_root}/prek"
export UV_CACHE_DIR="${cache_root}/uv"
export UV_PYTHON_INSTALL_DIR="${cache_root}/uv-python"
export UV_TOOL_BIN_DIR="${cache_root}/uv-bin"
export UV_TOOL_DIR="${cache_root}/uv-tools"
export CCACHE_DIR="${cache_root}/ccache"
export CCACHE_MAXSIZE="4GiB"
export CCACHE_TEMPDIR="${cache_root}/ccache/tmp"
export SCCACHE_CACHE_SIZE="4G"
export SCCACHE_DIR="${cache_root}/sccache"
export SCCACHE_IDLE_TIMEOUT="60"

case "$(uname -s)" in
  Darwin | Linux)
    export SCCACHE_SERVER_UDS="${cache_root}/sccache.sock"
    ;;
esac

exec "$@"
