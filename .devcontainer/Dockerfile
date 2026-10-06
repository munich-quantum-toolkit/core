FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

# Installs the basic underlying framework
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends clang clangd clang-tools lld

# Copy uv and uvx binaries from the official image and make them accessible to all users
COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /usr/local/bin/

# Store tool binaries and environments in system-wide locations
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_TOOL_DIR=/opt/uv/tools

# Install the latest versions of cmake, ninja, and prek
RUN uv tool install cmake && \
    uv tool install ninja && \
    uv tool install prek

# Downloads and installs a prebuilt LLVM/MLIR toolchain
ARG SETUP_MLIR_BASE=https://raw.githubusercontent.com/munich-quantum-software/setup-mlir/5f3c4771ebb8d8e4fe284378b34460f428860a66
RUN curl -LsSf "$SETUP_MLIR_BASE/installation/setup-mlir.sh" -o /tmp/setup-mlir.sh
RUN SETUP_MLIR_MANIFEST_URL="$SETUP_MLIR_BASE/version-manifest.json" bash /tmp/setup-mlir.sh -v 23.1.2 -p /opt/llvm && rm /tmp/setup-mlir.sh
