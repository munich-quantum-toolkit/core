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
RUN mkdir -p /tmp/setup-mlir && \
    curl -LsSf https://github.com/munich-quantum-software/setup-mlir/archive/a74d2bc79d070925638168c2d4b4aa84c068af51.tar.gz | tar -xz -C /tmp/setup-mlir --strip-components=1
RUN bash /tmp/setup-mlir/installation/setup-mlir.sh -v 23.1.2 -p /opt/llvm && rm -rf /tmp/setup-mlir
