# manylinux_2_28 builder image for optimized LLVM/MLIR (PGO + ThinLTO + BOLT)
#
# Purpose:
#   Minimal image to build LLVM/MLIR inside a manylinux_2_28 environment. Used by
#   scripts/toolchain/linux/build.sh to produce portable Linux binaries with PGO,
#   ThinLTO, and BOLT post-link optimization (when perf is available on host).
#
# Inputs:
#   ARG BASE_IMAGE  Base manylinux_2_28 image (arch-specific), provided by wrapper.
#
# Installs:
#   - Build tools: git, cmake, ninja-build, gcc-c++, make, which, tar, xz, zstd
#   - Runtime helpers: python3 + pip, curl, perf (for BOLT profiling), ccache
#
# Notes:
#   - The container runs without an entrypoint; the wrapper script binds /work and /out,
#     sets env, and invokes scripts/toolchain/linux/in-container.sh.

ARG BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${BASE_IMAGE}

# Install build dependencies
RUN yum -y update && \
    yum -y install git cmake ninja-build python3 python3-pip curl zstd xz tar which gcc-c++ make perf ccache && \
    yum clean all && rm -rf /var/cache/yum

ENV PATH=/usr/bin:$PATH

WORKDIR /work

# Expect to mount /work with scripts and output paths
# This image intentionally has no entrypoint; scripts will run with `docker run`.
