FROM ubuntu:22.04

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y curl git wget && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="$HOME/.local/bin:$PATH"

# Set up Python environment
RUN uv venv --python 3.12
SHELL ["/bin/bash", "-c"]
RUN source .venv/bin/activate

# Install CUDA cross-compiler for aarch64
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-aarch64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-cross-aarch64 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq \
    CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum \
    CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor \
    LLVM_INSTALL_PREFIX=/usr/local/llvm \
    BLAS_INSTALL_PREFIX=/usr/local/blas \
    ZLIB_INSTALL_PREFIX=/usr/local/zlib \
    OPENSSL_INSTALL_PREFIX=/usr/local/openssl \
    CURL_INSTALL_PREFIX=/usr/local/curl \
    AWS_INSTALL_PREFIX=/usr/local/aws

RUN export GCC_TOOLCHAIN=$(which gcc) && \
    export CXX="$GCC_TOOLCHAIN/bin/g++" && \
    export CC="$GCC_TOOLCHAIN/bin/gcc"

# Clone cudaq
WORKDIR /home
RUN git clone https://github.com/NVIDIA/cuda-quantum.git
WORKDIR /home/cuda-quantum

# Build cudaq
# RUN CUDAQ_ENABLE_STATIC_LINKING=TRUE \
#     CUDAQ_REQUIRE_OPENMP=TRUE \
#     CUDAQ_WERROR=TRUE \
#     CUDAQ_PYTHON_SUPPORT=OFF \
#     LLVM_PROJECTS='clang;flang;lld;mlir;openmp;runtimes' \
#     bash scripts/build_cudaq.sh -t llvm -v
