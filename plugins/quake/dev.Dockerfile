# Development container for developing the quake plugin
#
# Build the base image by checking out bde5f32d33 and running the following command:
# docker build -t nvidia/cuda-quantum-dev:latest -f docker/build/cudaq.dev.Dockerfile .
#
# Build this image by running the following command:
# docker build -t mqt/quake-dev:latest -f plugins/quake/dev.Dockerfile .
#
# Start up a container using the following command:
# docker run -it --rm \
# --mount type=bind,source="$(pwd)",target=/workspaces/core/ \
# mqt/quake-dev:latest

# Built locally with bde5f32d33
FROM nvidia/cuda-quantum-dev:latest

# Install missing dependencies
RUN apt-get update && apt-get install -y libcusparse-dev-12-0
RUN pip install cuquantum-cu12==25.06
RUN pip install cudensitymat-cu12==0.2.0

# Build cudaq
RUN bash scripts/build_cudaq.sh -j 1

# Install llvm
RUN apt update && apt install -y lsb-release wget software-properties-common gnupg
RUN wget https://apt.llvm.org/llvm.sh -O /tmp/llvm_install.sh
RUN chmod +x /tmp/llvm_install.sh
RUN bash /tmp/llvm_install.sh 20
RUN apt install -y libmlir-20-dev mlir-20-tools clang-20

# Set environment variables
ENV MQT_CORE_CC=/usr/lib/llvm-20/bin/clang
ENV MQT_CORE_CXX=/usr/lib/llvm-20/bin/clang++
ENV MQT_CORE_LLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm
ENV MQT_CORE_MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ENV MQT_CORE_LIT=/usr/local/bin/lit

# Change working directory
WORKDIR /workspaces/core
