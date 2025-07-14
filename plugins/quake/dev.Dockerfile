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
# --mount type=bind,source="$(pwd)",target=/workspace/core/ \
# mqt/quake-dev:latest

# Built locally with bde5f32d33
FROM nvidia/cuda-quantum-dev:latest

# Install missing dependencies
RUN apt-get update && apt-get install -y libcusparse-dev-12-0
RUN pip install cuquantum-cu12==25.06
RUN pip install cudensitymat-cu12==0.2.0

# Build cudaq
RUN bash scripts/build_cudaq.sh -j 1
