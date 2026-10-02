# Use the aarch64 CUDA 13 image already validated on the Thor host.
ARG BASE_IMAGE=vllm/vllm-openai:nightly-0cbac6cd1305f710e12193596b27488397bcb205
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build pkg-config libavformat-dev libavcodec-dev libavutil-dev \
    libswscale-dev libcurl4-openssl-dev cuda-nvtx-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /opt/ninfer
RUN cmake -S /opt/ninfer -B /opt/ninfer/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110a \
    && cmake --build /opt/ninfer/build -j --target ninfer-serve
ENTRYPOINT ["/opt/ninfer/build/apps/ninfer-serve"]
