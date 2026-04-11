# Unified Dockerfile for Kokoros TTS
#
# Builds a single image that supports both CPU and GPU execution.
# GPU acceleration is enabled automatically at runtime when a GPU is detected.
#
# Build:
#   docker build -f Kokoros/Dockerfile Kokoros/

ARG BASE_BUILD=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG BASE_RUNTIME=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# ============================================================
# Builder stage
# ============================================================
FROM ${BASE_BUILD} AS builderrs

ARG ONNXRUNTIME_VERSION=1.22.0

# Install build dependencies.
RUN apt-get update -qq && apt-get install -qq -y \
    wget pkg-config libssl-dev clang git cmake libopus-dev ffmpeg curl && \
    if ! command -v rustup >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0; \
    fi

ENV PATH="/root/.cargo/bin:/usr/local/cargo/bin:${PATH}"

RUN rustup component add rustfmt

WORKDIR /app

COPY . .

RUN chmod +x ./download_all.sh && ./download_all.sh

# Build with cuda + ort-dynamic: CUDA EP compiled in, ORT loaded at runtime
RUN cargo build --release --features cuda,kokoros/ort-dynamic

# Copy espeak-ng runtime data to a predictable path (hash in dir name varies per build)
RUN mkdir -p /app/espeak-ng-data && \
    cp -r /app/target/release/build/espeak-rs-sys-*/out/share/espeak-ng-data/* /app/espeak-ng-data/

# Download GPU-enabled ONNX Runtime (includes CUDA/TensorRT EPs + CPU fallback)
RUN mkdir -p /app/onnx_libs && \
    echo "Downloading ONNX Runtime GPU ${ONNXRUNTIME_VERSION}..." && \
    curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz | \
    tar xzf - -C /app/onnx_libs --strip-components=2 --wildcards "*/lib/libonnxruntime*.so*" && \
    echo "ONNX Runtime libraries:" && ls -la /app/onnx_libs/

# ============================================================
# Runtime stage
# ============================================================
FROM ${BASE_RUNTIME} AS runner

# Set CUDA paths (harmless on CPU-only execution)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY --from=builderrs /app/target/release/koko ./target/release/koko
# espeak-ng phoneme data (8.8 MB) - needed at runtime for text-to-phoneme conversion
COPY --from=builderrs /app/espeak-ng-data ./espeak-ng-data
COPY --from=builderrs /app/data ./data
COPY --from=builderrs /app/checkpoints ./checkpoints
# GPU-enabled ONNX Runtime libraries
COPY --from=builderrs /app/onnx_libs/ /usr/local/lib/

# Point espeak-rs to the directory *containing* espeak-ng-data/
ENV PIPER_ESPEAKNG_DATA_DIRECTORY=/app

RUN chmod +x ./target/release/koko && apt-get update -qq && apt-get install -qq -y libssl3 curl libopus0 ffmpeg && rm -rf /var/lib/apt/lists/*

RUN ldconfig

# Create tmp directory for kokoros temporary files
RUN mkdir -p tmp

EXPOSE 3025

ENTRYPOINT [ "./target/release/koko", "openai", "--port", "3025" ]
