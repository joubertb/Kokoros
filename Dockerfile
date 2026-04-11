# Unified Dockerfile for Kokoros TTS (CPU and GPU)
#
# CPU build (default):
#   docker build -f Kokoros/Dockerfile Kokoros/
#
# GPU build:
#   docker build -f Kokoros/Dockerfile Kokoros/ \
#     --build-arg BASE_BUILD=nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 \
#     --build-arg BASE_RUNTIME=nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 \
#     --build-arg CARGO_FEATURES=cuda

ARG BASE_BUILD=rust:1.88.0-slim-bookworm
ARG BASE_RUNTIME=debian:sid-slim

# ============================================================
# Builder stage
# ============================================================
FROM ${BASE_BUILD} AS builderrs

ARG CARGO_FEATURES=

# Install build dependencies.
# On nvidia/cuda base, Rust is not pre-installed — install it.
# On rust: base, this is a no-op (rustup already present).
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

RUN cargo build --release ${CARGO_FEATURES:+--features $CARGO_FEATURES}

# Copy espeak-ng runtime data to a predictable path (hash in dir name varies per build)
RUN mkdir -p /app/espeak-ng-data && \
    cp -r /app/target/release/build/espeak-rs-sys-*/out/share/espeak-ng-data/* /app/espeak-ng-data/

# ============================================================
# Runtime stage
# ============================================================
FROM ${BASE_RUNTIME} AS runner

# Set CUDA paths (only meaningful for GPU builds, harmless on CPU)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY --from=builderrs /app/target/release/koko ./target/release/koko
# espeak-ng phoneme data (8.8 MB) - needed at runtime for text-to-phoneme conversion
COPY --from=builderrs /app/espeak-ng-data ./espeak-ng-data
COPY --from=builderrs /app/data ./data
COPY --from=builderrs /app/checkpoints ./checkpoints

# Point espeak-rs to the directory *containing* espeak-ng-data/
ENV PIPER_ESPEAKNG_DATA_DIRECTORY=/app

RUN chmod +x ./target/release/koko && apt-get update -qq && apt-get install -qq -y libssl3 curl libopus0 ffmpeg && rm -rf /var/lib/apt/lists/*

# Create tmp directory for kokoros temporary files
RUN mkdir -p tmp

EXPOSE 3025

ENTRYPOINT [ "./target/release/koko", "openai", "--port", "3025" ]
