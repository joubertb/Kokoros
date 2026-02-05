# syntax=docker/dockerfile:1
FROM rust:1.88.0-slim-bookworm AS builderrs

RUN apt-get update -qq && apt-get install -qq -y wget pkg-config libssl-dev clang git cmake libopus-dev ffmpeg && rustup component add rustfmt

WORKDIR /app

COPY . .

RUN chmod +x ./download_all.sh && ./download_all.sh

RUN cargo build --release

# Copy espeak-ng runtime data to a predictable path (hash in dir name varies per build)
RUN mkdir -p /app/espeak-ng-data && \
    cp -r /app/target/release/build/espeak-rs-sys-*/out/share/espeak-ng-data/* /app/espeak-ng-data/

FROM debian:sid-slim AS runner

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
