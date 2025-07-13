# syntax=docker/dockerfile:1
FROM rust:1.86.0-slim-bookworm AS builderrs

RUN apt-get update -qq && apt-get install -qq -y wget pkg-config libssl-dev clang git cmake && rustup component add rustfmt

WORKDIR /app

COPY . .
COPY Cargo.toml .
COPY Cargo.lock .

RUN chmod +x ./download_all.sh && ./download_all.sh

RUN cargo build --release

FROM debian:sid-slim AS runner

WORKDIR /app

COPY --from=builderrs /app/target/release/build ./target/release/build
COPY --from=builderrs /app/target/release/koko ./target/release/koko
COPY --from=builderrs /app/data ./data
COPY --from=builderrs /app/checkpoints ./checkpoints

RUN chmod +x ./target/release/koko && apt-get update -qq && apt-get install -qq -y pkg-config libssl-dev 

# Create tmp directory for kokoros temporary files
RUN mkdir -p tmp

EXPOSE 3025

ENTRYPOINT [ "./target/release/koko", "openai", "--port", "3025" ]
