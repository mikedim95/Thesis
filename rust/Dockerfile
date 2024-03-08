
# Stage 2: Cross-compile the app for musl to create a statically-linked binary for Alpine
FROM rust:latest AS rust
RUN rustup target add aarch64-unknown-linux-gnu
RUN apt-get update && apt-get install -y --no-install-recommends g++-aarch64-linux-gnu libc6-dev-arm64-cross 
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

WORKDIR /app
COPY .cargo ./.cargo
COPY Cargo.toml Cargo.lock .rustfmt.toml ./
COPY src ./src 
RUN cargo build --target aarch64-unknown-linux-gnu --release
# Move the binary to a location free of the target since that is not available in the next stage.
RUN cp target/aarch64-unknown-linux-gnu/release/rust .

FROM arm64v8/debian:latest
ENV \
    # Show full backtraces for crashes.
    RUST_BACKTRACE=full
WORKDIR /app
COPY --from=rust /app/rust ./
# Set the working directory


# Run the executable when the container starts
CMD ["./rust"]