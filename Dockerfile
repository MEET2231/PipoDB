# Stage 1: Build Environment
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN mkdir -p build && cd build && cmake .. && make pipodb_server

# Stage 2: Minimal Runtime Container
FROM ubuntu:22.04 AS runner

RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/pipodb_server /app/pipodb_server
COPY --from=builder /app/python /app/python

EXPOSE 8080

CMD ["./pipodb_server", "8080"]
