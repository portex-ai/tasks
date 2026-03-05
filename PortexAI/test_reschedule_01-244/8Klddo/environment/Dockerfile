FROM ubuntu:24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG SYSTEM_PACKAGES=""
RUN if [ -n "$SYSTEM_PACKAGES" ]; then \
      apt-get update && apt-get install -y $SYSTEM_PACKAGES && rm -rf /var/lib/apt/lists/*; \
    fi

ARG PYTHON_PACKAGES=""
RUN if [ -n "$PYTHON_PACKAGES" ]; then \
      uv pip install --system $PYTHON_PACKAGES; \
    fi

WORKDIR /app
