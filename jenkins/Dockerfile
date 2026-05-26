FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      'python3.12' \
      'python3.12-venv' \
      git \
      curl \
      pandoc \
      jq \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*
