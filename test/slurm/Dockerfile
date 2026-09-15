# syntax=docker/dockerfile:1.19

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

FROM ubuntu:26.04 AS spank

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        cmake g++ libslurm-dev
COPY --from=core spank /src/spank
RUN cmake -S /src/spank -B /build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build --parallel 2 \
    && cmake --install /build --prefix /usr/local

FROM ubuntu:26.04 AS common

ENV container=docker

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        dbus \
        munge \
        python3 \
        slurm-wlm \
        systemd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY --from=spank /usr/local/lib/slurm/mqt-core-qdmi-spank.so /usr/local/lib/slurm/
COPY --from=spank /usr/local/share/licenses/mqt-core-qdmi-spank /usr/local/share/licenses/mqt-core-qdmi-spank
COPY mqt-slurm-prepare /usr/local/sbin/mqt-slurm-prepare
COPY mqt-slurm-install-provider /usr/local/sbin/mqt-slurm-install-provider
COPY mqt-slurm-prepare.service /etc/systemd/system/mqt-slurm-prepare.service

RUN useradd --uid 10000 --user-group --create-home mqt-test

RUN --mount=type=bind,from=dist,target=/tmp/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages --link-mode=copy /tmp/dist/mqt_core-*.whl \
    && chmod 0755 /usr/local/sbin/mqt-slurm-prepare \
    && systemctl enable mqt-slurm-prepare.service munge.service

ARG SETUP_SCRIPT=""
ARG PROVIDER_INSTALL_MODE="native"
ARG PROVIDER_RUNTIME_COMPONENT=""
COPY --from=workload \
    --exclude=.git --exclude=build --exclude=build-* --exclude=cmake-build-* \
    --exclude=.cache --exclude=.ccache --exclude=.venv* --exclude=.nox \
    --exclude=**/__pycache__ --exclude=test/slurm/runtime --exclude=test/slurm/dist \
    . /workload
RUN --mount=type=bind,from=dist,target=/tmp/dist \
    if [ -n "$SETUP_SCRIPT" ]; then \
        /bin/sh /usr/local/sbin/mqt-slurm-install-provider \
        && /bin/sh "/workload/$SETUP_SCRIPT"; \
    fi

STOPSIGNAL SIGRTMIN+3

CMD ["/usr/lib/systemd/systemd"]

FROM common AS controller

RUN systemctl enable slurmctld.service

FROM common AS compute

COPY mqt-slurm-test-environment.conf /etc/systemd/system/slurmd.service.d/mqt-test.conf

RUN systemctl enable slurmd.service
