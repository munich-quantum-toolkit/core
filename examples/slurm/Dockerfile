# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

FROM ghcr.io/astral-sh/uv:0.13.0 AS uv

FROM ubuntu:26.04 AS base

ENV container=docker \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        libcurl4t64 \
        dbus \
        munge \
        slurm-wlm \
        systemd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
ARG UV_LINK_MODE=copy
ARG UV_PYTHON_CACHE_DIR=/root/.cache/uv/python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.15 --compile-bytecode \
    && uv venv --python 3.15 /opt/venv
COPY --chmod=755 start.sh /usr/local/sbin/mqt-slurm-start
COPY --chmod=755 admission.sh /usr/local/sbin/mqt-qdmi-admission
COPY --chmod=755 monitor.sh /usr/local/sbin/mqt-qdmi-monitor
COPY --chmod=755 update_availability.py /usr/local/libexec/mqt-qdmi-availability.py
COPY qdmi-*.service /etc/systemd/system/
COPY node.conf /etc/systemd/system/slurmd.service.d/mqt-node.conf

RUN useradd --uid 10000 --user-group --create-home mqt-test \
    && systemctl disable slurmctld.service slurmd.service

FROM scratch AS provider-source

COPY --from=workload \
    --exclude=.git --exclude=.github --exclude=build --exclude=build-* --exclude=cmake-build-* \
    --exclude=.cache --exclude=.ccache --exclude=.venv* --exclude=.nox --exclude=CMakeUserPresets.json \
    --exclude=.ruff_cache --exclude=.rumdl_cache --exclude=.pytest_cache --exclude=.mypy_cache \
    --exclude=**/__pycache__ --exclude=dist --exclude=test/slurm/runtime --exclude=test/slurm/dist \
    . /workload

FROM ubuntu:26.04 AS provider-build

COPY --from=uv /uv /uvx /bin/
ARG UV_LINK_MODE=copy
ARG UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

ARG SETUP_SCRIPT=""
RUN if [ -n "$SETUP_SCRIPT" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
            build-essential ca-certificates cmake git libcurl4-openssl-dev libssl-dev; \
    fi

ARG PROVIDER_RUNTIME_COMPONENT=""
RUN --mount=type=bind,from=provider-source,source=/workload,target=/workload \
    --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /opt/provider/wheels /opt/provider/native \
    && if [ -n "$SETUP_SCRIPT" ]; then \
        CMAKE_BUILD_PARALLEL_LEVEL=2 uv build /workload --python 3.15 --wheel --out-dir /opt/provider/wheels \
            --config-setting=build-dir=/tmp/provider-build \
        && cmake --install /tmp/provider-build --prefix /opt/provider/native \
            --component "${PROVIDER_RUNTIME_COMPONENT:?Set the device Runtime component}"; \
    fi

FROM base AS common

ARG SETUP_SCRIPT=""
ARG PROVIDER_INSTALL_MODE="native"
RUN --mount=type=bind,from=dist,target=/tmp/dist \
    --mount=type=bind,from=provider-source,source=/workload,target=/workload \
    --mount=type=bind,from=provider-build,source=/opt/provider,target=/tmp/provider \
    --mount=type=cache,target=/root/.cache/uv \
    set -- /tmp/dist/mqt_core-*.whl \
    && test "$#" -eq 1 && test -f "$1" \
    && printf 'mqt-core[qiskit,pennylane] @ file://%s\n' "$1" > /tmp/core-override.txt \
    && printf '%s[qiskit,pennylane]\n' "$1" > /tmp/wheels.txt \
    && for wheel in /tmp/dist/*.whl /tmp/dist/*/*.whl /tmp/provider/wheels/*.whl; do \
        [ -f "$wheel" ] || continue; \
        case "$wheel" in /tmp/dist/mqt_core-*) ;; *) printf '%s[qiskit]\n' "$wheel" >> /tmp/wheels.txt ;; esac; \
    done \
    && uv pip install --compile-bytecode --overrides /tmp/core-override.txt -r /tmp/wheels.txt \
    && rm /tmp/core-override.txt /tmp/wheels.txt \
    && if [ -n "$SETUP_SCRIPT" ]; then \
        case "$PROVIDER_INSTALL_MODE" in native|wheel) ;; *) exit 1 ;; esac \
        && if [ "$PROVIDER_INSTALL_MODE" = native ]; then cp -a /tmp/provider/native /opt/provider-native; fi \
        && /bin/sh "/workload/$SETUP_SCRIPT"; \
    fi

STOPSIGNAL SIGRTMIN+3
CMD ["/usr/local/sbin/mqt-slurm-start", "node"]
