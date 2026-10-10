# syntax=docker/dockerfile:1.19

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

FROM ubuntu:26.04 AS base

ENV container=docker

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates \
        libcurl4t64 \
        dbus \
        munge \
        python3 \
        slurm-wlm \
        systemd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY --chmod=755 start.sh /usr/local/sbin/mqt-slurm-start
COPY node.conf /etc/systemd/system/slurmd.service.d/mqt-node.conf

RUN useradd --uid 10000 --user-group --create-home mqt-test \
    && systemctl disable slurmctld.service slurmd.service

RUN --mount=type=bind,from=dist,target=/tmp/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages --link-mode=copy /tmp/dist/mqt_core-*.whl

ARG SETUP_SCRIPT=""

RUN --mount=type=bind,from=dist,target=/tmp/dist \
    --mount=type=cache,target=/root/.cache/uv \
    set -- /tmp/dist/mqt_core-*.whl \
    && test "$#" -eq 1 && test -f "$1" && core_wheel="$1" \
    && printf 'mqt-core[qiskit,pennylane] @ file://%s\n' "$core_wheel" > /tmp/core-override.txt \
    && for directory in /tmp/dist/*/; do \
        [ -d "$directory" ] || continue; \
        test -z "$SETUP_SCRIPT" || { echo 'Use wheel runtimes or a source workload, not both.' >&2; exit 1; }; \
        directory="${directory%/}"; runtime="/opt/runtimes/${directory##*/}"; \
        set -- "$directory"/*.whl; \
        test "$#" -eq 1 && test -f "$1" \
        && uv venv --python /usr/bin/python3 "$runtime" \
        && uv pip install --python "$runtime/bin/python" --link-mode=copy \
            --overrides /tmp/core-override.txt "$core_wheel[qiskit,pennylane]" "$1[qiskit]" \
        || exit 1; \
    done \
    && rm /tmp/core-override.txt

FROM scratch AS provider-source

COPY --from=workload \
    --exclude=.git --exclude=.github --exclude=build --exclude=build-* --exclude=cmake-build-* \
    --exclude=.cache --exclude=.ccache --exclude=.venv* --exclude=.nox --exclude=CMakeUserPresets.json \
    --exclude=.ruff_cache --exclude=.rumdl_cache --exclude=.pytest_cache --exclude=.mypy_cache \
    --exclude=**/__pycache__ --exclude=dist --exclude=test/slurm/runtime --exclude=test/slurm/dist \
    . /workload

FROM ubuntu:26.04 AS provider-build

COPY --from=uv /uv /uvx /bin/

ARG SETUP_SCRIPT=""
RUN if [ -n "$SETUP_SCRIPT" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
            build-essential ca-certificates cmake git libcurl4-openssl-dev libssl-dev python3; \
    fi

ARG PROVIDER_RUNTIME_COMPONENT=""
RUN --mount=type=bind,from=provider-source,source=/workload,target=/workload \
    --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /opt/provider/wheels /opt/provider/native \
    && if [ -n "$SETUP_SCRIPT" ]; then \
        CMAKE_BUILD_PARALLEL_LEVEL=2 uv build /workload --wheel --out-dir /opt/provider/wheels \
            --config-setting=build-dir=/tmp/provider-build \
        && cmake --install /tmp/provider-build --prefix /opt/provider/native \
            --component "${PROVIDER_RUNTIME_COMPONENT:?Set the provider Runtime component}"; \
    fi

FROM base AS common

ARG PROVIDER_INSTALL_MODE="native"
RUN --mount=type=bind,from=dist,target=/tmp/dist \
    --mount=type=bind,from=provider-source,source=/workload,target=/workload \
    --mount=type=bind,from=provider-build,source=/opt/provider,target=/tmp/provider \
    --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$SETUP_SCRIPT" ]; then \
        case "$PROVIDER_INSTALL_MODE" in native|wheel) ;; *) exit 1 ;; esac \
        && set -- /tmp/provider/wheels/*.whl \
        && test "$#" -eq 1 && test -f "$1" && provider_wheel="$1[qiskit]" \
        && set -- /tmp/dist/mqt_core-*.whl \
        && test "$#" -eq 1 && test -f "$1" \
        && printf 'mqt-core[qiskit,pennylane] @ file://%s\n' "$1" > /tmp/core-override.txt \
        && uv pip install --system --break-system-packages --link-mode=copy \
            --overrides /tmp/core-override.txt "$provider_wheel" "$1[qiskit,pennylane]" \
        && if [ "$PROVIDER_INSTALL_MODE" = native ]; then cp -a /tmp/provider/native /opt/provider-native; fi \
        && /bin/sh "/workload/$SETUP_SCRIPT"; \
    fi

STOPSIGNAL SIGRTMIN+3
CMD ["/usr/local/sbin/mqt-slurm-start", "node"]
