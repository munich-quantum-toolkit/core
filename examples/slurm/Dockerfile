# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM ubuntu:26.04 AS slurm-build

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential ca-certificates curl pkg-config libdbus-1-dev libhwloc-dev \
        libjson-c-dev liblua5.4-dev libmariadb-dev libmunge-dev libpam0g-dev \
        libpmix-dev libreadline-dev libyaml-dev \
    && rm -rf /var/lib/apt/lists/*
# HRES Lua is still a development interface.
RUN curl -fLsS https://github.com/SchedMD/slurm/archive/ba178ef837c4ba799fb3de72bcdaf30bc2b1616d.tar.gz -o /tmp/slurm.tar.gz \
    && echo '7a355fb32825af3b3b84761c3e113aafe52482c8b729fffb398fc7a041703bf5  /tmp/slurm.tar.gz' | sha256sum -c - \
    && mkdir /src \
    && tar -xzf /tmp/slurm.tar.gz -C /src --strip-components=1
WORKDIR /src
RUN ./configure --prefix=/opt/slurm --sysconfdir=/etc/slurm \
        --with-systemdsystemunitdir=/etc/systemd/system --with-pmix \
    && make -j4 \
    && make install DESTDIR=/install \
    && test -f /install/opt/slurm/lib/slurm/hres_lua.so

FROM ubuntu:26.04 AS python

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/
ARG UV_LINK_MODE=copy
ARG UV_PYTHON_CACHE_DIR=/root/.cache/uv/python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.14 --compile-bytecode \
    && uv venv --python 3.14 /opt/venv

FROM python AS build-tools

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential clang lld cmake curl git libcurl4-openssl-dev libssl-dev zstd \
    && rm -rf /var/lib/apt/lists/*
ARG CMAKE_BUILD_PARALLEL_LEVEL=2

FROM scratch AS core-source

COPY --from=core \
    --exclude=**/.git --exclude=**/.github --exclude=**/.env* --exclude=**/build --exclude=**/build-* --exclude=**/cmake-build-* \
    --exclude=**/.cache --exclude=**/.ccache --exclude=**/.venv* --exclude=**/.nox --exclude=**/CMakeUserPresets.json \
    --exclude=**/.ruff_cache --exclude=**/.rumdl_cache --exclude=**/.pytest_cache --exclude=**/.mypy_cache \
    --exclude=**/__pycache__ --exclude=**/dist --exclude=.agent --exclude=docs --exclude=test --exclude=examples \
    . /core

FROM build-tools AS core-build

ARG TARGETARCH

RUN mkdir -p /tmp/setup-mlir \
    && curl -fLsS https://github.com/munich-quantum-software/setup-mlir/archive/246aa0c70414109aa55fa19a5370739519833512.tar.gz \
        | tar -xz -C /tmp/setup-mlir --strip-components=1 \
    && bash /tmp/setup-mlir/installation/setup-mlir.sh -v 23.1.4 -p /opt/llvm -a OFF \
    && rm -rf /tmp/setup-mlir
# Source snapshots omit Git metadata; give the private wheel an explicit version.
RUN --mount=type=bind,from=core-source,source=/core,target=/core,rw \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,id=mqt-slurm-core-${TARGETARCH},target=/tmp/core-build,sharing=locked \
    CC=clang CXX=clang++ MLIR_DIR=/opt/llvm/lib/cmake/mlir \
    VCS_VERSIONING_PRETEND_VERSION_FOR_MQT_CORE=4.1.0 \
    uv build /core --wheel --out-dir /opt/core-wheel -Cbuild-dir=/tmp/core-build \
        -Ccmake.define.DEPLOY=ON -Ccmake.define.CMAKE_LINKER_TYPE=LLD

FROM scratch AS workload-source

COPY --from=workload \
    --exclude=**/.git --exclude=**/.github --exclude=**/.env* --exclude=**/build --exclude=**/build-* --exclude=**/cmake-build-* \
    --exclude=**/.cache --exclude=**/.ccache --exclude=**/.venv* --exclude=**/.nox --exclude=**/CMakeUserPresets.json \
    --exclude=**/.ruff_cache --exclude=**/.rumdl_cache --exclude=**/.pytest_cache --exclude=**/.mypy_cache \
    --exclude=**/__pycache__ --exclude=**/dist --exclude=test/slurm/runtime --exclude=test/slurm/dist \
    . /workload

FROM build-tools AS environment

ARG PACKAGES=""
RUN --mount=type=bind,from=core-build,source=/opt/core-wheel,target=/tmp/core-wheel \
    --mount=type=bind,from=workload-source,source=/workload,target=/workload,rw \
    --mount=type=cache,target=/root/.cache/uv \
    set -- /tmp/core-wheel/*.whl \
    && printf 'mqt-core[qiskit,pennylane] @ file://%s\n' "$1" > /tmp/core-override.txt \
    && printf '%s[qiskit,pennylane]\n%s\n' "$1" "$PACKAGES" > /tmp/packages.txt \
    && uv pip install --compile-bytecode --overrides /tmp/core-override.txt -r /tmp/packages.txt

FROM python AS runtime

ENV container=docker PATH=/opt/slurm/bin:/opt/slurm/sbin:$PATH
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        libcurl4t64 dbus munge systemd libhwloc15 libjson-c5 liblua5.4-0 \
        libmariadb3 libnuma1 libopenmpi40 libpmix2t64 libreadline8t64 libyaml-0-2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=slurm-build /install/ /
COPY --from=environment /opt/venv /opt/venv
COPY --chmod=755 start.sh /usr/local/sbin/mqt-slurm-start
COPY --chmod=755 probe.py /usr/local/libexec/mqt-qdmi-probe
COPY hres_qdmi.lua /etc/slurm/hres_qdmi.lua
COPY slurmctld.conf /etc/systemd/system/slurmctld.service.d/qdmi.conf
COPY slurm-accounting.service /etc/systemd/system/
COPY --chmod=755 accounting.sh /usr/local/sbin/mqt-slurm-accounting
COPY node.conf slurmdbd.conf /usr/local/share/mqt-slurm/
COPY profile.sh /etc/profile.d/mqt-qdmi.sh
RUN useradd --uid 10000 --user-group --create-home mqt-dev \
    && useradd --system --user-group --home-dir /var/lib/slurm slurm \
    && systemctl disable slurmctld.service slurmd.service slurmdbd.service

STOPSIGNAL SIGRTMIN+3
CMD ["/usr/local/sbin/mqt-slurm-start", "node"]
