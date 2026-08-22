# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv

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
COPY dist/*.whl /tmp/dist/
COPY mqt-slurm-prepare /usr/local/sbin/mqt-slurm-prepare
COPY mqt-slurm-prepare.service /etc/systemd/system/mqt-slurm-prepare.service

RUN uv pip install --system --break-system-packages /tmp/dist/*.whl \
    && rm -rf /root/.cache/uv /tmp/dist \
    && chmod 0755 /usr/local/sbin/mqt-slurm-prepare \
    && systemctl enable mqt-slurm-prepare.service munge.service

STOPSIGNAL SIGRTMIN+3

CMD ["/usr/lib/systemd/systemd"]

FROM common AS controller

RUN systemctl enable slurmctld.service

FROM common AS compute

RUN systemctl enable slurmd.service
