# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM ubuntu:26.04
ENV container=docker UV_PYTHON_INSTALL_DIR=/opt/python
COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        ca-certificates libcurl4t64 dbus munge slurm-wlm slurmdbd systemd \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --uid 10000 --user-group --create-home mqt-dev \
    && systemctl disable slurmctld.service slurmd.service slurmdbd.service
STOPSIGNAL SIGRTMIN+3
RUN uv venv --python 3.14.8 /opt/afqmc \
    && uv pip install --python /opt/afqmc/bin/python \
        numpy==2.5.3 pennylane==0.45.1 pyscf==2.14.0 \
        openfermion==1.8.1 numba==0.68.0 scipy==1.18.1
COPY --from=core /python/mqt/core /opt/afqmc/lib/python3.14/site-packages/mqt/core
COPY --from=native / /opt/afqmc/lib/python3.14/site-packages/mqt/core/
COPY --from=metadata / /opt/afqmc/lib/python3.14/site-packages/
COPY --from=cluster --chmod=755 /start.sh /usr/local/sbin/mqt-slurm-start
COPY --from=cluster --chmod=755 /admission.sh /usr/local/sbin/mqt-qdmi-admission
COPY --from=cluster --chmod=755 /update_availability.py /usr/local/libexec/mqt-qdmi-availability.py
COPY --from=cluster /qdmi-*.service /slurm-accounting.service /etc/systemd/system/
COPY --from=cluster --chmod=755 /accounting.sh /usr/local/sbin/mqt-slurm-accounting
COPY --from=cluster /node.conf /qan.conf /slurmdbd.conf /usr/local/share/mqt-slurm/
COPY --from=cluster /profile.sh /etc/profile.d/mqt-qdmi.sh
ENV PATH=/opt/afqmc/bin:/opt/afqmc/lib/python3.14/site-packages/mqt/core/bin:/opt/venv/bin:$PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN rm -rf /opt/venv && ln -s /opt/afqmc /opt/venv
RUN uv pip install --python /opt/afqmc/bin/python nanobind-backend==1.0.0
# Run the unchanged monitor in the shared, pinned chemistry environment.
RUN sed -i '1c#!/opt/venv/bin/python' /usr/local/libexec/mqt-qdmi-availability.py
