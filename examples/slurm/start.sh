#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -eu

case "${1:-node}" in
    controller) daemon=slurmctld ;;
    node) daemon=slurmd ;;
    *) echo "Expected controller or node" >&2; exit 1 ;;
esac

test -s /runtime/munge.key

install -d -o munge -g munge -m 0755 /etc/munge /run/munge
install -o munge -g munge -m 0400 /runtime/munge.key /etc/munge/munge.key
install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld /var/log/slurm
install -d -o root -g root -m 0755 /var/spool/slurmd
systemctl enable munge.service "$daemon.service"
if [ "$daemon" = slurmctld ]; then
    systemctl enable qdmi-admission.service
fi
exec /usr/lib/systemd/systemd
