#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -eu

role=${1:-node}
case "$role" in
    controller) set -- slurmctld.service qdmi-admission.service ;;
    accounting)
        set -- slurmdbd.service slurm-accounting.service
        install -o slurm -g slurm -m 0600 /usr/local/share/mqt-slurm/slurmdbd.conf /etc/slurm/slurmdbd.conf
        printf 'StoragePass=%s\n' "$(cat /run/secrets/accounting_password)" >> /etc/slurm/slurmdbd.conf
        ;;
    node|qan)
        set -- slurmd.service
        install -D -m 0644 "/usr/local/share/mqt-slurm/$role.conf" /etc/systemd/system/slurmd.service.d/mqt-node.conf
        ;;
    login) set -- ;;
    *) echo "Expected controller, accounting, login, node, or qan" >&2; exit 1 ;;
esac

test -s /runtime/munge.key
install -d -o munge -g munge -m 0755 /etc/munge /run/munge
install -o munge -g munge -m 0400 /runtime/munge.key /etc/munge/munge.key
install -d -o slurm -g slurm -m 0755 /var/spool/slurmctld /var/log/slurm
install -d -o root -g root -m 0755 /var/spool/slurmd
systemctl enable munge.service "$@"
exec /usr/lib/systemd/systemd
