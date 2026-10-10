#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -eu

licenses=$(awk -F= '/^Licenses=/ {gsub(/,/, " ", $2); print $2}' /etc/slurm/slurm.conf)
for license in $licenses; do
    /usr/local/libexec/mqt-qdmi-availability.py --license "$license" --block-only
done
for license in $licenses; do
    systemctl start "qdmi-availability@$license.service"
done
for partition in compute iqm braket; do
    scontrol update "PartitionName=$partition" State=UP
done
