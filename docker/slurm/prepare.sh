#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -eu
umask 077
runtime=${1:-"$(dirname "$0")/../../build/slurm"}
mkdir -p "$(dirname "$runtime")"
mkdir "$runtime"
head -c 1024 /dev/urandom > "$runtime/munge.key"
mkdir "$runtime/jobs"
chmod 1777 "$runtime/jobs"
cp "$(dirname "$0")/slurm.conf" "$runtime/slurm.conf"
touch "$runtime/plugstack.conf"
chmod 644 "$runtime/slurm.conf" "$runtime/plugstack.conf"
