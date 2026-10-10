#!/bin/sh
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -eu

while /usr/local/libexec/mqt-qdmi-availability.py --license "$1"; do
    sleep 30
done
exit 1
