#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Extract a portable SDK with its documented 2 GiB zstd window limit (Python 3.14)."""

import argparse
import tarfile
from pathlib import Path

from compression.zstd import DecompressionParameter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
with tarfile.open(args.archive, "r:zst", options={DecompressionParameter.window_log_max: 31}) as archive:
    archive.extractall(args.destination, filter="data")
