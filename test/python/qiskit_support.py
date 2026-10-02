# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test selection from the compiler's Qiskit adapter registry."""

from __future__ import annotations

import os
import re
from pathlib import Path

import qiskit
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def supports_qiskit_translation(version: str = qiskit.__version__) -> bool:
    """Return whether the installed release has a registered test adapter."""
    if version == os.environ.get("MQT_QISKIT_TEST_CANDIDATE_VERSION"):
        return True
    installed = Version(version)
    registry = Path(__file__).resolve().parents[2] / "bindings/mlir/qiskit/SupportedVersions.inc"
    ranges = re.findall(r'^MQT_QISKIT_VERSION\([^\n]+"([^"\n]+)"\)', registry.read_text(), re.MULTILINE)
    return (
        not installed.is_prerelease
        and not installed.is_devrelease
        and installed.local is None
        and any(installed in SpecifierSet(supported) for supported in ranges)
    )
