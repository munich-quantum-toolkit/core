# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""IQM-specific QDMI integration for MQT Core.

This module exposes the upstream IQM backend provided by the external
``iqm-qdmi`` package as MQT Core's IQM device integration.
"""

from __future__ import annotations

from .._compat.optional import OptionalDependencyTester

HAS_IQM_QDMI = OptionalDependencyTester(
    "iqm.qdmi.qiskit",
    install_msg="Install with 'pip install mqt-core[iqm]'",
)

if HAS_IQM_QDMI:
    from iqm.qdmi.qiskit import IQMBackend

    __all__ = ["IQMBackend"]
else:
    __all__ = []
