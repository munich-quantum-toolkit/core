# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Qiskit Plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._compat.optional import OptionalDependencyTester

# Optional dependency tester for Qiskit
HAS_QISKIT = OptionalDependencyTester(
    "qiskit",
    install_msg="Install with 'pip install mqt-core[qiskit]'",
)

if TYPE_CHECKING or HAS_QISKIT:
    from .backend import QiskitBackend
    from .exceptions import (
        CircuitValidationError,
        JobSubmissionError,
        QDMIQiskitError,
        TranslationError,
        UnsupportedFormatError,
        UnsupportedOperationError,
    )
    from .mqt_to_qiskit import mqt_to_qiskit
    from .provider import QDMIProvider
    from .qiskit_to_mqt import qiskit_to_mqt


__all__ = [
    "HAS_QISKIT",
]

if TYPE_CHECKING or HAS_QISKIT:
    __all__ += [
        "CircuitValidationError",
        "JobSubmissionError",
        "QDMIProvider",
        "QDMIQiskitError",
        "QiskitBackend",
        "TranslationError",
        "UnsupportedFormatError",
        "UnsupportedOperationError",
        "mqt_to_qiskit",
        "qiskit_to_mqt",
    ]
