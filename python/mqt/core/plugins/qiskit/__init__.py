# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Qiskit Plugin."""

# ruff: file-ignore[non-empty-init-module]

from __future__ import annotations

__lazy_modules__ = {
    "mqt.core.plugins.qiskit.backend",
    "mqt.core.plugins.qiskit.exceptions",
    "mqt.core.plugins.qiskit.job",
    "mqt.core.plugins.qiskit.provider",
    "mqt.core.plugins.qiskit.serializers",
}

from importlib import import_module
from typing import TYPE_CHECKING

try:
    import_module("qiskit")
except ModuleNotFoundError as error:
    if error.name != "qiskit":
        raise
    HAS_QISKIT = False
else:
    HAS_QISKIT = True

__all__ = [
    "HAS_QISKIT",
]

if TYPE_CHECKING or HAS_QISKIT:
    from mqt.core.plugins.qiskit.backend import QDMIBackend
    from mqt.core.plugins.qiskit.exceptions import (
        CircuitValidationError,
        JobSubmissionError,
        QDMIQiskitError,
        TranslationError,
        UnsupportedFormatError,
        UnsupportedOperationError,
    )
    from mqt.core.plugins.qiskit.job import QDMIJob
    from mqt.core.plugins.qiskit.mqt_to_qiskit import mqt_to_qiskit
    from mqt.core.plugins.qiskit.provider import QDMIProvider
    from mqt.core.plugins.qiskit.qiskit_to_mqt import qiskit_to_mqt
    from mqt.core.plugins.qiskit.serializers import (
        ProgramSerializer,
        program_serializer,
        register_program_serializer,
        unregister_program_serializer,
    )

    __all__ += [
        "CircuitValidationError",
        "JobSubmissionError",
        "ProgramSerializer",
        "QDMIBackend",
        "QDMIJob",
        "QDMIProvider",
        "QDMIQiskitError",
        "TranslationError",
        "UnsupportedFormatError",
        "UnsupportedOperationError",
        "mqt_to_qiskit",
        "program_serializer",
        "qiskit_to_mqt",
        "register_program_serializer",
        "unregister_program_serializer",
    ]
