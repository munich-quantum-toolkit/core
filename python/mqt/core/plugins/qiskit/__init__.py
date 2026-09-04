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
    from .backend import QDMIBackend
    from .exceptions import (
        CircuitValidationError,
        JobSubmissionError,
        QDMIQiskitError,
        TranslationError,
        UnsupportedFormatError,
        UnsupportedOperationError,
    )
    from .job import QDMIJob
    from .provider import QDMIProvider
    from .serializers import (
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
        "program_serializer",
        "register_program_serializer",
        "unregister_program_serializer",
    ]
