# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
HAS_QISKIT = OptionalDependencyTester(  # noqa: RUF067 Used for handling optional plugin
    "qiskit",
    install_msg="Install with 'pip install mqt-core[qiskit]'",
)

__all__ = [
    "HAS_QISKIT",
]

if TYPE_CHECKING or HAS_QISKIT:  # noqa: RUF067 Used for handling optional plugin
    from .backend import QDMIBackend
    from .converters import qiskit_to_iqm_json
    from .estimator import QDMIEstimator
    from .exceptions import (
        CircuitValidationError,
        JobSubmissionError,
        QDMIQiskitError,
        TranslationError,
        UnsupportedFormatError,
        UnsupportedOperationError,
    )
    from .job import QDMIJob
    from .mqt_to_qiskit import mqt_to_qiskit
    from .provider import QDMIProvider
    from .qiskit_to_mqt import qiskit_to_mqt
    from .sampler import QDMISampler

    __all__ += [
        "CircuitValidationError",
        "JobSubmissionError",
        "QDMIBackend",
        "QDMIEstimator",
        "QDMIJob",
        "QDMIProvider",
        "QDMIQiskitError",
        "QDMISampler",
        "TranslationError",
        "UnsupportedFormatError",
        "UnsupportedOperationError",
        "mqt_to_qiskit",
        "qiskit_to_iqm_json",
        "qiskit_to_mqt",
    ]
