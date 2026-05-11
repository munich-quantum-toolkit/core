# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""IQM-specific QDMI integration for Qiskit.

This module provides an MQT Core-native wrapper for the externally packaged
``iqm-qdmi`` device library. Importing this module is opt-in and requires the
``mqt-core[iqm]`` extra. The library path is ``iqm.qdmi.IQM_QDMI_LIBRARY_PATH``.
"""

from __future__ import annotations

import os
from typing import Any

from ... import fomac
from ..._compat.optional import OptionalDependencyTester
from .backend import QDMIBackend
from .estimator import QDMIEstimator
from .sampler import QDMISampler

HAS_IQM_QDMI = OptionalDependencyTester(
    "iqm.qdmi",
    install_msg="Install with 'pip install mqt-core[iqm]'",
)
HAS_QISKIT_IQM = OptionalDependencyTester(
    "qiskit",
    install_msg="Install with 'pip install mqt-core[iqm]'",
)

__all__ = ["HAS_IQM_QDMI", "IQMBackend"]

HAS_QISKIT_IQM.require_now("use the IQM QDMI backend")
HAS_IQM_QDMI.require_now("use the IQM QDMI backend")


def __dir__() -> list[str]:
    return __all__


def _iqm_qdmi_library_path() -> str:
    """Return the packaged IQM QDMI device library path exported by ``iqm-qdmi``."""
    iqm_qdmi_module = HAS_IQM_QDMI.require_module("use the IQM QDMI backend")
    return str(iqm_qdmi_module.IQM_QDMI_LIBRARY_PATH)


class IQMBackend(QDMIBackend):
    """Qiskit backend for the packaged IQM QDMI device library.

    This backend loads the shared library distributed with ``iqm-qdmi`` and
    exposes it through MQT Core's Qiskit-compatible QDMI backend surface.

    Args:
        base_url: Base URL of the IQM service. Defaults to ``IQM_BASE_URL`` or
            the standard Resonance endpoint.
        token: Authentication token. Defaults to ``IQM_TOKEN`` or
            ``RESONANCE_API_KEY``.
        tokens_file: Path to an authentication file. Defaults to
            ``IQM_TOKENS_FILE``.
        qc_id: Optional IQM quantum computer identifier.
        qc_alias: Optional IQM quantum computer alias.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        tokens_file: str | None = None,
        qc_id: str | None = None,
        qc_alias: str | None = None,
    ) -> None:
        """Initialize the IQM backend."""
        device = fomac.add_dynamic_device_library(
            library_path=_iqm_qdmi_library_path(),
            prefix="IQM",
            base_url=base_url or os.getenv("IQM_BASE_URL") or "https://resonance.iqm.tech",
            token=token or os.getenv("IQM_TOKEN") or os.getenv("RESONANCE_API_KEY"),
            auth_file=tokens_file or os.getenv("IQM_TOKENS_FILE"),
            custom1=qc_id or os.getenv("IQM_QC_ID"),
            custom2=qc_alias or os.getenv("IQM_QC_ALIAS"),
        )
        super().__init__(device=device)

    def sampler(
        self,
        *,
        default_shots: int = 1024,
        options: dict[str, Any] | None = None,
    ) -> QDMISampler:
        """Return a SamplerV2 primitive bound to this backend."""
        return QDMISampler(self, default_shots=default_shots, options=options)

    def estimator(
        self,
        *,
        default_precision: float = 0.0,
        options: dict[str, Any] | None = None,
    ) -> QDMIEstimator:
        """Return an EstimatorV2 primitive bound to this backend."""
        return QDMIEstimator(self, default_precision=default_precision, options=options)
