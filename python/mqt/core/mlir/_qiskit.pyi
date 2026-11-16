# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native QDMI readout for Qiskit."""

import mqt.core.mlir
import mqt.core.qdmi

class ResultDecodingError(ValueError):
    """The provider result violates its declared shot count or output layout."""

def readout(
    job: mqt.core.qdmi.Job,
    width: int,
    shots: int,
    program_index: int,
    layout: mqt.core.mlir.ResultLayout | None,
    *,
    memory: bool,
) -> dict[str, object]:
    """Return Qiskit hexadecimal counts and optional ordered memory.

    Args:
        job: Completed provider job whose results are read.
        width: Number of logical classical output bits.
        shots: Expected number of shots.
        program_index: Program within a multi-program job.
        layout: Projection from provider output to logical classical bits.
        memory: Read ordered SHOTS and derive counts from those same samples.
            When false, read the provider histogram instead.

    Returns:
        A dictionary with ``counts`` and, when requested, ``memory``. Keys use
        lowercase hexadecimal strings without a machine-word width limit.

    Raises:
        ResultDecodingError: Results violate the declared shot count or layout.
        RuntimeError: Provider result retrieval fails. Ordered memory requires
            SHOTS support and never falls back to a histogram.
    """
