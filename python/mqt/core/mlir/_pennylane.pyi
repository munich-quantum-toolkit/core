# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Runtime support for the PennyLane QDMI device."""

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

import mqt.core.mlir
import mqt.core.qdmi
from mqt.core.mlir._qdmi import Context as _ExecutionContext

class Context(_ExecutionContext):
    """Retain a QDMI device, target snapshot, and job parameters."""

    def __init__(
        self,
        device: mqt.core.qdmi.Device,
        *,
        custom1: str | bool | float | bytes | None = None,
        custom2: str | bool | float | bytes | None = None,
        custom3: str | bool | float | bytes | None = None,
        custom4: str | bool | float | bytes | None = None,
        custom5: str | bool | float | bytes | None = None,
        program_format: mqt.core.qdmi.ProgramFormat | None = None,
    ) -> None:
        """Create a context for native compilation and submission.

        Args:
            device: Open QDMI device retained for the context lifetime.
            custom1: First provider-specific job parameter.
            custom2: Second provider-specific job parameter.
            custom3: Third provider-specific job parameter.
            custom4: Fourth provider-specific job parameter.
            custom5: Fifth provider-specific job parameter.
            program_format: Preferred program format, or automatic selection.

        Byte parameters are copied so later submissions do not borrow Python storage.
        """

    @property
    def id(self) -> int:
        """Identifier used by Catalyst to retrieve this retained context."""

def aggregate(
    job: mqt.core.qdmi.Job,
    num_wires: int,
    shots: int,
    measurements: Sequence[tuple[Sequence[int], bool]],
    program_index: int = 0,
    result_layout: mqt.core.mlir.ResultLayout | None = None,
) -> list[object]:
    """Compute measurement marginals from validated provider readout.

    Args:
        job: Completed provider job whose results are read.
        num_wires: Logical output width after layout projection.
        shots: Positive expected shot count.
        measurements: Pairs of program wire indices and a counts flag. True
            selects a binary-key count dictionary; false selects a one-dimensional
            NumPy probability array. The first selected wire is most significant.
        program_index: Program within a multi-program job.
        result_layout: Projection from provider output to program wires.

    Returns:
        One marginal per measurement, in the requested order. Counts are preferred;
        unavailable or empty histograms fall back to shots. Malformed results fail
        validation without fallback. Observable eigenvalues are applied by Python.
    """

def samples(
    job: mqt.core.qdmi.Job,
    num_wires: int,
    shots: int,
    program_index: int = 0,
    result_layout: mqt.core.mlir.ResultLayout | None = None,
) -> Annotated[NDArray[np.int8], {"shape": (None, None), "order": "C", "device": "cpu"}]:
    """Return a shot-by-wire NumPy array in program wire order.

    Args:
        job: Completed provider job whose results are read.
        num_wires: Number of logical output columns after layout projection.
        shots: Positive expected number of rows.
        program_index: Program within a multi-program job.
        result_layout: Projection from provider output to program wires.

    Returns:
        A contiguous int8 array that owns its storage independently of the job.
        Ordered SHOTS are preferred; unavailable or empty shots fall back to
        histogram expansion, which preserves frequencies but has no shot order.
        Malformed results fail validation without fallback.
    """
