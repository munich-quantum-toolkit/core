# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Runtime support for the PennyLane QDMI device."""

from typing import Annotated

import numpy as np
from numpy.typing import NDArray

import mqt.core.mlir
import mqt.core.qdmi

class Context:
    def __init__(
        self,
        device: mqt.core.qdmi.Device,
        *,
        custom1: str | bool | float | bytes | None = None,
        custom2: str | bool | float | bytes | None = None,
        custom3: str | bool | float | bytes | None = None,
        custom4: str | bool | float | bytes | None = None,
        custom5: str | bool | float | bytes | None = None,
        profile: mqt.core.mlir.SerializationProfile = mqt.core.mlir.SerializationProfile.DEFAULT,
    ) -> None: ...
    @property
    def id(self) -> int: ...

def samples(
    job: mqt.core.qdmi.Job,
    num_wires: int,
    shots: int,
    program_index: int = 0,
    result_layout: mqt.core.mlir.ResultLayout | None = None,
) -> Annotated[NDArray[np.int8], {"shape": (None, None), "order": "C", "device": "cpu"}]: ...
