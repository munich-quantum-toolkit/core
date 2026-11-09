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

class ResultDecodingError(ValueError): ...

def readout(
    job: mqt.core.qdmi.Job,
    width: int,
    shots: int,
    program_index: int,
    layout: mqt.core.mlir.ResultLayout | None,
    *,
    memory: bool,
) -> dict[str, object]: ...
