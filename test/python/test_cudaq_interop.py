# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Live CUDA-Q interoperability smoke test."""

from __future__ import annotations

import pytest

from mqt.core.mlir import QuakeProgram

cudaq = pytest.importorskip("cudaq", reason="CUDA-Q is an optional dependency")


def test_live_cudaq_quake_round_trip() -> None:
    """Import a synthesized kernel and execute MQT-emitted Quake in CUDA-Q."""

    @cudaq.kernel
    def bell() -> None:
        qubits = cudaq.qvector(2)
        h(qubits[0])  # ruff: ignore[undefined-name]  # ty: ignore[unresolved-reference]
        x.ctrl(qubits[0], qubits[1])  # ruff: ignore[undefined-name]  # ty: ignore[unresolved-reference]
        mz(qubits)  # ruff: ignore[undefined-name]  # ty: ignore[unresolved-reference]

    synthesized = str(cudaq.synthesize(bell))
    quake = QuakeProgram.from_mlir_str(synthesized)
    qc = quake.to_qc()
    emitted = qc.to_quake(name="mqt_bell")

    @cudaq.kernel
    def merge_anchor() -> None:
        pass

    merged = merge_anchor.merge_quake_source(emitted.ir)
    # CUDA-Q 0.15.1 keeps the merge anchor's unique launch name even though it
    # selects the newly merged entry point. Align the launch name with that
    # selected kernel until CUDA-Q preserves it itself.
    if merged.uniqName != merged.name:
        merged.uniqName = merged.name
    counts = cudaq.sample(merged, shots_count=100)

    assert counts
    assert all(bits in {"00", "11"} for bits in counts)
