# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core MLIR compiler bindings."""

import enum
import os
from collections.abc import Sequence
from typing import Literal, overload

import qiskit

import mqt.core.ir

class QIRProfile(enum.Enum):
    """QIR target profiles."""

    BASE = 0
    """The QIR Base Profile."""

    ADAPTIVE = 1
    """The QIR Adaptive Profile."""

class OutputFormat(enum.Enum):
    """Default compiler output formats."""

    QC_IMPORT = 0
    """QC directly after frontend import."""

    QCO = 1
    """QCO immediately after conversion, before optimization."""

    QCO_OPTIMIZED = 2
    """QCO after the configured optimization pipeline."""

    QC = 3
    """QC after the optimized QCO round trip."""

    JEFF = 4
    """Serializable Jeff MLIR."""

    QIR_BASE = 5
    """QIR for the Base Profile."""

    QIR_ADAPTIVE = 6
    """QIR for the Adaptive Profile."""

class Program:
    @property
    def is_valid(self) -> bool: ...
    @property
    def ir(self) -> str: ...

class QCProgram(Program):
    @staticmethod
    def from_mlir_str(source: str) -> QCProgram: ...
    @staticmethod
    def from_mlir_file(path: str | os.PathLike) -> QCProgram: ...
    @staticmethod
    def from_qasm_str(source: str) -> QCProgram: ...
    @staticmethod
    def from_qasm_file(path: str | os.PathLike) -> QCProgram: ...
    @staticmethod
    def from_quantum_computation(computation: mqt.core.ir.QuantumComputation) -> QCProgram: ...
    @staticmethod
    def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) -> QCProgram: ...
    def copy(self) -> QCProgram: ...
    def cleanup(self) -> None: ...
    def to_qco(self, *, copy: bool = False) -> QCOProgram: ...
    def to_qir(self, profile: QIRProfile, *, copy: bool = False) -> QIRProgram: ...

class QCOProgram(Program):
    @staticmethod
    def from_mlir_str(source: str) -> QCOProgram: ...
    @staticmethod
    def from_mlir_file(path: str | os.PathLike) -> QCOProgram: ...
    def copy(self) -> QCOProgram: ...
    def cleanup(self) -> None: ...
    def run_pass_pipeline(
        self, pipeline: str, *, enable_timing: bool = False, enable_statistics: bool = False
    ) -> None: ...
    def merge_single_qubit_rotation_gates(self) -> None: ...
    def fuse_single_qubit_unitary_runs(self, *, basis: str = "zyz") -> None: ...
    def unroll_quantum_loops(self, *, unroll_factor: int = -1) -> None: ...
    def lift_hadamards(self) -> None: ...
    def place_and_route(
        self,
        coupling: Sequence[tuple[int, int]],
        *,
        nlookahead: int = 1,
        alpha: float = 1.0,
        lambda_: float = 0.5,
        niterations: int = 1,
        ntrials: int = 4,
        seed: int = 42,
    ) -> None: ...
    def to_qc(self, *, copy: bool = False) -> QCProgram: ...
    def to_jeff(self, *, copy: bool = False) -> JeffProgram: ...

class JeffProgram(Program):
    @staticmethod
    def from_file(path: str | os.PathLike) -> JeffProgram: ...
    @staticmethod
    def from_bytes(data: bytes) -> JeffProgram: ...
    def copy(self) -> JeffProgram: ...
    def cleanup(self) -> None: ...
    def to_bytes(self) -> bytes: ...
    def write(self, path: str | os.PathLike) -> None: ...
    def to_qco(self, *, copy: bool = False) -> QCOProgram: ...

class QIRProgram(Program):
    def copy(self) -> QIRProgram: ...
    def cleanup(self) -> None: ...
    @property
    def profile(self) -> QIRProfile: ...
    @property
    def llvm_ir(self) -> str: ...
    def to_bitcode(self) -> bytes: ...
    def write_bitcode(self, path: str | os.PathLike) -> None: ...

@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | mqt.core.ir.QuantumComputation
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram,
    *,
    output: Literal[OutputFormat.QC, OutputFormat.QC_IMPORT] = ...,
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCProgram: ...
@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | mqt.core.ir.QuantumComputation
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram,
    *,
    output: Literal[OutputFormat.QCO, OutputFormat.QCO_OPTIMIZED],
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCOProgram: ...
@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | mqt.core.ir.QuantumComputation
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram,
    *,
    output: Literal[OutputFormat.JEFF],
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> JeffProgram: ...
@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | mqt.core.ir.QuantumComputation
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram,
    *,
    output: Literal[OutputFormat.QIR_BASE, OutputFormat.QIR_ADAPTIVE],
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QIRProgram: ...
@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | mqt.core.ir.QuantumComputation
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram,
    *,
    output: OutputFormat,
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCProgram | QCOProgram | JeffProgram | QIRProgram:
    """Run the coordinated default MQT compiler pipeline.

    Input source strings, files, MQT `QuantumComputation` objects, Qiskit circuits,
    and typed compiler programs can be combined with any supported output format.
    Typed program inputs are copied by default; set `inplace=True` to consume them.
    Use the typed programs directly to construct a custom pipeline stage by stage.
    """
