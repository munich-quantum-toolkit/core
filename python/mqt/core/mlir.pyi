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
from typing import Literal, TypeAlias, overload

import qiskit

import mqt.core.ir

InputProgram: TypeAlias = (
    str
    | os.PathLike[str]
    | mqt.core.ir.QuantumComputation
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
)

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

    QC = 1
    """QC after the optimized QCO round trip."""

    QCO = 2
    """Optimized QCO."""

    JEFF = 3
    """Serializable Jeff MLIR."""

    QIR_BASE = 4
    """QIR for the Base Profile."""

    QIR_ADAPTIVE = 5
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
    def from_mlir_file(path: str | os.PathLike[str]) -> QCProgram: ...
    @staticmethod
    def from_qasm_str(source: str) -> QCProgram: ...
    @staticmethod
    def from_qasm_file(path: str | os.PathLike[str]) -> QCProgram: ...
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
    def from_mlir_file(path: str | os.PathLike[str]) -> QCOProgram: ...
    def copy(self) -> QCOProgram: ...
    def cleanup(self) -> None: ...
    def optimize(self, *, merge_single_qubit_rotations: bool = True, enable_hadamard_lifting: bool = False) -> None: ...
    def to_qc(self, *, copy: bool = False) -> QCProgram: ...
    def to_jeff(self, *, copy: bool = False) -> JeffProgram: ...

class JeffProgram(Program):
    @staticmethod
    def from_file(path: str | os.PathLike[str]) -> JeffProgram: ...
    @staticmethod
    def from_bytes(data: bytes) -> JeffProgram: ...
    def copy(self) -> JeffProgram: ...
    def cleanup(self) -> None: ...
    def to_bytes(self) -> bytes: ...
    def write(self, path: str | os.PathLike[str]) -> None: ...
    def to_qco(self, *, copy: bool = False) -> QCOProgram: ...

class QIRProgram(Program):
    def copy(self) -> QIRProgram: ...
    def cleanup(self) -> None: ...
    @property
    def profile(self) -> QIRProfile: ...
    @property
    def llvm_ir(self) -> str: ...

@overload
def compile_program(
    program: InputProgram,
    *,
    output: Literal[OutputFormat.QC_IMPORT, OutputFormat.QC] = ...,
    inplace: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCProgram:
    """Run the coordinated default MQT compiler pipeline.

    Input source strings, files, MQT `QuantumComputation` objects, Qiskit circuits,
    and typed compiler programs can be combined with any supported output format.
    Typed program inputs are copied by default; set `inplace=True` to consume them.
    Use the typed programs directly to construct a custom pipeline stage by stage.
    """

@overload
def compile_program(
    program: InputProgram,
    *,
    output: Literal[OutputFormat.QCO],
    inplace: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCOProgram: ...
@overload
def compile_program(
    program: InputProgram,
    *,
    output: Literal[OutputFormat.JEFF],
    inplace: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> JeffProgram: ...
@overload
def compile_program(
    program: InputProgram,
    *,
    output: Literal[OutputFormat.QIR_BASE, OutputFormat.QIR_ADAPTIVE],
    inplace: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QIRProgram: ...
@overload
def compile_program(
    program: InputProgram,
    *,
    output: OutputFormat,
    inplace: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCProgram | QCOProgram | JeffProgram | QIRProgram: ...
