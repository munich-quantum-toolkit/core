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
    """Serializable ``jeff`` MLIR."""

    QIR_BASE = 5
    """QIR for the Base Profile."""

    QIR_ADAPTIVE = 6
    """QIR for the Adaptive Profile."""

class Program:
    """Base class for a typed MLIR compiler program.

    Programs own their MLIR module. Conversions can consume a program; use
    ``is_valid`` to check whether it can still be used.
    """

    @property
    def is_valid(self) -> bool:
        """Whether this program still owns its module."""

    @property
    def ir(self) -> str:
        """The textual MLIR representation of this program."""

class QCProgram(Program):
    """A compiler program in the QC dialect.

    QC programs use reference semantics and represent frontend quantum programs
    before conversion to QCO.
    """

    @staticmethod
    def from_mlir_str(source: str) -> QCProgram:
        """Parse a QC MLIR source string."""

    @staticmethod
    def from_mlir_file(path: str | os.PathLike) -> QCProgram:
        """Parse QC MLIR from a file."""

    @staticmethod
    def from_qasm_str(source: str) -> QCProgram:
        """Translate an OpenQASM 3 source string to QC MLIR."""

    @staticmethod
    def from_qasm_file(path: str | os.PathLike) -> QCProgram:
        """Translate an OpenQASM 3 file to QC MLIR."""

    @staticmethod
    def from_quantum_computation(computation: mqt.core.ir.QuantumComputation) -> QCProgram:
        """Translate an MQT ``QuantumComputation`` to QC MLIR."""

    @staticmethod
    def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) -> QCProgram:
        """Translate a Qiskit ``QuantumCircuit`` to QC MLIR."""

    def copy(self) -> QCProgram:
        """Return an independent copy of this program."""

    def cleanup(self) -> None:
        """Run the standard QC cleanup pipeline in place."""

    def to_qco(self, *, copy: bool = False) -> QCOProgram:
        """Convert this program to QCO.

        Set ``copy=True`` to preserve it.
        """

    def to_qir(self, profile: QIRProfile, *, copy: bool = False) -> QIRProgram:
        """Lower this program to QIR for the requested profile.

        Set ``copy=True`` to preserve it.
        """

class QCOProgram(Program):
    """A compiler program in the QCO dialect.

    QCO programs use value semantics and expose optimization and transformation
    operations.
    """

    @staticmethod
    def from_mlir_str(source: str) -> QCOProgram:
        """Parse a QCO dialect MLIR source string."""

    @staticmethod
    def from_mlir_file(path: str | os.PathLike) -> QCOProgram:
        """Parse QCO dialect MLIR from a file."""

    def copy(self) -> QCOProgram:
        """Return an independent copy of this program."""

    def cleanup(self) -> None:
        """Run the standard QCO cleanup pipeline in place."""

    def run_pass_pipeline(self, pipeline: str, *, enable_timing: bool = False, enable_statistics: bool = False) -> None:
        """Run a textual MLIR pass pipeline in place."""

    def merge_single_qubit_rotation_gates(self) -> None:
        """Merge compatible consecutive single-qubit rotation gates."""

    def fuse_single_qubit_unitary_runs(self, *, basis: str = "zyz") -> None:
        """Fuse single-qubit unitary runs into the chosen decomposition basis."""

    def unroll_quantum_loops(self, *, unroll_factor: int = -1) -> None:
        """Unroll quantum loops, optionally using a maximum unroll factor."""

    def lift_hadamards(self) -> None:
        """Move Hadamard gates through compatible operations."""

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
    ) -> None:
        """Place and route the program for a coupling graph."""

    def to_qc(self, *, copy: bool = False) -> QCProgram:
        """Convert this program to QC.

        Set ``copy=True`` to preserve it.
        """

    def to_jeff(self, *, copy: bool = False) -> JeffProgram:
        """Serialize this program as ``jeff``.

        Set ``copy=True`` to preserve it.
        """

class JeffProgram(Program):
    """A serialized ``jeff`` compiler program.

    ``jeff`` programs can be stored as bytes or files and converted back to QCO for
    further compilation.
    """

    @staticmethod
    def from_file(path: str | os.PathLike) -> JeffProgram:
        """Read a ``jeff`` program from a file."""

    @staticmethod
    def from_bytes(data: bytes) -> JeffProgram:
        """Deserialize a ``jeff`` program from bytes."""

    def copy(self) -> JeffProgram:
        """Return an independent copy of this program."""

    def cleanup(self) -> None:
        """Run the standard ``jeff`` cleanup pipeline in place."""

    def to_bytes(self) -> bytes:
        """Serialize this program to its ``jeff`` byte representation."""

    def write(self, path: str | os.PathLike) -> None:
        """Write this program to a ``jeff`` file."""

    def to_qco(self, *, copy: bool = False) -> QCOProgram:
        """Deserialize this program to QCO.

        Set ``copy=True`` to preserve it.
        """

class QIRProgram(Program):
    """A compiler program lowered to QIR.

    QIR programs retain their target profile and can be emitted as LLVM IR or
    LLVM bitcode.
    """

    def copy(self) -> QIRProgram:
        """Return an independent copy of this program."""

    def cleanup(self) -> None:
        """Run the standard QIR cleanup pipeline in place."""

    @property
    def profile(self) -> QIRProfile:
        """The QIR target profile used to produce this program."""

    @property
    def llvm_ir(self) -> str:
        """The program as textual LLVM IR."""

    def to_bitcode(self) -> bytes:
        """Serialize this program as LLVM bitcode."""

    def write_bitcode(self, path: str | os.PathLike) -> None:
        """Write this program as LLVM bitcode."""

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

    Input source strings, files, MQT ``QuantumComputation`` objects, Qiskit
    ``QuantumCircuit`` objects, and typed compiler programs can be combined with
    any supported output format. Typed program inputs are copied by default; set
    ``inplace=True`` to consume them. Use the typed programs directly to construct
    a custom pipeline stage by stage.

    Args:
        program: Source text, a file path, a circuit, or a typed compiler program.
        output: The requested output stage of the compiler pipeline.
        inplace: Whether a typed input program may be consumed.
        qco_pipeline: The QCO optimization pipeline to run.
        enable_timing: Whether to collect pass timing information.
        enable_statistics: Whether to collect pass statistics.

    Returns:
        A typed compiler program for the requested output format.
    """
