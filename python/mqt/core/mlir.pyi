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
from typing import Annotated, Literal, Unpack, overload

import numpy as np
import qiskit.circuit

import mqt.core.dd
from mqt.core.qdmi import Device
from mqt.core.typing import QDMISessionParameters

def _generate_benchmark(instance_specification_json: str) -> QCProgram:
    """Generate the QC program described by an instance specification."""

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

    OPENQASM3 = 4
    """OpenQASM 3 after the optimized QCO round trip."""

    JEFF = 5
    """Serializable ``jeff`` MLIR."""

    QIR_BASE = 6
    """QIR for the Base Profile."""

    QIR_ADAPTIVE = 7
    """QIR for the Adaptive Profile."""

class PayloadEncoding(enum.Enum):
    """Payload representation encoding."""

    TEXT = 0

    BINARY = 1

class PayloadFormat:
    """Exact payload identity."""

    def __init__(
        self, format_id: str, version: str, profile: str = "", encoding: PayloadEncoding = PayloadEncoding.TEXT
    ) -> None: ...
    @property
    def format_id(self) -> str: ...
    @format_id.setter
    def format_id(self, arg: str, /) -> None: ...
    @property
    def version(self) -> str: ...
    @version.setter
    def version(self, arg: str, /) -> None: ...
    @property
    def profile(self) -> str: ...
    @profile.setter
    def profile(self, arg: str, /) -> None: ...
    @property
    def encoding(self) -> PayloadEncoding: ...
    @encoding.setter
    def encoding(self, arg: PayloadEncoding, /) -> None: ...

class ProgramConstraint:
    """One payload capability constraint."""

    def __init__(self, constraint_id: str, value: int) -> None: ...
    @property
    def constraint_id(self) -> str: ...
    @constraint_id.setter
    def constraint_id(self, arg: str, /) -> None: ...
    @property
    def value(self) -> int: ...
    @value.setter
    def value(self, arg: int, /) -> None: ...

class ProgramCapability:
    """One payload execution capability."""

    def __init__(self, capability_id: str, value: int = 0, constraints: Sequence[ProgramConstraint] = []) -> None: ...
    @property
    def capability_id(self) -> str: ...
    @capability_id.setter
    def capability_id(self, arg: str, /) -> None: ...
    @property
    def value(self) -> int: ...
    @value.setter
    def value(self, arg: int, /) -> None: ...
    @property
    def constraints(self) -> list[ProgramConstraint]: ...
    @constraints.setter
    def constraints(self, arg: Sequence[ProgramConstraint], /) -> None: ...

class PayloadSpecification:
    """Selected payload execution contract."""

    def __init__(
        self,
        payload_format: PayloadFormat,
        capabilities: Sequence[ProgramCapability] = [],
        optional_capabilities_known: bool = False,
    ) -> None: ...
    @property
    def format(self) -> PayloadFormat:
        """The exact selected payload format."""

    @property
    def capabilities(self) -> list[ProgramCapability]:
        """The effective payload capabilities."""

    @property
    def optional_capabilities_known(self) -> bool:
        """Whether optional capability metadata is complete."""

class CompilerTarget:
    """Immutable MLIR compiler target.

    Every target has either all-to-all or explicitly enumerated connectivity and
    either unrestricted or explicitly enumerated native-operation support.
    """

    @overload
    def __init__(
        self,
        num_sites: int,
        *,
        connectivity: CompilerTarget.Connectivity,
        native_operations: CompilerTarget.NativeOperations,
        duration_unit: CompilerTarget.DurationUnit | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: str,
        num_sites: int,
        *,
        connectivity: CompilerTarget.Connectivity,
        native_operations: CompilerTarget.NativeOperations,
        duration_unit: CompilerTarget.DurationUnit | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        sites: Sequence[CompilerTarget.Site],
        *,
        connectivity: CompilerTarget.Connectivity,
        native_operations: CompilerTarget.NativeOperations,
        duration_unit: CompilerTarget.DurationUnit | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        name: str,
        sites: Sequence[CompilerTarget.Site],
        *,
        connectivity: CompilerTarget.Connectivity,
        native_operations: CompilerTarget.NativeOperations,
        duration_unit: CompilerTarget.DurationUnit | None = None,
    ) -> None: ...

    class DurationUnit:
        """Unit for raw target timing metadata."""

        def __init__(self, unit: str, scale_factor: float) -> None: ...
        @property
        def unit(self) -> str:
            """The reported duration unit."""

        @property
        def scale_factor(self) -> float:
            """The multiplier applied to raw timing values."""

    class Site:
        """A hardware site and its optional metadata."""

        def __init__(
            self, site_id: int, name: str | None = None, t1: int | None = None, t2: int | None = None
        ) -> None: ...
        @property
        def id(self) -> int:
            """The target-defined nonnegative site identifier."""

        @property
        def name(self) -> str | None:
            """The reported site name, if available."""

        @property
        def t1(self) -> int | None:
            """The raw T1 coherence time, if available."""

        @property
        def t2(self) -> int | None:
            """The raw T2 coherence time, if available."""

    class SiteTuple:
        """A supported ordered placement with optional calibration."""

        def __init__(
            self, sites: Sequence[int], duration: int | None = None, fidelity: float | None = None
        ) -> None: ...
        @property
        def sites(self) -> list[int]:
            """The ordered target site identifiers."""

        @property
        def duration(self) -> int | None:
            """The raw operation duration, if available."""

        @property
        def fidelity(self) -> float | None:
            """The operation fidelity, if available."""

    class OperationArityKind(enum.Enum):
        """How an operation capability accepts qubit widths."""

        FIXED = 0

        VARIADIC = 1

    class OperationArity:
        """Accepted operation qubit widths."""

        @staticmethod
        def fixed(value: int) -> CompilerTarget.OperationArity:
            """Create an exact operation arity."""

        @staticmethod
        def variadic(minimum: int) -> CompilerTarget.OperationArity:
            """Create an operation arity with an inclusive minimum. Operation construction requires a positive minimum."""

        @property
        def kind(self) -> CompilerTarget.OperationArityKind:
            """The arity kind."""

        @property
        def value(self) -> int:
            """The exact arity or inclusive variadic minimum."""

        def accepts(self, width: int) -> bool:
            """Whether this arity accepts a concrete width."""

    class Operation:
        """A target operation capability, calibration, and ordered applicability."""

        def __init__(
            self,
            name: str,
            arity: int | CompilerTarget.OperationArity,
            num_parameters: int,
            site_tuples: Sequence[CompilerTarget.SiteTuple | Sequence[int]] | None = None,
            duration: int | None = None,
            fidelity: float | None = None,
        ) -> None: ...
        @property
        def name(self) -> str:
            """The exact reported operation name."""

        @property
        def canonical_name(self) -> str:
            """The normalized compiler operation name."""

        @property
        def arity(self) -> CompilerTarget.OperationArity:
            """The accepted operation arity."""

        @property
        def num_parameters(self) -> int:
            """The number of real-valued parameters."""

        @property
        def site_tuples(self) -> list[CompilerTarget.SiteTuple]:
            """Supported ordered placements with optional calibration; empty means general applicability."""

        @property
        def duration(self) -> int | None:
            """The raw default duration, if available."""

        @property
        def fidelity(self) -> float | None:
            """The default fidelity, if available."""

    class GateKind(enum.Enum):
        """Recognized native gate capability."""

        U = 0

        X = 1

        SX = 2

        RZ = 3

        RX = 4

        RY = 5

        R = 6

        RXX = 7

        RYY = 8

        RZX = 9

        RZZ = 10

        ISWAP = 11

        CZ = 12

        CX = 13

        ECR = 14

    class SingleQubitBasis(enum.Enum):
        """Recognized target-wide single-qubit synthesis basis."""

        U = 0

        ZSXX = 1

        R = 2

        XZX = 3

        XYX = 4

        ZYZ = 5

        ZXZ = 6

    class SynthesisBasis:
        """One synthesis basis usable across the complete target."""

        @property
        def single_qubit(self) -> CompilerTarget.SingleQubitBasis:
            """The single-qubit synthesis basis."""

        @property
        def entangler(self) -> CompilerTarget.GateKind:
            """The two-qubit entangler."""

    class ConnectivityKind(enum.Enum):
        """The target connectivity model."""

        ALL_TO_ALL = 0

        EXPLICIT = 1

    class Connectivity:
        """A target connectivity model."""

        def __init__(self, couplings: Sequence[tuple[int, int]]) -> None:
            """Create an explicit connectivity model."""

        @staticmethod
        def all_to_all() -> CompilerTarget.Connectivity:
            """Create an all-to-all connectivity model."""

        @property
        def kind(self) -> CompilerTarget.ConnectivityKind:
            """The connectivity model."""

        @property
        def couplings(self) -> list[tuple[int, int]]:
            """The explicit couplings, if present."""

    class NativeOperationsKind(enum.Enum):
        """The native-operation support model."""

        UNRESTRICTED = 0

        EXPLICIT = 1

    class NativeOperations:
        """Native-operation support."""

        def __init__(self, operations: Sequence[CompilerTarget.Operation]) -> None:
            """Create explicit native-operation support."""

        @staticmethod
        def unrestricted() -> CompilerTarget.NativeOperations:
            """Create unrestricted native-operation support."""

        @property
        def kind(self) -> CompilerTarget.NativeOperationsKind:
            """The native-operation support model."""

        @property
        def operations(self) -> list[CompilerTarget.Operation]:
            """The explicit operations, if present."""

    @staticmethod
    def from_device(device: Device) -> CompilerTarget:
        """Snapshot a circuit-model QDMI device."""

    @staticmethod
    def from_device_id(device_id: str, **session_parameters: Unpack[QDMISessionParameters]) -> CompilerTarget:
        """Open a registered device and snapshot its compiler target."""

    @property
    def name(self) -> str | None:
        """The target name, if available."""

    @property
    def duration_unit(self) -> CompilerTarget.DurationUnit | None:
        """The target timing unit, if available."""

    @property
    def num_sites(self) -> int:
        """The number of target sites."""

    @property
    def sites(self) -> list[CompilerTarget.Site]:
        """Detailed sites in compiler-vertex order."""

    @property
    def connectivity_kind(self) -> CompilerTarget.ConnectivityKind:
        """The target connectivity model."""

    @property
    def couplings(self) -> list[tuple[int, int]]:
        """Canonical undirected couplings in target site IDs."""

    @property
    def native_operations_kind(self) -> CompilerTarget.NativeOperationsKind:
        """The target native-operation support model."""

    @property
    def operations(self) -> list[CompilerTarget.Operation]:
        """Operation capabilities in reported order."""

    @property
    def supported_gates(self) -> list[CompilerTarget.GateKind]:
        """Recognized native gates supported by the target."""

    @property
    def synthesis_basis(self) -> CompilerTarget.SynthesisBasis | None:
        """A complete target-wide synthesis basis, if available."""

    def supports_operation(
        self, name: str, arity: int, num_parameters: int | None = None, sites: Sequence[int] | None = None
    ) -> bool:
        """Whether the target supports an operation."""

class TargetEnvironment:
    """A compiler target and its selected payload specification."""

    def __init__(self, target: CompilerTarget, payload_specification: PayloadSpecification) -> None: ...
    @property
    def target(self) -> CompilerTarget:
        """The compiler target."""

    @property
    def payload_specification(self) -> PayloadSpecification:
        """The selected payload specification."""

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
    def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) -> QCProgram:
        """Translate a Qiskit {py:class}`~qiskit.circuit.QuantumCircuit` to QC MLIR."""

    def copy(self) -> QCProgram:
        """Return an independent copy of this program."""

    def cleanup(self) -> None:
        """Run the standard QC cleanup pipeline in place."""

    def normalize_global_phases(self) -> None:
        """Normalize scoped global phases in place."""

    def to_openqasm3(self) -> OpenQASMProgram:
        """Clean up and emit this QC program as OpenQASM 3 without QCO optimization."""

    def to_qiskit(self, *, target: CompilerTarget | None = None) -> qiskit.circuit.QuantumCircuit:
        """Translate this QC program to a Qiskit {py:class}`~qiskit.circuit.QuantumCircuit` without consuming it.

        Args:
            target: The optional compiler target used for mapping. When provided, emit
                a canonical physical circuit. All qubits must be static, and their site
                IDs must belong to the target.
        """

    def to_qco(self, *, copy: bool = False) -> QCOProgram:
        """Convert this program to QCO.

        Set ``copy=True`` to preserve it.
        """

    def to_qir(self, profile: QIRProfile, *, copy: bool = False) -> QIRProgram:
        """Lower this program to QIR for the requested profile.

        Set ``copy=True`` to preserve it.
        """

    def num_gates(self) -> int:
        """Count the gates in the program.

        Any operation that implements the ``UnitaryOpInterface`` is counted. Operations
        in every structured control-flow region are counted once, regardless of how
        often the region executes. Operations within modifiers are not counted
        recursively, and barriers are skipped.
        """

    def num_single_qubit_gates(self) -> int:
        """Count the single-qubit gates in the program.

        Any operation that implements the ``UnitaryOpInterface`` and acts on one qubit
        is counted. Operations in every structured control-flow region are counted
        once, regardless of how often the region executes. Operations within modifiers
        are not counted recursively, and barriers are skipped.
        """

    def num_two_qubit_gates(self) -> int:
        """Count the two-qubit gates in the program.

        Any operation that implements the ``UnitaryOpInterface`` and acts on two qubits
        is counted. Operations in every structured control-flow region are counted
        once, regardless of how often the region executes. Operations within modifiers
        are not counted recursively, and barriers are skipped.
        """

class QCOProgram(Program):
    """A compiler program in the QCO dialect.

    QCO programs use value semantics and expose optimization and transformation
    operations.
    """

    @staticmethod
    def from_mlir_str(source: str) -> QCOProgram:
        """Parse a QCO MLIR source string."""

    @staticmethod
    def from_mlir_file(path: str | os.PathLike) -> QCOProgram:
        """Parse QCO MLIR from a file."""

    def copy(self) -> QCOProgram:
        """Return an independent copy of this program."""

    def cleanup(self) -> None:
        """Run the standard QCO cleanup pipeline in place."""

    def normalize_global_phases(self) -> None:
        """Normalize scoped global phases in place."""

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

    def reuse_qubits(self) -> None:
        """Reuse independent single-qubit allocations."""

    def run_qubit_reuse_pipeline(self) -> None:
        """Prepare the program for qubit reuse and reuse eligible qubits."""

    def decompose_multi_controlled(self, *, min_qubits: int = 3) -> None:
        """Decompose controlled X/Z/SWAP gates, qco.rccx, and constant-angle phase gates that act on at least min_qubits qubits (min_qubits must be at least 3; default 3 means wider than two-qubit)."""

    def compile_for_target(
        self, target_environment: TargetEnvironment, *, enable_timing: bool = False, enable_statistics: bool = False
    ) -> None:
        """Compile this QCO program for the target in place. Do not rely on its contents if compilation fails."""

    def to_qc(self, *, copy: bool = False) -> QCProgram:
        """Convert this program to QC.

        Set ``copy=True`` to preserve it.
        """

    def to_jeff(self, *, copy: bool = False) -> JeffProgram:
        """Serialize this program as ``jeff``.

        Set ``copy=True`` to preserve it.
        """

    def build_functionality(self, dd_package: mqt.core.dd.DDPackage) -> mqt.core.dd.MatrixDD:
        """Build a matrix DD for a static unitary QCO program.

        Args:
            dd_package: DD package with enough qubits for the program.

        Returns:
            Matrix DD of the program functionality.

        Raises:
            ValueError: When the program is unsupported for functionality construction.
        """

    def simulate(
        self, initial_state: mqt.core.dd.VectorDD, dd_package: mqt.core.dd.DDPackage, seed: int = 0
    ) -> mqt.core.dd.VectorDD:
        """Simulate a QCO program on a DD state.

        Args:
            initial_state: Input state DD that spans at least the program's qubits and
                has a live reference in ``dd_package``. Higher wires are preserved. A
                valid input reference is consumed.
            dd_package: DD package with enough qubits for the program.
            seed: RNG seed. ``0`` (default) selects nondeterministic seeding. Any other
                value produces reproducible measurement and reset results.

        Returns:
            Output state DD.

        Raises:
            ValueError: When ``initial_state`` has no live reference in ``dd_package``,
                has too few qubits, or the program is unsupported for simulation.
        """

    def sample(self, shots: int = 1024, seed: int = 0) -> dict[str, int]:
        """Sample the declared outputs of a QCO program.

        Args:
            shots: Number of shots (default 1024).
            seed: RNG seed. ``0`` (default) selects nondeterministic seeding. Any other
                value produces reproducible results.

        Returns:
            Histogram keys use conventional count-string order. The last returned
            register comes first, and each register is MSB-first. If no CBit result
            exists, final ``measureAll`` bitstrings are used instead.

        Raises:
            ValueError: When the program is unsupported for sampling.
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

class OpenQASMProgram:
    """An immutable compiler program containing OpenQASM 3 source."""

    @property
    def source(self) -> str:
        """The emitted OpenQASM 3 source."""

    def write(self, path: str | os.PathLike) -> None:
        """Write the emitted source to a file."""

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

def build_functionality(
    program: str
    | os.PathLike[str]
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
) -> Annotated[np.typing.NDArray[np.complex128], {"shape": (None, None)}]:
    """Build the full unitary matrix of a supported compiler input.

    The DD package is managed internally. The matrix is materialized directly into
    the returned NumPy array without an additional copy. The full matrix grows
    exponentially, and the caller is responsible for requesting a result that fits
    in memory.

    Raises:
        MemoryError: When the dense matrix does not fit in memory.
        ValueError: When the program is unsupported or the matrix dimensions exceed
            addressable memory.
    """

def simulate(
    program: str
    | os.PathLike[str]
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
) -> Annotated[np.typing.NDArray[np.complex128], {"shape": (None,)}]:
    """Simulate a closed compiler input from the all-zero state.

    The DD package is managed internally. Terminal measurements that only assemble
    returned classical registers do not collapse the state. Mid-circuit measurement
    feedback and resets are unsupported; use {py:meth}`QCOProgram.simulate` with an
    explicit DD package for those workflows or for a custom initial state.

    Args:
        program: Compiler input to lower directly to QCO.

    Returns:
        Full statevector, materialized directly into the returned NumPy array.

    Raises:
        MemoryError: When the dense statevector does not fit in memory.
        ValueError: When the program is not closed, is unsupported for statevector
            simulation, or the statevector dimensions exceed addressable memory.
    """

def sample(
    program: str
    | os.PathLike[str]
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
    shots: int = 1024,
    seed: int = 0,
) -> dict[str, int]:
    """Sample a supported input after lowering it directly to QCO.

    An existing QCO program is used without copying. See
    {py:meth}`QCOProgram.sample` for the shot, seed, histogram, and error
    contracts.
    """

@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
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
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
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
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
    *,
    output: Literal[OutputFormat.OPENQASM3],
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> OpenQASMProgram: ...
@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
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
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
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
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
    *,
    output: OutputFormat,
    inplace: bool = False,
    qco_pipeline: str = "mqt-qco-default",
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> QCProgram | QCOProgram | OpenQASMProgram | JeffProgram | QIRProgram:
    """Run the coordinated default MQT compiler pipeline.

    Input source strings, files, Qiskit
    {py:class}`~qiskit.circuit.QuantumCircuit` objects, and typed compiler programs
    can be combined with any supported output format. Typed program inputs are
    copied by default; set ``inplace=True`` to consume them. Use the typed programs
    directly to construct a custom pipeline stage by stage.

    Args:
        program: Source text, a file path, a Qiskit circuit, or a typed compiler program.
        output: The requested output stage of the compiler pipeline.
        inplace: Whether a typed input program may be consumed.
        qco_pipeline: The QCO optimization pipeline to run. A custom pipeline
            cannot be combined with target compilation.
        enable_timing: Whether to collect pass timing information.
        enable_statistics: Whether to collect pass statistics.

    Returns:
        A typed compiler program for the requested output format.
    """

@overload
def compile_program(
    program: str
    | os.PathLike[str]
    | qiskit.circuit.QuantumCircuit
    | QCProgram
    | QCOProgram
    | JeffProgram
    | OpenQASMProgram,
    *,
    inplace: bool = False,
    target_environment: TargetEnvironment,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> OpenQASMProgram | QIRProgram:
    """Compile a program for a target and return the selected executable payload.

    The payload specification determines the output format. Typed program inputs
    are copied by default; set ``inplace=True`` to consume them.

    Args:
        program: Source text, a file path, a Qiskit circuit, or a typed compiler program.
        inplace: Whether a typed input program may be consumed.
        target_environment: The compiler target and selected payload specification.
        enable_timing: Whether to collect pass timing information.
        enable_statistics: Whether to collect pass statistics.

    Returns:
        A typed compiler program for the selected payload format.
    """
