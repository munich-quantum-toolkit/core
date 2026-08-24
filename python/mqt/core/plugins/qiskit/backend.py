# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Qiskit Backend.

Provides a Qiskit BackendV2-compatible interface to QDMI devices.
"""

from __future__ import annotations

import inspect
import itertools
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

from qiskit import qasm2, qasm3
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    MCPhaseGate,
    MCXGate,
    get_standard_gate_name_mapping,
)
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, Target

from ...qdmi import Device as QDMIDevice
from ...qdmi import Job as QDMIJobHandle
from ...qdmi import ProgramFormat, is_binary_program_format
from ...qdmi.driver import open_device
from .estimator import QDMIEstimator
from .exceptions import (
    CircuitValidationError,
    JobSubmissionError,
    TranslationError,
    UnsupportedDeviceError,
    UnsupportedFormatError,
    UnsupportedOperationError,
)
from .job import QDMIJob
from .sampler import QDMISampler
from .serializers import preferred_program_formats, program_serializer, register_program_serializer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableSet, Sequence
    from typing import Unpack

    from qiskit.circuit import Instruction, Parameter
    from qiskit.circuit.parameterexpression import ParameterValueType

    from ...typing import QDMISessionParameters
    from .provider import QDMIProvider

    # Type alias for parameter values
    ParametersType = Mapping[Parameter, ParameterValueType] | Iterable[ParameterValueType]

__all__ = ["QDMIBackend"]


def __dir__() -> list[str]:
    return __all__


def _build_gate_mappings_for_backend(
    gate_aliases: dict[str, set[str]],
    extra_gates: dict[str, Instruction | type[Instruction]],
) -> tuple[dict[str, set[str]], dict[str, Instruction | type[Instruction]]]:
    """Build both forward (Qiskit→QDMI) and inverse (QDMI→Gate) mappings.

    Uses Qiskit's standard gate mapping as the canonical source of truth,
    combined with a list of device-specific aliases and gates.

    Args:
        gate_aliases: Maps canonical names to their aliases.
        extra_gates: Maps names of gates outside Qiskit's standard library to
            the gate that represents them.

    Returns:
        Tuple of (qiskit_to_qdmi_map, operation_to_gate_map).
    """
    # Get Qiskit's standard gate name mapping as our canonical source
    canonical_gates = get_standard_gate_name_mapping()

    # Augment the canonical mapping with any additional gates that may not be in Qiskit's standard library
    canonical_gates.update({
        "mcx": MCXGate,
        "mcphase": MCPhaseGate,
        "mcp": MCPhaseGate,
        "mcx_gray": MCXGate,
    })
    canonical_gates.update(extra_gates)

    qiskit_to_qdmi: dict[str, set[str]] = {}
    operation_to_gate: dict[str, Instruction | type[Instruction]] = {}

    # Process each canonical gate from Qiskit's standard library
    for canonical_name, gate in canonical_gates.items():
        # Get all names for this gate (canonical + aliases)
        all_names = {canonical_name}
        if canonical_name in gate_aliases:
            all_names.update(gate_aliases[canonical_name])

        # For each name, map it to all names (bidirectional aliases)
        for name in all_names:
            qiskit_to_qdmi[name] = all_names.copy()
            operation_to_gate[name] = gate

    return qiskit_to_qdmi, operation_to_gate


def _serialize_to_qasm3(circuit: QuantumCircuit, backend: QDMIBackend) -> str:
    """Serialize a circuit into an OpenQASM 3 program.

    Args:
        circuit: The circuit to serialize.
        backend: The backend that runs the circuit. Its Target supplies the
            basis gates.

    Returns:
        The OpenQASM 3 program.
    """
    # Qiskit's OpenQASM3 exporter is fairly limited in terms of which gates it supports natively.
    # So it needs some help from us.
    exclusion_list = set()

    # Qiskit treats "measure", "reset", and "barrier" as keywords rather than gates
    exclusion_list.update({"measure", "reset", "barrier"})

    # We also need to remove all gates that are defined in the OpenQASM `stdlib.inc`.
    # Qiskit's exporter will otherwise complain about duplicate definitions.
    exclusion_list.update({
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "rx",
        "ry",
        "rz",
        "cx",
        "cy",
        "cz",
        "cp",
        "crx",
        "cry",
        "crz",
        "ch",
        "swap",
        "ccx",
        "cswap",
        "cu",
        "CX",
        "phase",
        "cphase",
        "id",
        "u1",
        "u2",
        "u3",
    })

    # By excluding already defined gates, we allow the exporter to emit otherwise unsupported gates without
    # needing to provide a definition for them. The exporter will then treat them as opaque gates, which is fine
    # as long as the target device supports them.
    basis_gates = [gate for gate in backend.target.operation_names if gate not in exclusion_list] + ["U"]

    return qasm3.dumps(circuit, basis_gates=basis_gates)


def _serialize_to_qasm2(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
    """Serialize a circuit into an OpenQASM 2 program.

    Args:
        circuit: The circuit to serialize.
        backend: The backend that runs the circuit. Qiskit's OpenQASM 2 exporter
            takes no information from it.

    Returns:
        The OpenQASM 2 program.
    """
    return qasm2.dumps(circuit)


def _check_payload_type(program: str | bytes, fmt: ProgramFormat) -> None:
    """Check that a serialized program has the payload type its format requires.

    Args:
        program: The program a serializer returned.
        fmt: The program format the serializer produces.

    Raises:
        TranslationError: If the payload type does not match the format.
    """
    expected = bytes if is_binary_program_format(fmt) else str
    if not isinstance(program, expected):
        msg = (
            f"The program serializer for {fmt.name} returned {type(program).__name__}, "
            f"but {fmt.name} requires {expected.__name__}"
        )
        raise TranslationError(msg)


class QDMIBackend(BackendV2):
    """A Qiskit BackendV2 adapter for QDMI devices.

    This backend provides program submission to QDMI devices.
    It automatically introspects device capabilities and constructs a
    :class:`~qiskit.transpiler.Target` object with supported operations.

    Use :meth:`from_device_id` to open one registered device. Use
    :class:`~mqt.core.plugins.qiskit.provider.QDMIProvider` to enumerate
    registered devices.

    Args:
        device: QDMI device wrapper.
        provider: The provider instance that created this backend.

    Examples:
        Open a backend by stable device ID:

        >>> backend = QDMIBackend.from_device_id("mqt.ddsim.default")
    """

    @staticmethod
    def is_convertible(device: QDMIDevice) -> bool:
        """Returns whether a device can be represented in Qiskit's Target model."""
        # Zoned operations cannot easily be represented in Qiskit's Target model
        return not any(op.is_zoned() for op in device.operations())

    # Class-level counter for generating unique circuit names
    _circuit_counter = itertools.count()

    # Define known aliases
    _GATE_ALIASES: ClassVar[dict[str, set[str]]] = {
        "id": {"i"},  # Identity gate can also be called 'i'
        "p": {"phase"},  # Phase gate can also be called 'phase'
        "r": {"prx"},  # R gate can also be called 'prx' (IQM naming)
        "u": {"u3"},  # U and U3 are the same gate
        "cu": {"cu3"},  # CU and CU3 are the same gate
        "cx": {"cnot"},  # CX and CNOT are the same gate
        "global_phase": {"gphase"},  # Qiskit canonical name
        "gphase": {"global_phase"},  # OpenQASM canonical name
        "mcphase": {"mcp"},  # Qiskit canonical name
        "mcp": {"mcphase"},  # OpenQASM canonical name
        "mcx_gray": {"mcx"},  # Alias for MCX with specific encoding
        "mcx_vchain": {"mcx"},  # Alias for MCX with specific encoding
        "mcx_recursive": {"mcx"},  # Alias for MCX with specific encoding
    }

    #: Gates outside Qiskit's standard library that the device natively supports.
    #: A subclass for a device with such a gate sets this to map the device
    #: operation name to the gate that represents it in the Target.
    _EXTRA_GATES: ClassVar[dict[str, Instruction | type[Instruction]]] = {}

    _QDMI_TO_QISKIT_GATE_MAP: ClassVar[dict[str, str]] = {
        "i": "id",
        "prx": "r",
        "mcp": "mcphase",
        "u3": "u",
        "gphase": "global_phase",
        "cu3": "cu",
    }

    _QISKIT_TO_QDMI_GATE_MAP: ClassVar[dict[str, set[str]]]
    _OPERATION_TO_GATE_MAP: ClassVar[dict[str, Instruction | type[Instruction]]]

    # Initialize derived mappings at class definition time
    _QISKIT_TO_QDMI_GATE_MAP, _OPERATION_TO_GATE_MAP = _build_gate_mappings_for_backend(_GATE_ALIASES, _EXTRA_GATES)

    def __init_subclass__(cls, **kwargs: Any) -> None:  # ruff:ignore[any-type]
        """Rebuild the gate mappings so a subclass sees its own aliases and gates.

        Args:
            **kwargs: Keyword arguments for the base implementation.
        """
        super().__init_subclass__(**kwargs)
        cls._QISKIT_TO_QDMI_GATE_MAP, cls._OPERATION_TO_GATE_MAP = _build_gate_mappings_for_backend(
            cls._GATE_ALIASES, cls._EXTRA_GATES
        )

    def __init__(
        self,
        device: QDMIDevice,
        provider: QDMIProvider | None = None,
        *,
        device_id: str | None = None,
    ) -> None:
        """Initialize the backend with a QDMI device wrapper.

        Args:
            device: QDMI device wrapper.
            provider: Provider instance that created this backend.
            device_id: Stable registry ID for the opened device, if known.

        Raises:
            UnsupportedDeviceError: If the device cannot be represented in Qiskit's Target model.
        """
        if not self.is_convertible(device):
            msg = f"Device '{device.name()}' cannot be represented in Qiskit's Target model"
            raise UnsupportedDeviceError(msg)

        super().__init__(name=device.name(), provider=provider, backend_version=device.version())
        self._device = device
        self._device_id = device_id

        # Build Target from device
        self._target = self._build_target()

    @classmethod
    def from_device_id(
        cls,
        device_id: str,
        *,
        provider: QDMIProvider | None = None,
        **session_parameters: Unpack[QDMISessionParameters],
    ) -> QDMIBackend:
        """Open a registered QDMI device and adapt it for Qiskit.

        Args:
            device_id: Stable ID from the QDMI device registry.
            provider: Provider to associate with the backend.
            session_parameters: Optional overrides for this device session.

        Returns:
            A Qiskit backend for a fresh QDMI device session.
        """
        return cls(
            device=open_device(device_id, **session_parameters),
            provider=provider,
            device_id=device_id,
        )

    @property
    def device(self) -> QDMIDevice:
        """The QDMI device the backend runs on."""
        return self._device

    @property
    def device_id(self) -> str | None:
        """Stable QDMI device ID, if known."""
        return self._device_id

    def sampler(self, *, default_shots: int = 1024) -> QDMISampler:
        """Construct a QDMI sampler for this backend.

        Returns:
            A sampler that executes on this backend.
        """
        return QDMISampler(self, default_shots=default_shots)

    def estimator(
        self,
        *,
        default_precision: float = 0.0,
        default_shots: int = 1024,
    ) -> QDMIEstimator:
        """Construct a QDMI estimator for this backend.

        Returns:
            An estimator that executes on this backend.
        """
        return QDMIEstimator(
            self,
            default_precision=default_precision,
            default_shots=default_shots,
        )

    @property
    def target(self) -> Target:
        """The Target describing the capabilities of the backend."""
        return self._target

    @property
    def provider(self) -> Any | None:  # ruff:ignore[any-type]
        """The provider that created the backend."""
        return self._provider

    @property
    def max_circuits(self) -> int | None:
        """The maximum number of circuits that can be run in a single job."""
        return None  # No limit, processed sequentially

    @property
    def options(self) -> Options:
        """The backend options."""
        return self._options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default backend options.

        Returns:
            Default Options with shots=1024.
        """
        return Options(shots=1024)

    def _target_num_qubits(self) -> int:
        """Number of addressable qubits to expose in the Target.

        Subclasses may override this to hide device sites that should not be
        directly addressable by the transpiler (e.g. computational
        resonators on star-topology architectures).

        Returns:
            Number of qubits to expose in the Target.
        """
        return self._device.qubits_num()

    def _build_target(self) -> Target:
        """Construct a Qiskit Target from device capabilities.

        Returns:
            Target object with device operations and properties.
        """
        target = Target(
            description=f"QDMI device: {self._device.name()}",
            num_qubits=self._target_num_qubits(),
        )

        # Deduplicate operations by Qiskit gate name (not device operation name)
        # Multiple device operations may map to the same Qiskit gate
        seen_gate_names: set[str] = set()

        # Add operations from device
        for op in self._device.operations():
            self._add_operation_to_target(target, op, seen_gate_names)

        # Check if the measurement operation is defined
        if "measure" not in seen_gate_names:
            warnings.warn(
                f"{self._device.name()} does not define a measurement operation. This may limit practical usage.",
                UserWarning,
                stacklevel=2,
            )

        return target

    def _add_operation_to_target(
        self, target: Target, op: QDMIDevice.Operation, seen_gate_names: MutableSet[str]
    ) -> None:
        """Add a single device operation to the Target, if it maps to a Qiskit gate.

        Subclasses may override this to customize how an individual device
        operation is represented in the Target, e.g. substituting fictional
        pairs of qubit sites for an operation that natively acts on non-qubit
        sites (such as a qubit-resonator gate).

        Args:
            target: The Target being constructed.
            op: The device operation to add.
            seen_gate_names: Qiskit gate names already added to the target (mutated in place).
        """
        # Map known operations to Qiskit gates
        op_name = op.name().lower()

        # Skip control flow operations that don't belong in the Target
        # (barrier is handled separately by Qiskit, if_else is a circuit construct)
        if op_name in {"barrier", "if_else"}:
            return

        if op_name in self._QDMI_TO_QISKIT_GATE_MAP:
            op_name = self._QDMI_TO_QISKIT_GATE_MAP[op_name]

        gate = self._map_operation_to_gate(op_name)
        if gate is None:
            warnings.warn(
                f"Device operation '{op_name}' cannot be mapped to a Qiskit gate and will be skipped",
                UserWarning,
                stacklevel=2,
            )
            return

        is_class = inspect.isclass(gate)

        # Skip if we've already added this Qiskit gate to the target
        gate_name = op_name if is_class else gate.name
        if gate_name in seen_gate_names:
            return
        seen_gate_names.add(gate_name)

        # Determine which qubits this operation applies to
        qargs = self._get_operation_qargs(op)

        # Globally supported gates (such as MCX) must specify a name and no properties
        if is_class:
            target.add_instruction(gate, name=op_name)
            return

        # If qargs is [None], it means the operation is available on all qubits
        if qargs == [None]:
            # Create instruction properties
            props = None
            duration = op.duration()
            fidelity = op.fidelity()
            if duration is not None or fidelity is not None:
                error = 1.0 - fidelity if fidelity is not None else None
                props = InstructionProperties(
                    duration=duration,
                    error=error,
                )
            target.add_instruction(gate, {None: props})
            return

        # Add the operation without properties and populate them iteratively later
        target.add_instruction(gate, dict.fromkeys(qargs))

        num_qubits = op.qubits_num()
        if num_qubits == 1:
            op_sites = op.sites()
            assert op_sites is not None
            for qarg, site in zip(qargs, op_sites, strict=True):
                duration = op.duration(sites=[site])
                fidelity = op.fidelity(sites=[site])
                if duration is not None or fidelity is not None:
                    error = 1.0 - fidelity if fidelity is not None else None
                    props = InstructionProperties(
                        duration=duration,
                        error=error,
                    )
                    target.update_instruction_properties(gate_name, qarg, props)
            return

        if num_qubits == 2:
            op_site_pairs = op.site_pairs()
            assert op_site_pairs is not None
            for qarg, (site1, site2) in zip(qargs, op_site_pairs, strict=True):
                duration = op.duration(sites=[site1, site2])
                fidelity = op.fidelity(sites=[site1, site2])
                if duration is not None or fidelity is not None:
                    error = 1.0 - fidelity if fidelity is not None else None
                    props = InstructionProperties(
                        duration=duration,
                        error=error,
                    )
                    target.update_instruction_properties(gate_name, qarg, props)
            return

    @classmethod
    def _map_operation_to_gate(cls, op_name: str) -> Instruction | type[Instruction] | None:
        """Map a device operation name to a Qiskit gate.

        Args:
            op_name: Device operation name.

        Returns:
            Qiskit gate instance or None if not mappable.
        """
        return cls._OPERATION_TO_GATE_MAP.get(op_name.lower())

    @classmethod
    def _map_qiskit_gate_to_operation_names(cls, qiskit_gate_name: str) -> set[str]:
        """Map a Qiskit gate name to possible QDMI device operation names.

        This is the inverse of _map_operation_to_gate, accounting for the fact that
        different devices may use different naming conventions for the same operation.

        Args:
            qiskit_gate_name: Qiskit gate name.

        Returns:
            Set of possible QDMI device operation names that could map to this gate.
        """
        return cls._QISKIT_TO_QDMI_GATE_MAP.get(qiskit_gate_name.lower(), {qiskit_gate_name.lower()})

    def _get_operation_qargs(self, op: QDMIDevice.Operation) -> list[tuple[int]] | list[tuple[int, int]] | list[None]:
        """Get the qubit argument tuples for an operation.

        This method determines which qubit indices an operation can act on by:
        1. Checking explicit site lists from the operation (sites() for 1-qubit, site_pairs() for 2-qubit)
        2. For operations without site lists (returns None):
           - Single-qubit: Available on all individual qubits
           - Two-qubit with coupling map: Misconfigured device (error)
           - Two-qubit without coupling map: Available on all qubit pairs (all-to-all)
           - Multi-qubit (3+): Assumed to be globally available

        Args:
            op: QDMI device operation.

        Returns:
            Sequence of qubit index tuples this operation can act on.
            Returns [None] for globally available operations (will be converted to {None: None} in Target).

        Raises:
            UnsupportedOperationError: If the device is misconfigured.
        """
        qubits_num = op.qubits_num()

        # For single-qubit operations, first check for explicit sites
        if qubits_num == 1:
            site_list = op.sites()
            if site_list is not None:
                # Operation explicitly defines where it can be executed
                return [(s.index(),) for s in site_list]

            # No explicit sites - operation is globally available on all qubits
            return [None]

        # For two-qubit operations, first check for explicit site_pairs
        if qubits_num == 2:
            site_pairs = op.site_pairs()
            if site_pairs is not None:
                return [(s1.index(), s2.index()) for s1, s2 in site_pairs]

            # Two-qubit operations without explicit site_pairs
            # Check device-level coupling map
            coupling_map = self._device.coupling_map()
            if coupling_map is not None:
                # Device has coupling map but operation doesn't expose sites
                msg = (
                    f"Device provides a coupling map (stating connectivity constraints), "
                    f"but operation '{op.name()}' does not expose site pairs. This indicates "
                    f"a misconfigured device. Devices with connectivity constraints must expose "
                    f"sites for their operations."
                )
                raise UnsupportedOperationError(msg)

            # No coupling map and no site pairs - operation is globally available (all-to-all)
            return [None]

        # Operation has unspecified qubit count or 3+ qubits -> assume it applies to all qubits
        return [None]

    def _preprocess_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:  # ruff:ignore[no-self-use]
        """Rewrite a bound circuit before validation and conversion.

        Called once per circuit in :meth:`run`, after parameter binding and
        before operation-support validation and program conversion.
        Subclasses may override this to transform a circuit into a
        device-native equivalent, e.g. inserting MOVE gates and widening the
        circuit to address computational resonators. The default
        implementation is the identity function.

        Args:
            circuit: The bound circuit to preprocess.

        Returns:
            The (possibly rewritten) circuit to use for validation and conversion.
        """
        return circuit

    def _serialize_circuit(
        self, circuit: QuantumCircuit, supported_program_formats: Iterable[ProgramFormat]
    ) -> tuple[str | bytes, ProgramFormat]:
        """Serialize a :class:`~qiskit.circuit.QuantumCircuit` into a program the device accepts.

        The method walks the formats the device supports in the order of
        :data:`~mqt.core.plugins.qiskit.serializers.PROGRAM_FORMAT_PREFERENCE`
        and uses the first one that has a registered serializer. See
        :mod:`mqt.core.plugins.qiskit.serializers` for how a package registers a
        serializer.

        Args:
            circuit: The circuit to serialize.
            supported_program_formats: The program formats the device accepts.

        Returns:
            Tuple of (program, program format). The program is a string for a
            text format and bytes for a binary format.

        Raises:
            UnsupportedFormatError: If the device reports no program format that
                has a serializer.
            UnsupportedOperationError: If the circuit contains an operation the
                chosen format cannot express.
            TranslationError: If serialization fails.
        """
        formats = list(supported_program_formats)
        if not formats:
            msg = "The device reports no supported program formats"
            raise UnsupportedFormatError(msg)

        for fmt in preferred_program_formats(formats):
            serializer = program_serializer(fmt)
            if serializer is None:
                continue
            try:
                program = serializer(circuit, self)
            except UnsupportedOperationError:
                # A circuit the chosen format cannot express must fail loudly
                # rather than arrive at the device in a weaker format.
                raise
            except Exception as exc:
                msg = f"Failed to serialize the circuit to {fmt.name}: {exc}"
                raise TranslationError(msg) from exc
            _check_payload_type(program, fmt)
            return program, fmt

        msg = f"No program serializer for any format the device supports: {[fmt.name for fmt in formats]}"
        raise UnsupportedFormatError(msg)

    def run(
        self,
        run_input: QuantumCircuit | Sequence[QuantumCircuit],
        parameter_values: Sequence[ParametersType] | None = None,
        **options: Any,  # ruff:ignore[any-type]
    ) -> QDMIJob:
        """Execute one or more :class:`~qiskit.circuit.QuantumCircuit` instances on the backend.

        Args:
            run_input: A single quantum circuit or a sequence of quantum circuits to execute.
            parameter_values: Optional parameter values to bind to the circuits. If provided, must be a sequence
                with one entry per circuit. Each entry can be either a dictionary mapping parameters to values,
                or a sequence of values in the order of circuit.parameters.
            **options: Execution options (e.g., shots).

        Returns:
            Job handle for the execution. For multiple circuits, the job aggregates results from all circuits.

        Raises:
            CircuitValidationError: If circuit validation fails (e.g., invalid shots, unbound parameters,
                parameter_values length mismatch).
            UnsupportedOperationError: If a circuit contains unsupported operations.
            JobSubmissionError: If job submission to the device fails.

        Examples:
            Run a single circuit with parameter values:

            >>> from qiskit.circuit import Parameter, QuantumCircuit
            >>> theta = Parameter("theta")
            >>> qc = QuantumCircuit(1)
            >>> qc.ry(theta, 0)
            >>> qc.measure_all()
            >>> job = backend.run(qc, parameter_values=[{theta: 1.5708}])

            Run multiple circuits with different parameter values:

            >>> qc1 = QuantumCircuit(1)
            >>> qc1.ry(theta, 0)
            >>> qc1.measure_all()
            >>> qc2 = QuantumCircuit(1)
            >>> qc2.ry(theta, 0)
            >>> qc2.measure_all()
            >>> job = backend.run([qc1, qc2], parameter_values=[{theta: 0.5}, {theta: 1.5}])
        """
        # Normalize input to a list of circuits
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else run_input

        # Validate non-empty circuit list
        if not circuits:
            msg = "No circuits provided to run. At least one circuit is required."
            raise CircuitValidationError(msg)

        # Validate parameter_values length if provided
        if parameter_values is not None and len(parameter_values) != len(circuits):
            msg = (
                f"Length of parameter_values ({len(parameter_values)}) must match "
                f"the number of circuits ({len(circuits)})"
            )
            raise CircuitValidationError(msg)

        # Get shots option
        shots_opt = options.get("shots", self._options.shots)
        try:
            shots = int(shots_opt)
        except Exception as exc:
            msg = f"Invalid 'shots' value: {shots_opt!r}"
            raise CircuitValidationError(msg) from exc
        if shots < 0:
            msg = f"'shots' must be >= 0, got {shots}"
            raise CircuitValidationError(msg)

        # Build set of all supported QDMI operation names once
        device_ops = {op.name().lower() for op in self._device.operations()}

        # Process each circuit
        qdmi_jobs: list[QDMIJobHandle] = []
        circuit_names: list[str] = []
        # First pass: validate and serialize all circuits
        serialized_circuits: list[tuple[str | bytes, ProgramFormat, str]] = []

        for idx, circuit in enumerate(circuits):
            # Bind parameters if provided
            bound_circuit = circuit
            if parameter_values is not None:
                try:
                    bound_circuit = circuit.assign_parameters(parameter_values[idx])
                except Exception as exc:
                    msg = f"Failed to bind parameters for circuit {idx}: {exc}"
                    raise CircuitValidationError(msg) from exc

            # Validate circuit has no unbound parameters
            if bound_circuit.parameters:
                params = ", ".join(sorted(p.name for p in bound_circuit.parameters))
                msg = (
                    f"Circuit contains unbound parameters: {params}. Provide `parameter_values` or bind them manually."
                )
                raise CircuitValidationError(msg)

            bound_circuit = self._preprocess_circuit(bound_circuit)

            # Validate operations are supported
            for instruction in bound_circuit.data:
                op_name = instruction.operation.name
                # Map the Qiskit gate name to possible QDMI operation names and check if any match
                possible_qdmi_names = self._map_qiskit_gate_to_operation_names(op_name)
                # Check if any of the possible QDMI names are supported by the device
                # Also always allow 'barrier' as it's a directive, not an operation
                if op_name != "barrier" and not any(qdmi_name in device_ops for qdmi_name in possible_qdmi_names):
                    msg = f"Unsupported operation: '{op_name}'"
                    raise UnsupportedOperationError(msg)

            # Serialize the circuit into a program format the device accepts
            program, program_format = self._serialize_circuit(bound_circuit, self._device.supported_program_formats())
            circuit_name = circuit.name or f"circuit-{next(QDMIBackend._circuit_counter)}"
            serialized_circuits.append((program, program_format, circuit_name))

        # Second pass: submit all validated circuits
        for program, program_format, circuit_name in serialized_circuits:
            # Submit job to QDMI device
            try:
                qdmi_job = self._device.submit_job(
                    program=program,
                    program_format=program_format,
                    num_shots=shots,
                )
            except Exception as exc:
                msg = f"Failed to submit job to device: {exc}"
                raise JobSubmissionError(msg) from exc

            # Track the job and circuit name
            qdmi_jobs.append(qdmi_job)
            circuit_names.append(circuit_name)

        # Create and return Qiskit job wrapper (handles single or multiple jobs)
        return QDMIJob(backend=self, jobs=qdmi_jobs, circuit_names=circuit_names)


# MQT Core owns the two OpenQASM formats and registers them through the same
# registry as everyone else, so the backend walks one ordered list of formats
# with no format-specific branch. `mqt.core.plugins.qiskit.__init__` imports this
# module whenever Qiskit is installed, so both formats are available as soon as
# the adapter is.
register_program_serializer(ProgramFormat.QASM3, _serialize_to_qasm3)
register_program_serializer(ProgramFormat.QASM2, _serialize_to_qasm2)
