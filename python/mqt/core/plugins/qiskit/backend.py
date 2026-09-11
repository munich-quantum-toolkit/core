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
import warnings
from functools import cached_property
from math import isfinite
from numbers import Integral
from typing import TYPE_CHECKING, Any, ClassVar

from qiskit import qasm2, qasm3
from qiskit.circuit import ControlFlowOp, QuantumCircuit
from qiskit.circuit.library import (
    MCPhaseGate,
    MCXGate,
    get_standard_gate_name_mapping,
)
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, Target

from ...qdmi import Device as QDMIDevice
from ...qdmi import Job as QDMIJobHandle
from ...qdmi import ProgramFormat, is_binary_program_format
from ...qdmi.driver import open_device
from .exceptions import (
    CircuitValidationError,
    JobSubmissionError,
    TranslationError,
    UnsupportedDeviceError,
    UnsupportedFormatError,
    UnsupportedOperationError,
)
from .job import QDMIJob, _cancel_jobs
from .serializers import preferred_program_formats, program_serializer, register_program_serializer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, MutableSet, Sequence
    from typing import Unpack

    from qiskit.circuit import Instruction, Parameter
    from qiskit.circuit.parameterexpression import ParameterValueType

    from ...typing import QDMISessionParameters, QiskitEstimatorOptions, QiskitSamplerOptions
    from .provider import QDMIProvider

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
    canonical_gates = get_standard_gate_name_mapping()

    canonical_gates.update({
        "mcx": MCXGate,
        "mcphase": MCPhaseGate,
        "mcp": MCPhaseGate,
        "mcx_gray": MCXGate,
    })
    canonical_gates.update(extra_gates)

    qiskit_to_qdmi: dict[str, set[str]] = {}
    operation_to_gate: dict[str, Instruction | type[Instruction]] = {}

    for canonical_name, gate in canonical_gates.items():
        all_names = {canonical_name}
        if canonical_name in gate_aliases:
            all_names.update(gate_aliases[canonical_name])

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
    backend._validate_circuit(circuit, native=True)  # ruff: ignore[private-member-access] Built-in serializer.
    # Qiskit classical bits start at zero, while OpenQASM 3 bits are
    # uninitialized. Preserve Qiskit's semantics and make every output valid
    # even when the circuit measures only part of a register.
    if circuit.num_clbits:
        initialization = circuit.copy_empty_like(vars_mode="drop")
        initialization.global_phase = 0
        for clbit in initialization.clbits:
            initialization.store(
                clbit,
                False,  # ruff: ignore[boolean-positional-value-in-call] Qiskit store arguments are positional-only.
            )
        circuit = circuit.compose(initialization, front=True, inplace=False)

    exclusion_list = set()

    # Qiskit treats "measure", "reset", and "barrier" as keywords rather than gates
    exclusion_list.update({"measure", "reset", "barrier"})

    # Exclude standard-library gates to avoid duplicate definitions.
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

    # Emit device-supported gates outside the standard library as opaque gates.
    basis_gates = [gate for gate in backend.target.operation_names if gate not in exclusion_list] + ["mcx_gray", "U"]

    return qasm3.dumps(circuit, basis_gates=basis_gates)


def _serialize_to_qasm2(circuit: QuantumCircuit, backend: QDMIBackend) -> str:
    """Serialize a circuit into an OpenQASM 2 program.

    Args:
        circuit: The circuit to serialize.
        backend: The backend whose native placements constrain the circuit.

    Returns:
        The OpenQASM 2 program.
    """
    backend._validate_circuit(circuit, native=True)  # ruff: ignore[private-member-access] Built-in serializer.
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

    _GATE_ALIASES: ClassVar[dict[str, set[str]]] = {
        "id": {"i"},
        "p": {"phase"},
        "r": {"prx"},  # R gate can also be called 'prx' (IQM naming)
        "u": {"u3"},
        "cu": {"cu3"},
        "cx": {"cnot"},
        "global_phase": {"gphase"},  # Qiskit canonical name
        "gphase": {"global_phase"},  # OpenQASM canonical name
        "mcphase": {"mcp"},  # Qiskit canonical name
        "mcp": {"mcphase"},  # OpenQASM canonical name
        "mcx_gray": {"mcx"},
        "mcx_vchain": {"mcx"},
        "mcx_recursive": {"mcx"},
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

    def sampler(self, **options: Unpack[QiskitSamplerOptions]) -> BackendSamplerV2:
        """Construct Qiskit's native sampler with typed keyword options.

        Returns:
            A sampler that executes on this backend.
        """
        return BackendSamplerV2(backend=self, options=dict(options))

    def estimator(self, **options: Unpack[QiskitEstimatorOptions]) -> BackendEstimatorV2:
        """Construct Qiskit's native estimator with typed keyword options.

        Returns:
            An estimator that executes on this backend.
        """
        return BackendEstimatorV2(backend=self, options=dict(options))

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
        return None

    @property
    def options(self) -> Options:
        """The backend options."""
        return self._options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default backend options.

        Returns:
            Default Options with shots=1024 and memory=False.
        """
        return Options(shots=1024, memory=False)

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
        self._duration_conversion: tuple[float, float] | None = None
        target = Target(
            description=f"QDMI device: {self._device.name()}",
            num_qubits=self._target_num_qubits(),
        )

        # Device aliases can map several operations to the same Qiskit gate.
        seen_gate_names: set[str] = set()

        for op in self._device.operations():
            self._add_operation_to_target(target, op, seen_gate_names)

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

        gate_name = op_name if is_class else gate.name
        if gate_name in seen_gate_names:
            return
        seen_gate_names.add(gate_name)

        qargs = self._get_operation_qargs(op)

        # Globally supported gates (such as MCX) must specify a name and no properties
        if is_class:
            target.add_instruction(gate, name=op_name)
            return

        # If qargs is [None], it means the operation is available on all qubits
        if qargs == [None]:
            props = None
            duration = self._duration_seconds(op.duration())
            fidelity = op.fidelity()
            if duration is not None or fidelity is not None:
                error = 1.0 - fidelity if fidelity is not None else None
                props = InstructionProperties(
                    duration=duration,
                    error=error,
                )
            target.add_instruction(gate, {None: props})
            return

        target.add_instruction(gate, dict.fromkeys(qargs))

        site_tuples = self._get_operation_site_tuples(op)
        assert site_tuples is not None
        for qarg, sites in zip(qargs, site_tuples, strict=True):
            duration = self._duration_seconds(op.duration(sites=sites))
            fidelity = op.fidelity(sites=sites)
            if duration is not None or fidelity is not None:
                error = 1.0 - fidelity if fidelity is not None else None
                target.update_instruction_properties(
                    gate_name, qarg, InstructionProperties(duration=duration, error=error)
                )

    def _duration_seconds(self, duration: int | None) -> float | None:
        """Convert a raw QDMI duration to Qiskit's seconds.

        Returns:
            The duration in seconds, or None when it is unavailable.

        Raises:
            UnsupportedOperationError: If the duration unit or scale is invalid.
        """
        if duration is None:
            return None
        if self._duration_conversion is None:
            unit = self._device.duration_unit()
            seconds_per_unit = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12, "fs": 1e-15}
            if unit not in seconds_per_unit:
                msg = f"Cannot convert operation duration with device duration unit {unit!r} to seconds"
                raise UnsupportedOperationError(msg)
            scale = self._device.duration_scale_factor()
            if scale is None:
                scale = 1.0
            if not isfinite(scale) or scale <= 0:
                msg = f"Device duration scale factor must be positive and finite, got {scale!r}"
                raise UnsupportedOperationError(msg)
            self._duration_conversion = scale, seconds_per_unit[unit]
        scale, seconds_per_unit_value = self._duration_conversion
        return duration * scale * seconds_per_unit_value

    @staticmethod
    def _get_operation_site_tuples(op: QDMIDevice.Operation) -> Sequence[tuple[QDMIDevice.Site, ...]] | None:
        """Read explicit operation placements without widening their support.

        Returns:
            Ordered site tuples, or None when placements are unspecified.

        Raises:
            UnsupportedOperationError: If a site tuple is incomplete.
        """
        arity = op.qubits_num()
        if arity is None or arity == 0:
            return None
        if arity == 2:
            return op.site_pairs()
        sites = op.sites()
        if sites is None:
            return None
        if len(sites) % arity:
            msg = f"Operation '{op.name()}' has an incomplete {arity}-qubit site tuple"
            raise UnsupportedOperationError(msg)
        return [tuple(sites[i : i + arity]) for i in range(0, len(sites), arity)]

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

    def _get_operation_qargs(self, op: QDMIDevice.Operation) -> list[tuple[int, ...]] | list[None]:
        """Get explicit qubit tuples, or global support when placements are absent.

        Returns:
            Ordered qubit tuples, or [None] for global support.

        Raises:
            UnsupportedOperationError: If a site tuple is incomplete or a two-qubit
                operation omits placements on a device with a coupling map.
        """
        site_tuples = self._get_operation_site_tuples(op)
        if site_tuples is not None:
            return [tuple(site.index() for site in sites) for sites in site_tuples]
        if op.qubits_num() == 2 and self._device.coupling_map() is not None:
            msg = (
                f"Device provides a coupling map (stating connectivity constraints), "
                f"but operation '{op.name()}' does not expose site pairs. This indicates "
                f"a misconfigured device. Devices with connectivity constraints must expose "
                f"sites for their operations."
            )
            raise UnsupportedOperationError(msg)
        return [None]

    @cached_property
    def _native_operation_loci(self) -> dict[str, tuple[int | None, frozenset[tuple[int, ...] | None]]]:
        """Normalize native placements once for the opened device session.

        Returns:
            Native arity and placements by QDMI operation name.
        """
        return {
            operation.name().lower(): (operation.qubits_num(), frozenset(self._get_operation_qargs(operation)))
            for operation in self._device.operations()
        }

    def _validate_circuit(self, circuit: QuantumCircuit, *, native: bool = False) -> None:
        """Check supported operations, including operations inside control flow.

        Built-in QASM serializers also check native width and placements after
        preprocessing. Custom serializers can perform further compilation and
        therefore retain responsibility for validating their output placements.

        Raises:
            CircuitValidationError: If the circuit exceeds the native device width.
            UnsupportedOperationError: If an operation or placement is unsupported.
        """
        if native and circuit.num_qubits > self._device.qubits_num():
            msg = f"Circuit has {circuit.num_qubits} qubits, but the native device has {self._device.qubits_num()}."
            raise CircuitValidationError(msg)
        device_ops = {operation.name().lower() for operation in self._device.operations()}

        pending = [(circuit, tuple(range(circuit.num_qubits)))]
        while pending:
            block, indices = pending.pop()
            for instruction in block.data:
                operation = instruction.operation
                qargs = tuple(indices[block.find_bit(bit).index] for bit in instruction.qubits)
                if isinstance(operation, ControlFlowOp):
                    if operation.name not in self._target.operation_names:
                        msg = f"Unsupported control flow operation: '{operation.name}'"
                        raise UnsupportedOperationError(msg)
                    pending.extend((body, qargs) for body in reversed(operation.blocks))
                    continue
                if operation.name == "barrier":
                    continue
                names = self._map_qiskit_gate_to_operation_names(operation.name) & device_ops
                if not names:
                    msg = f"Unsupported operation: '{operation.name}'"
                    raise UnsupportedOperationError(msg)
                if native and not any(
                    arity in {None, len(qargs)} and (None in loci or qargs in loci)
                    for arity, loci in (self._native_operation_loci[name] for name in names)
                ):
                    msg = f"Operation '{operation.name}' is not advertised on native device qubits {qargs}."
                    raise UnsupportedOperationError(msg)

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
            CircuitValidationError: If native circuit validation fails.
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
            except (CircuitValidationError, UnsupportedOperationError):
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
            **options: Execution options: nonnegative integer ``shots`` and boolean ``memory``.
                Memory requires genuine QDMI SHOTS results. Simulator seeds are unsupported.

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
        """  # ruff:ignore[docstring-extraneous-exception] The validation helper raises operation errors.
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else run_input

        if not circuits:
            msg = "No circuits provided to run. At least one circuit is required."
            raise CircuitValidationError(msg)

        if parameter_values is not None and len(parameter_values) != len(circuits):
            msg = (
                f"Length of parameter_values ({len(parameter_values)}) must match "
                f"the number of circuits ({len(circuits)})"
            )
            raise CircuitValidationError(msg)

        # Native primitives pass an unset simulator seed to every backend.
        if options.get("seed_simulator") is None:
            options.pop("seed_simulator", None)
        if unsupported := options.keys() - self._options.keys():
            msg = f"Unsupported execution options: {', '.join(sorted(unsupported))}"
            raise CircuitValidationError(msg)

        shots_opt = options.get("shots", self._options.shots)
        if not isinstance(shots_opt, Integral) or isinstance(shots_opt, bool):
            msg = f"Invalid 'shots' value: {shots_opt!r}"
            raise CircuitValidationError(msg)
        shots = int(shots_opt)
        if shots < 0:
            msg = f"'shots' must be >= 0, got {shots}"
            raise CircuitValidationError(msg)
        memory = options.get("memory", self._options.memory)
        if not isinstance(memory, bool):
            msg = f"Invalid 'memory' value: {memory!r}"
            raise CircuitValidationError(msg)

        supported_formats = self._device.supported_program_formats()

        qdmi_jobs: list[QDMIJobHandle] = []
        prepared_circuits: list[QuantumCircuit] = []
        # Prepare every circuit before submitting any job, so validation cannot leave a partial batch.
        serialized_circuits: list[tuple[str | bytes, ProgramFormat]] = []

        for idx, circuit in enumerate(circuits):
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
            if [bit for register in bound_circuit.cregs for bit in register] != bound_circuit.clbits:
                msg = "Classical registers must partition circuit.clbits in register order."
                raise CircuitValidationError(msg)

            self._validate_circuit(bound_circuit)

            # Serialize the circuit into a program format the device accepts
            serialized_circuits.append(self._serialize_circuit(bound_circuit, supported_formats))
            prepared_circuits.append(bound_circuit)

        # Second pass: submit all validated circuits
        try:
            for program, program_format in serialized_circuits:
                try:
                    qdmi_jobs.append(
                        self._device.submit_job(program=program, program_format=program_format, num_shots=shots)
                    )
                except Exception as exc:
                    msg = f"Failed to submit job to device: {exc}"
                    raise JobSubmissionError(msg) from exc
            return QDMIJob(self, qdmi_jobs, prepared_circuits, shots=shots, memory=memory)
        except BaseException:
            _cancel_jobs(qdmi_jobs)
            raise


# Register bundled OpenQASM serializers when the Qiskit adapter is imported.
register_program_serializer(ProgramFormat.QASM3, _serialize_to_qasm3)
register_program_serializer(ProgramFormat.QASM2, _serialize_to_qasm2)
