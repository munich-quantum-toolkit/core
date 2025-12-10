# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Qiskit Backend.

Provides a Qiskit BackendV2-compatible interface to QDMI devices via FoMaC.
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import qiskit.circuit.library as qcl
from qiskit import qasm2, qasm3
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, Target

from ... import fomac
from .converters import qiskit_to_iqm_json
from .exceptions import (
    CircuitValidationError,
    JobSubmissionError,
    TranslationError,
    UnsupportedDeviceError,
    UnsupportedFormatError,
    UnsupportedOperationError,
)
from .job import QDMIJob

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit.circuit import Instruction, QuantumCircuit

    from .provider import QDMIProvider

__all__ = ["QDMIBackend"]


def __dir__() -> list[str]:
    return __all__


class QDMIBackend(BackendV2):  # type: ignore[misc]
    """A Qiskit BackendV2 adapter for QDMI devices via FoMaC.

    This backend provides program submission to QDMI devices.
    It automatically introspects device capabilities and constructs a
    :class:`~qiskit.transpiler.Target` object with supported operations.

    Backends should be obtained through :class:`~mqt.core.qdmi.qiskit.QDMIProvider`
    rather than instantiated directly.

    Args:
        device: FoMaC device to wrap.
        provider: The provider instance that created this backend.

    Examples:
        Get a backend through the provider:

        >>> from mqt.core.plugins.qiskit import QDMIProvider
        >>> provider = QDMIProvider()
        >>> backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    """

    @staticmethod
    def is_convertible(device: fomac.Device) -> bool:
        """Returns whether a device can be represented in Qiskit's Target model."""
        # Zoned operations cannot easily be represented in Qiskit's Target model
        return not any(op.is_zoned() for op in device.operations())

    # Class-level counter for generating unique circuit names
    _circuit_counter = itertools.count()

    # Mapping from Qiskit gate names to possible QDMI device operation names
    # This is used by _map_qiskit_gate_to_operation_names for efficient lookup
    _QISKIT_TO_QDMI_GATE_MAP: ClassVar[dict[str, set[str]]] = {
        # Single-qubit Pauli gates
        "x": {"x"},
        "y": {"y"},
        "z": {"z"},
        "id": {"id", "i"},
        "i": {"id", "i"},
        # Hadamard
        "h": {"h"},
        # Phase gates
        "s": {"s"},
        "sdg": {"sdg"},
        "t": {"t"},
        "tdg": {"tdg"},
        "sx": {"sx"},
        "sxdg": {"sxdg"},
        "p": {"p", "phase"},
        "phase": {"p", "phase"},
        "gphase": {"gphase"},
        # Rotation gates (parametric)
        "rx": {"rx"},
        "ry": {"ry"},
        "rz": {"rz"},
        "r": {"r", "prx"},  # Some devices use 'prx' for the R gate
        "prx": {"r", "prx"},  # Bidirectional: 'prx' also maps to both aliases
        # Universal gates (parametric)
        "u": {"u", "u3"},
        "u1": {"u1"},
        "u2": {"u2"},
        "u3": {"u", "u3"},
        # Two-qubit gates
        "cx": {"cx", "cnot"},
        "cnot": {"cx", "cnot"},
        "cy": {"cy"},
        "cz": {"cz"},
        "ch": {"ch"},
        "cs": {"cs"},
        "csdg": {"csdg"},
        "csx": {"csx"},
        "swap": {"swap"},
        "iswap": {"iswap"},
        "dcx": {"dcx"},
        "ecr": {"ecr"},
        # Two-qubit gates (parametric)
        "cp": {"cp"},
        "cu1": {"cu1"},
        "cu3": {"cu3"},
        "crx": {"crx"},
        "cry": {"cry"},
        "crz": {"crz"},
        "rxx": {"rxx"},
        "ryy": {"ryy"},
        "rzz": {"rzz"},
        "rzx": {"rzx"},
        "xx_plus_yy": {"xx_plus_yy"},
        "xx_minus_yy": {"xx_minus_yy"},
        # Three-qubit gates
        "ccx": {"ccx"},
        "ccz": {"ccz"},
        "cswap": {"cswap"},
        # Multi-controlled gates
        "mcx": {"mcx"},
        "mcz": {"mcz"},
        "mcp": {"mcp"},
        "mcrx": {"mcrx"},
        "mcry": {"mcry"},
        "mcrz": {"mcrz"},
        # Nonunitary operations
        "reset": {"reset"},
        "measure": {"measure"},
    }

    # Mapping from device operation names to Qiskit gate instances
    # This is used by _map_operation_to_gate for efficient lookup
    _OPERATION_TO_GATE_MAP: ClassVar[dict[str, Instruction]] = {
        # Single-qubit Pauli gates
        "x": qcl.XGate(),
        "y": qcl.YGate(),
        "z": qcl.ZGate(),
        "id": qcl.IGate(),
        "i": qcl.IGate(),
        # Hadamard
        "h": qcl.HGate(),
        # Phase gates
        "s": qcl.SGate(),
        "sdg": qcl.SdgGate(),
        "t": qcl.TGate(),
        "tdg": qcl.TdgGate(),
        "sx": qcl.SXGate(),
        "sxdg": qcl.SXdgGate(),
        "p": qcl.PhaseGate(Parameter("lambda")),
        "phase": qcl.PhaseGate(Parameter("lambda")),
        "gphase": qcl.GlobalPhaseGate(Parameter("theta")),
        # Rotation gates (parametric)
        "rx": qcl.RXGate(Parameter("theta")),
        "ry": qcl.RYGate(Parameter("theta")),
        "rz": qcl.RZGate(Parameter("phi")),
        "r": qcl.RGate(Parameter("theta"), Parameter("phi")),
        "prx": qcl.RGate(Parameter("theta"), Parameter("phi")),
        # Universal gates (parametric)
        "u": qcl.UGate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
        "u1": qcl.U1Gate(Parameter("lambda")),
        "u2": qcl.U2Gate(Parameter("phi"), Parameter("lambda")),
        "u3": qcl.UGate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
        # Two-qubit gates
        "cx": qcl.CXGate(),
        "cnot": qcl.CXGate(),
        "cy": qcl.CYGate(),
        "cz": qcl.CZGate(),
        "ch": qcl.CHGate(),
        "cs": qcl.CSGate(),
        "csdg": qcl.CSdgGate(),
        "csx": qcl.CSXGate(),
        "swap": qcl.SwapGate(),
        "iswap": qcl.iSwapGate(),
        "dcx": qcl.DCXGate(),
        "ecr": qcl.ECRGate(),
        # Two-qubit gates (parametric)
        "cp": qcl.CPhaseGate(Parameter("lambda")),
        "cu1": qcl.CU1Gate(Parameter("lambda")),
        "cu3": qcl.CU3Gate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
        "crx": qcl.CRXGate(Parameter("theta")),
        "cry": qcl.CRYGate(Parameter("theta")),
        "crz": qcl.CRZGate(Parameter("phi")),
        "rxx": qcl.RXXGate(Parameter("theta")),
        "ryy": qcl.RYYGate(Parameter("theta")),
        "rzz": qcl.RZZGate(Parameter("theta")),
        "rzx": qcl.RZXGate(Parameter("theta")),
        "xx_plus_yy": qcl.XXPlusYYGate(Parameter("theta"), Parameter("beta")),
        "xx_minus_yy": qcl.XXMinusYYGate(Parameter("theta"), Parameter("beta")),
        # Three-qubit gates
        "ccx": qcl.CCXGate(),
        "ccz": qcl.CCZGate(),
        "cswap": qcl.CSwapGate(),
        # Multi-controlled gates
        "mcx": qcl.MCXGate(num_ctrl_qubits=2),
        "mcz": qcl.MCPhaseGate(Parameter("lambda"), num_ctrl_qubits=2),
        "mcp": qcl.MCPhaseGate(Parameter("lambda"), num_ctrl_qubits=2),
        "mcrx": qcl.MCXGate(num_ctrl_qubits=2),  # Approximation
        "mcry": qcl.MCXGate(num_ctrl_qubits=2),  # Approximation
        "mcrz": qcl.MCXGate(num_ctrl_qubits=2),  # Approximation
        # nonunitary operations
        "reset": qcl.Reset(),
        "measure": qcl.Measure(),
    }

    def __init__(self, device: fomac.Device, provider: QDMIProvider | None = None) -> None:
        """Initialize the backend with a FoMaC device.

        Args:
            device: FoMaC device instance.
            provider: Provider instance that created this backend.

        Raises:
            UnsupportedDeviceError: If the device cannot be represented in Qiskit's Target model.
        """
        if not self.is_convertible(device):
            msg = f"Device '{device.name()}' cannot be represented in Qiskit's Target model"
            raise UnsupportedDeviceError(msg)

        super().__init__(name=device.name(), provider=provider, backend_version=device.version())
        self._device = device

        # Build Target from device
        self._target = self._build_target()

    @property
    def target(self) -> Target:
        """Return the Target describing backend capabilities."""
        return self._target

    @property
    def provider(self) -> Any | None:  # noqa: ANN401
        """Return the provider that created this backend."""
        return self._provider

    @property
    def max_circuits(self) -> int | None:
        """Maximum number of circuits that can be run in a single job."""
        return None  # No limit, processed sequentially

    @property
    def options(self) -> Options:
        """Return backend options."""
        return self._options

    @classmethod
    def _default_options(cls) -> Options:
        """Return default backend options.

        Returns:
            Default Options with shots=1024.
        """
        return Options(shots=1024)

    def _build_target(self) -> Target:
        """Construct a Qiskit Target from device capabilities.

        Returns:
            Target object with device operations and properties.
        """
        target = Target(
            description=f"QDMI device: {self._device.name()}",
            num_qubits=self._device.qubits_num(),
        )

        # Deduplicate operations by Qiskit gate name (not device operation name)
        # Multiple device operations may map to the same Qiskit gate
        seen_gate_names: set[str] = set()

        # Add operations from device
        for op in self._device.operations():
            # Map known operations to Qiskit gates
            op_name = op.name()

            # Skip control flow operations that don't belong in the Target
            # (barrier is handled separately by Qiskit, if_else is a circuit construct)
            if op_name.lower() in {"barrier", "if_else"}:
                continue

            gate = self._map_operation_to_gate(op_name)
            if gate is None:
                warnings.warn(
                    f"Device operation '{op_name}' cannot be mapped to a Qiskit gate and will be skipped",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Skip if we've already added this Qiskit gate to the target
            gate_name = gate.name
            if gate_name in seen_gate_names:
                continue
            seen_gate_names.add(gate_name)

            # Determine which qubits this operation applies to
            qargs = self._get_operation_qargs(op)

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
                continue

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
                continue

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
                continue

        # Check if the measurement operation is defined
        if "measure" not in seen_gate_names:
            warnings.warn(
                "Device does not define a measurement operation. This may limit practical usage.",
                UserWarning,
                stacklevel=2,
            )

        return target

    @staticmethod
    def _map_operation_to_gate(op_name: str) -> Instruction | None:
        """Map a device operation name to a Qiskit gate.

        Args:
            op_name: Device operation name.

        Returns:
            Qiskit gate instance or None if not mappable.
        """
        return QDMIBackend._OPERATION_TO_GATE_MAP.get(op_name.lower())

    @staticmethod
    def _map_qiskit_gate_to_operation_names(qiskit_gate_name: str) -> set[str]:
        """Map a Qiskit gate name to possible QDMI device operation names.

        This is the inverse of _map_operation_to_gate, accounting for the fact that
        different devices may use different naming conventions for the same operation.

        Args:
            qiskit_gate_name: Qiskit gate name.

        Returns:
            Set of possible QDMI device operation names that could map to this gate.
        """
        return QDMIBackend._QISKIT_TO_QDMI_GATE_MAP.get(qiskit_gate_name.lower(), {qiskit_gate_name.lower()})

    def _get_operation_qargs(self, op: fomac.Device.Operation) -> list[tuple[int]] | list[tuple[int, int]] | list[None]:
        """Get the qubit argument tuples for an operation.

        This method determines which qubit indices an operation can act on by:
        1. Checking explicit site lists from the operation (sites() for 1-qubit, site_pairs() for 2-qubit)
        2. For operations without site lists (returns None):
           - Single-qubit: Available on all individual qubits
           - Two-qubit with coupling map: Misconfigured device (error)
           - Two-qubit without coupling map: Available on all qubit pairs (all-to-all)
           - Multi-qubit (3+): Assumed to be globally available

        Args:
            op: Device operation from FoMaC.

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

    def _convert_circuit(
        self, circuit: QuantumCircuit, supported_program_formats: Iterable[fomac.ProgramFormat]
    ) -> tuple[str, fomac.ProgramFormat]:
        """Convert a :class:`~qiskit.circuit.QuantumCircuit` to one of the supported program formats.

        The conversion priority order is:
        1. IQM JSON (if supported) - device-specific format
        2. OpenQASM 3 (if supported) - superset of QASM 2
        3. OpenQASM 2 (if supported) - legacy format

        Args:
            circuit: The quantum circuit to convert.
            supported_program_formats: Supported program formats.

        Returns:
            Tuple of (program string, program format).

        Raises:
            UnsupportedFormatError: If no supported program formats are found.
            UnsupportedOperationError: If the circuit contains operations not supported by IQM JSON.
            TranslationError: If conversion fails.
        """
        if not supported_program_formats:
            msg = "No supported program formats found"
            raise UnsupportedFormatError(msg)

        # Try IQM JSON format first (device-specific)
        if fomac.ProgramFormat.IQM_JSON in supported_program_formats:
            try:
                return qiskit_to_iqm_json(circuit, self._device), fomac.ProgramFormat.IQM_JSON
            except UnsupportedOperationError:
                # Let this propagate so caller can handle fallback
                raise
            except Exception as exc:
                msg = f"Failed to convert circuit to IQM JSON: {exc}"
                raise TranslationError(msg) from exc

        # Try OpenQASM3
        if fomac.ProgramFormat.QASM3 in supported_program_formats:
            try:
                return str(qasm3.dumps(circuit)), fomac.ProgramFormat.QASM3
            except Exception as exc:
                msg = f"Failed to convert circuit to QASM3: {exc}"
                raise TranslationError(msg) from exc

        # Try OpenQASM2 (legacy)
        if fomac.ProgramFormat.QASM2 in supported_program_formats:
            try:
                return str(qasm2.dumps(circuit)), fomac.ProgramFormat.QASM2
            except Exception as exc:
                msg = f"Failed to convert circuit to QASM2: {exc}"
                raise TranslationError(msg) from exc

        msg = f"No conversion from Qiskit to any of the supported program formats: {supported_program_formats}"
        raise UnsupportedFormatError(msg)

    def run(self, run_input: QuantumCircuit, **options: Any) -> QDMIJob:  # noqa: ANN401
        """Execute a :class:`~qiskit.circuit.QuantumCircuit` on the backend.

        Args:
            run_input: Circuit to execute.
            **options: Execution options (e.g., shots, program_format).

        Returns:
            Job handle for the execution.

        Raises:
            CircuitValidationError: If circuit validation fails (e.g., invalid shots, unbound parameters).
            UnsupportedOperationError: If circuit contains unsupported operations.
            JobSubmissionError: If job submission to the device fails.
        """
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

        # Validate circuit has no unbound parameters
        if run_input.parameters:
            param_names = ", ".join(sorted(p.name for p in run_input.parameters))
            msg = f"Circuit contains unbound parameters: {param_names}"
            raise CircuitValidationError(msg)

        # Validate operations are supported - build set of all supported QDMI operation names
        device_ops = {op.name().lower() for op in self._device.operations()}

        for instruction in run_input.data:
            op_name = instruction.operation.name
            # Map the Qiskit gate name to possible QDMI operation names and check if any match
            possible_qdmi_names = self._map_qiskit_gate_to_operation_names(op_name)
            # Check if any of the possible QDMI names are supported by the device
            # Also always allow 'barrier' as it's a directive, not an operation
            if op_name != "barrier" and not any(qdmi_name in device_ops for qdmi_name in possible_qdmi_names):
                msg = f"Unsupported operation: '{op_name}'"
                raise UnsupportedOperationError(msg)

        # Convert circuit to specified program format
        program_str, program_format = self._convert_circuit(run_input, self._device.supported_program_formats())

        # Submit job to QDMI device
        try:
            qdmi_job = self._device.submit_job(
                program=program_str,
                program_format=program_format,
                num_shots=shots,
            )
        except Exception as exc:
            msg = f"Failed to submit job to device: {exc}"
            raise JobSubmissionError(msg) from exc

        # Create and return Qiskit job wrapper
        circuit_name = run_input.name or f"circuit-{next(QDMIBackend._circuit_counter)}"

        return QDMIJob(backend=self, job=qdmi_job, circuit_name=circuit_name)
