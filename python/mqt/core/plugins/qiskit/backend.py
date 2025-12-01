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
from typing import TYPE_CHECKING, Any

import qiskit.circuit.library as qcl
from qiskit import qasm2, qasm3
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import InstructionProperties, Target

from ... import fomac
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
        # Map known operations to Qiskit gates
        gate_map: dict[str, Instruction] = {
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
            # Rotation gates (parametric)
            "rx": qcl.RXGate(Parameter("theta")),
            "ry": qcl.RYGate(Parameter("theta")),
            "rz": qcl.RZGate(Parameter("phi")),
            "r": qcl.RGate(Parameter("theta"), Parameter("phi")),
            "prx": qcl.RGate(Parameter("theta"), Parameter("phi")),
            # Universal gates (parametric)
            "u": qcl.UGate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            "u3": qcl.UGate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            "u2": qcl.U2Gate(Parameter("phi"), Parameter("lambda")),
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
            "crx": qcl.CRXGate(Parameter("theta")),
            "cry": qcl.CRYGate(Parameter("theta")),
            "crz": qcl.CRZGate(Parameter("phi")),
            "rxx": qcl.RXXGate(Parameter("theta")),
            "ryy": qcl.RYYGate(Parameter("theta")),
            "rzz": qcl.RZZGate(Parameter("theta")),
            "rzx": qcl.RZXGate(Parameter("theta")),
            "xx_plus_yy": qcl.XXPlusYYGate(Parameter("theta"), Parameter("beta")),
            "xx_minus_yy": qcl.XXMinusYYGate(Parameter("theta"), Parameter("beta")),
            # nonunitary operations
            "reset": qcl.Reset(),
            "measure": qcl.Measure(),
        }

        return gate_map.get(op_name.lower())

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

    @staticmethod
    def _convert_circuit(
        circuit: QuantumCircuit, supported_program_formats: Iterable[fomac.ProgramFormat]
    ) -> tuple[str, fomac.ProgramFormat]:
        """Convert a :class:`~qiskit.circuit.QuantumCircuit` to one of the supported program formats.

        OpenQASM 3 takes precedence over OpenQASM 2 since it is a superset of the latter.

        Args:
            circuit: The quantum circuit to convert.
            supported_program_formats: Supported program formats.

        Returns:
            String representation of the circuit in the specified format.

        Raises:
            UnsupportedFormatError: If no supported program formats are found.
            TranslationError: If conversion fails.
        """
        if not supported_program_formats:
            msg = "No supported program formats found"
            raise UnsupportedFormatError(msg)

        if fomac.ProgramFormat.QASM3 in supported_program_formats:
            try:
                return str(qasm3.dumps(circuit)), fomac.ProgramFormat.QASM3
            except Exception as exc:
                msg = f"Failed to convert circuit to QASM3: {exc}"
                raise TranslationError(msg) from exc

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

        # Validate operations are supported
        allowed_ops = {op.name() for op in self._device.operations()}
        allowed_ops.update({"measure", "barrier"})  # Always allow measure and barrier

        for instruction in run_input.data:
            op_name = instruction.operation.name
            if op_name not in allowed_ops:
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
