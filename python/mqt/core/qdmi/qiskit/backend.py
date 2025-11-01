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

import warnings
from typing import TYPE_CHECKING, Any

from qiskit.circuit import Measure, Parameter
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from .exceptions import (
    CircuitValidationError,
    JobSubmissionError,
    TranslationError,
    UnsupportedFormatError,
    UnsupportedOperationError,
)
from .job import QiskitJob

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.circuit import Instruction, QuantumCircuit

    from mqt.core import fomac

__all__ = ["QiskitBackend"]


class QiskitBackend(BackendV2):  # type: ignore[misc]
    """A Qiskit BackendV2 adapter for QDMI devices via FoMaC.

    This backend provides OpenQASM-based program submission to QDMI devices.
    It automatically introspects device capabilities and constructs a Target
    object with supported operations.

    Backends should be obtained through :class:`~mqt.core.qdmi.qiskit.QDMIProvider`
    rather than instantiated directly.

    Args:
        device: FoMaC device to wrap.
        provider: The provider instance that created this backend.

    Examples:
        Get a backend through the provider:

        >>> from mqt.core.qdmi.qiskit import QDMIProvider
        >>> provider = QDMIProvider()
        >>> backend = provider.get_backend("MQT NA Default QDMI Device")
    """

    # Class-level counter for generating unique circuit names
    _circuit_counter = 0

    def __init__(self, device: fomac.Device, provider: Any | None = None) -> None:  # noqa: ANN401
        """Initialize the backend with a FoMaC device.

        Args:
            device: FoMaC device instance.
            provider: Provider instance that created this backend.
        """
        self._device = device
        self._provider = provider

        # Filter non-zone sites for qubit indexing
        all_sites = sorted(self._device.sites(), key=lambda x: x.index())
        self._non_zone_sites = [s for s in all_sites if not s.is_zone()]
        self._site_index_to_qubit_index = {s.index(): i for i, s in enumerate(self._non_zone_sites)}

        # Initialize parent with device name and provider
        super().__init__(name=self._device.name(), provider=provider)

        # Build Target from device
        self._target = self._build_target()

        # Set backend options
        self._options = self._default_options()

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
            Default Options with shots=1024 and program_format=QASM3.
        """
        from mqt.core import fomac

        return Options(shots=1024, program_format=fomac.ProgramFormat.QASM3)

    def _build_target(self) -> Target:
        """Construct a Qiskit Target from device capabilities.

        Returns:
            Target object with device operations and properties.
        """
        from qiskit.transpiler import InstructionProperties

        target = Target(description=f"QDMI device: {self._device.name()}")

        # Deduplicate operations by name (device may return duplicates)
        seen_operations: set[str] = set()

        # Add operations from device
        for op in self._device.operations():
            op_name = op.name()

            # Skip if we've already processed this operation name
            if op_name in seen_operations:
                continue
            seen_operations.add(op_name)

            # Handle the measurement operation
            if op_name == "measure":
                qargs = self._get_operation_qargs(op)
                target.add_instruction(Measure(), dict.fromkeys(qargs))
                continue

            # Map known operations to Qiskit gates
            gate = self._map_operation_to_gate(op_name)
            if gate is None:
                warnings.warn(
                    f"Device operation '{op_name}' cannot be mapped to a Qiskit gate and will be skipped",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Determine which qubits this operation applies to
            qargs = self._get_operation_qargs(op)

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

            # Add to target
            target.add_instruction(gate, dict.fromkeys(qargs, props))

        # Check if the measurement operation is defined
        if "measure" not in seen_operations:
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
        import qiskit.circuit.library as qcl

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
            "u": qcl.U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            "u3": qcl.U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            "u2": qcl.U2Gate(Parameter("phi"), Parameter("lambda")),
            # Two-qubit gates
            "cx": qcl.CXGate(),
            "cnot": qcl.CXGate(),
            "cy": qcl.CYGate(),
            "cz": qcl.CZGate(),
            "ch": qcl.CHGate(),
            "swap": qcl.SwapGate(),
            "iswap": qcl.iSwapGate(),
            "dcx": qcl.DCXGate(),
            "ecr": qcl.ECRGate(),
            # Two-qubit gates (parametric)
            "rxx": qcl.RXXGate(Parameter("theta")),
            "ryy": qcl.RYYGate(Parameter("theta")),
            "rzz": qcl.RZZGate(Parameter("theta")),
            "rzx": qcl.RZXGate(Parameter("theta")),
            "xx_plus_yy": qcl.XXPlusYYGate(Parameter("theta"), Parameter("beta")),
            "xx_minus_yy": qcl.XXMinusYYGate(Parameter("theta"), Parameter("beta")),
        }

        return gate_map.get(op_name.lower())

    def _get_operation_qargs(self, op: fomac.Device.Operation) -> Sequence[tuple[int, ...]]:
        """Get the qubit argument tuples for an operation.

        Args:
            op: Device operation from FoMaC.

        Returns:
            Sequence of qubit index tuples this operation can act on.

        Raises:
            ValueError: If multi-qubit operation sites are improperly structured.
        """
        from itertools import combinations

        qubits_num = op.qubits_num()
        qubits_num = qubits_num if qubits_num is not None else 1
        num_qubits = self._device.qubits_num()

        site_list = op.sites()
        is_zoned = op.is_zoned()

        # If operation is zoned or has no site list, generate all possible combinations
        if site_list is None or is_zoned:
            # Generate all possible qubit combinations
            if qubits_num == 1:
                return [(i,) for i in range(num_qubits)]
            if qubits_num == 2:
                # Use coupling map if available
                coupling_map = self._device.coupling_map()
                if coupling_map:
                    # Remap coupling map to qubit indices
                    remapped_coupling: list[tuple[int, ...]] = []
                    for pair in coupling_map:
                        idx0, idx1 = pair[0].index(), pair[1].index()
                        if idx0 in self._site_index_to_qubit_index and idx1 in self._site_index_to_qubit_index:
                            remapped_coupling.append((
                                self._site_index_to_qubit_index[idx0],
                                self._site_index_to_qubit_index[idx1],
                            ))
                    if remapped_coupling:
                        return remapped_coupling
                # Otherwise all pairs
                return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

            # Multi-qubit operations (3 or more qubits) - generate all combinations
            return list(combinations(range(num_qubits), qubits_num))

        # Extract and remap site indices to qubit indices
        raw_indices = [s.index() for s in site_list]

        # Filter out zone sites and remap to qubit indices
        remapped_indices: list[int] = [
            self._site_index_to_qubit_index[idx] for idx in raw_indices if idx in self._site_index_to_qubit_index
        ]

        # For single-qubit operations, use flat list
        if qubits_num == 1:
            return [(idx,) for idx in sorted(set(remapped_indices))]

        # For multi-qubit operations, the flat list represents consecutive pairs
        # due to reinterpret_cast from vector<pair<Site, Site>> to vector<Site>
        # Group consecutive elements into tuples of size qubits_num
        if len(remapped_indices) % qubits_num == 0:
            site_tuples: list[tuple[int, ...]] = [
                tuple(remapped_indices[i : i + qubits_num]) for i in range(0, len(remapped_indices), qubits_num)
            ]
            return site_tuples

        # Fallback: if not evenly divisible, something is wrong
        msg = (
            f"Multi-qubit operation '{op.name()}' (qubits_num={qubits_num}) "
            f"has improperly structured sites (expected {len(remapped_indices)} to be divisible by {qubits_num}). "
            "This indicates a device capability specification error."
        )
        raise ValueError(msg)

    @staticmethod
    def _convert_circuit(circuit: QuantumCircuit, program_format: fomac.ProgramFormat) -> str:
        """Convert a quantum circuit to the specified program format.

        Args:
            circuit: The quantum circuit to convert.
            program_format: The target program format.

        Returns:
            String representation of the circuit in the specified format.

        Raises:
            UnsupportedFormatError: If the program format is not supported.
            TranslationError: If conversion fails.
        """
        from mqt.core import fomac

        # Check for supported formats first
        if program_format not in {fomac.ProgramFormat.QASM2, fomac.ProgramFormat.QASM3}:
            msg = f"Unsupported program format: {program_format}"
            raise UnsupportedFormatError(msg)

        try:
            if program_format == fomac.ProgramFormat.QASM2:
                from qiskit import qasm2

                return str(qasm2.dumps(circuit))
            # Must be QASM3 at this point
            from qiskit import qasm3

            return str(qasm3.dumps(circuit))
        except Exception as exc:
            msg = f"Failed to convert circuit to {program_format}: {exc}"
            raise TranslationError(msg) from exc

    def run(self, run_input: QuantumCircuit, **options: Any) -> QiskitJob:  # noqa: ANN401
        """Execute a circuit on the backend.

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
        circuit = run_input

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
        if circuit.parameters:
            param_names = ", ".join(sorted(p.name for p in circuit.parameters))
            msg = f"Circuit contains unbound parameters: {param_names}"
            raise CircuitValidationError(msg)

        # Validate operations are supported
        allowed_ops = {op.name() for op in self._device.operations()}
        allowed_ops.update({"measure", "barrier"})  # Always allow measure and barrier

        for instruction in circuit.data:
            op_name = instruction.operation.name
            if op_name not in allowed_ops:
                msg = f"Unsupported operation: '{op_name}'"
                raise UnsupportedOperationError(msg)

        # Get program format from options
        program_format = options.get("program_format", self._options.program_format)

        # Convert circuit to specified program format
        program_str = self._convert_circuit(circuit, program_format)

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
        circuit_name = circuit.name or f"circuit-{QiskitBackend._circuit_counter}"
        if not circuit.name:
            QiskitBackend._circuit_counter += 1

        return QiskitJob(backend=self, job=qdmi_job, circuit_name=circuit_name)
