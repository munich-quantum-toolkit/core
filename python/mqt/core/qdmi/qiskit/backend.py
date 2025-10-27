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
from typing import TYPE_CHECKING, Any, cast

from qiskit.circuit import Measure, Parameter, QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from mqt.core import fomac

from .capabilities import extract_capabilities
from .exceptions import TranslationError, UnsupportedOperationError
from .job import QiskitJob

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.circuit import Instruction

__all__ = ["QiskitBackend"]


class QiskitBackend(BackendV2):  # type: ignore[misc]
    """A Qiskit BackendV2 adapter for QDMI devices via FoMaC.

    This backend provides OpenQASM-based program submission to QDMI devices.
    It automatically introspects device capabilities and constructs a Target
    object with supported operations.

    Args:
        device_index: Index of the device to use from fomac.devices() (default: 0).

    Raises:
        RuntimeError: If no FoMaC devices are available.
        IndexError: If device_index is out of range.

    Attributes:
        capabilities_hash: SHA256 hash of the device capabilities snapshot.
    """

    # Class-level counter for generating unique circuit names
    _circuit_counter = 0

    def __init__(self, device_index: int = 0) -> None:
        """Initialize the backend with a FoMaC device.

        Args:
            device_index: Index of the device to use from fomac.devices() (default: 0).

        Raises:
            RuntimeError: If no FoMaC devices are available.
            IndexError: If device_index is out of range.
        """
        # Get the device from FoMaC
        devices_list = list(fomac.devices())
        if not devices_list:
            msg = "No FoMaC devices available"
            raise RuntimeError(msg)
        if device_index < 0 or device_index >= len(devices_list):
            msg = f"Device index {device_index} out of range (available: {len(devices_list)})"
            raise IndexError(msg)

        self._device = devices_list[device_index]
        self._capabilities = extract_capabilities(self._device)

        # Initialize parent with device name
        super().__init__(name=self._capabilities.device_name)

        # Build Target from capabilities
        self._target = self._build_target()

        # Set backend options
        self._options = self._default_options()

    @property
    def capabilities_hash(self) -> str:
        """Return the capabilities hash for metadata embedding."""
        return self._capabilities.capabilities_hash or ""

    @property
    def target(self) -> Target:
        """Return the Target describing backend capabilities."""
        return self._target

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
        from qiskit.transpiler import InstructionProperties

        target = Target(description=f"QDMI device: {self._capabilities.device_name}")

        # Add operations from device capabilities
        for op_name, op_info in self._capabilities.operations.items():
            # Handle measurement operation
            if op_name == "measure":
                qargs = self._get_operation_qargs(op_info)
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

            # Determine which qubit pairs/singles this operation applies to
            qargs = self._get_operation_qargs(op_info)

            # Create instruction properties
            props = None
            if op_info.duration is not None or op_info.fidelity is not None:
                error = 1.0 - op_info.fidelity if op_info.fidelity is not None else None
                props = InstructionProperties(
                    duration=op_info.duration,
                    error=error,
                )

            # Add to target
            target.add_instruction(gate, dict.fromkeys(qargs, props))

        # Warn if no measurement operation is defined
        if "measure" not in self._capabilities.operations:
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

    def _get_operation_qargs(self, op_info: Any) -> Sequence[tuple[int, ...]]:  # noqa: ANN401
        """Get the qubit argument tuples for an operation.

        Args:
            op_info: Device operation metadata.

        Returns:
            Sequence of qubit index tuples this operation can act on.

        Raises:
            ValueError: If multi-qubit operation sites are improperly structured.
        """
        from itertools import combinations

        qubits_num = op_info.qubits_num if op_info.qubits_num is not None else 1
        num_qubits = self._capabilities.num_qubits

        if op_info.sites is not None:
            # Check if sites is a tuple of tuples (local multi-qubit with valid combinations)
            # or a flat tuple (single-qubit or global operations)
            if op_info.sites and isinstance(op_info.sites[0], tuple):
                # Local multi-qubit operation: sites already contains valid combinations
                # Each inner tuple represents a valid combination of qubit indices
                return cast("list[tuple[int, ...]]", list(op_info.sites))

            # Single-qubit or global operation: sites is a flat tuple of individual indices
            if qubits_num == 1:
                return [(site,) for site in op_info.sites]

            # Multi-qubit operations must have sites as tuple of tuples
            msg = (
                f"Multi-qubit operation '{op_info.name}' (qubits_num={qubits_num}) "
                f"has improperly structured sites (expected tuple of tuples). "
                "This indicates a device capability specification error."
            )
            raise ValueError(msg)

        # Generate all possible qubit combinations
        if qubits_num == 1:
            return [(i,) for i in range(num_qubits)]
        if qubits_num == 2:
            # Use coupling map if available
            if self._capabilities.coupling_map:
                # Convert coupling map tuples to list ensuring proper types
                return cast(
                    "list[tuple[int, ...]]",
                    [(int(pair[0]), int(pair[1])) for pair in self._capabilities.coupling_map],
                )
            # Otherwise all pairs
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]

        # Multi-qubit operations (3 or more qubits) - generate all combinations
        return list(combinations(range(num_qubits), qubits_num))

    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], **options: Any) -> QiskitJob:  # noqa: ANN401
        """Execute circuits on the backend.

        Args:
            run_input: Circuit(s) to execute.
            **options: Execution options (e.g., shots).

        Returns:
            Job handle for the execution.

        Raises:
            TranslationError: If circuit translation or execution fails.
        """
        # Normalize input to list
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else run_input

        # Update options
        shots_opt = options.get("shots", self._options.shots)
        try:
            shots = int(shots_opt)
        except Exception as exc:
            msg = f"Invalid 'shots' value: {shots_opt!r}"
            raise TranslationError(msg) from exc
        if shots < 0:
            msg = f"'shots' must be >= 0, got {shots}"
            raise TranslationError(msg)

        # Execute circuits
        results = [self._execute_circuit(circuit, shots=shots) for circuit in circuits]

        # Create and submit job
        job = QiskitJob(backend=self, circuits=circuits, results=results, shots=shots)
        job.submit()
        return job

    def _submit_to_device(self, circuit: QuantumCircuit, shots: int) -> dict[str, Any]:
        """Submit circuit to device for execution.

        Override this method to implement actual device submission.
        Default returns zero-state counts.

        Args:
            circuit: Circuit to execute.
            shots: Number of shots.

        Returns:
            Dictionary with 'counts', 'shots', 'success', and 'metadata' keys.
        """
        # Default: return zero-state counts
        num_clbits = circuit.num_clbits
        zero_state = "0" * num_clbits
        counts = {zero_state: shots}
        return {
            "counts": counts,
            "shots": shots,
            "success": True,
            "metadata": {
                "simulation": True,
                "capabilities_hash": self.capabilities_hash,
            },
        }

    def _execute_circuit(self, circuit: QuantumCircuit, shots: int) -> dict[str, Any]:
        """Execute a single circuit and return result dictionary.

        Args:
            circuit: Circuit to execute.
            shots: Number of shots.

        Returns:
            Result dictionary with counts and metadata.

        Raises:
            TranslationError: If circuit contains unbound parameters or invalid configuration.
            UnsupportedOperationError: If circuit contains unsupported operations.
        """
        # Validate circuit has no unbound parameters
        if circuit.parameters:
            param_names = ", ".join(sorted(p.name for p in circuit.parameters))
            msg = f"Circuit contains unbound parameters: {param_names}"
            raise TranslationError(msg)

        # Validate operations are supported
        allowed_ops = set(self._capabilities.operations.keys())
        allowed_ops.update({"measure", "barrier"})  # Always allow measure and barrier

        for instruction in circuit.data:
            op_name = instruction.operation.name
            if op_name not in allowed_ops:
                msg = f"Unsupported operation: '{op_name}'"
                raise UnsupportedOperationError(msg)

        # Submit circuit to device
        result = self._submit_to_device(circuit, shots)
        result.setdefault("metadata", {})

        # Generate unique circuit name - use provided name or generate one with counter
        if circuit.name:
            circuit_name = circuit.name
        else:
            # Generate unique name using class-level counter (similar to Qiskit's approach)
            QiskitBackend._circuit_counter += 1
            circuit_name = f"circuit-{QiskitBackend._circuit_counter}"

        result["metadata"]["circuit_name"] = circuit_name

        return result
