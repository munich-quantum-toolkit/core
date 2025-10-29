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

from qiskit.circuit import Measure, Parameter, QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from mqt.core import fomac

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

        # Filter non-zone sites for qubit indexing
        all_sites = sorted(self._device.sites(), key=lambda x: x.index())
        self._non_zone_sites = [s for s in all_sites if not s.is_zone()]
        self._site_index_to_qubit_index = {s.index(): i for i, s in enumerate(self._non_zone_sites)}

        # Initialize parent with device name
        super().__init__(name=self._device.name())

        # Build Target from device
        self._target = self._build_target()

        # Set backend options
        self._options = self._default_options()

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

        if site_list is not None:
            # Extract and remap site indices to logical qubit indices
            raw_indices = [s.index() for s in site_list]

            # Filter out zone sites and remap to logical qubit indices
            remapped_indices: list[int] = [
                self._site_index_to_qubit_index[idx] for idx in raw_indices if idx in self._site_index_to_qubit_index
            ]

            # For local multi-qubit operations, the flat list represents consecutive pairs
            # due to reinterpret_cast from vector<pair<Site, Site>> to vector<Site>
            if not is_zoned and qubits_num > 1:
                # Group consecutive elements into tuples of size qubits_num
                if len(remapped_indices) % qubits_num == 0:
                    site_tuples: list[tuple[int, ...]] = [
                        tuple(remapped_indices[i : i + qubits_num]) for i in range(0, len(remapped_indices), qubits_num)
                    ]
                    return site_tuples
                # Fallback: treat as flat list if not evenly divisible
                if qubits_num == 1:
                    return [(idx,) for idx in sorted(set(remapped_indices))]
                # Multi-qubit operations must have properly structured sites
                msg = (
                    f"Multi-qubit operation '{op.name()}' (qubits_num={qubits_num}) "
                    f"has improperly structured sites (expected pairs). "
                    "This indicates a device capability specification error."
                )
                raise ValueError(msg)

            # For single-qubit and global operations, use flat list
            if qubits_num == 1:
                return [(idx,) for idx in sorted(set(remapped_indices))]

            msg = (
                f"Multi-qubit operation '{op.name()}' (qubits_num={qubits_num}) "
                f"has improperly structured sites (expected pairs). "
                f"This indicates a device capability specification error."
            )
            raise ValueError(msg)

        # Generate all possible qubit combinations
        if qubits_num == 1:
            return [(i,) for i in range(num_qubits)]
        if qubits_num == 2:
            # Use coupling map if available
            coupling_map = self._device.coupling_map()
            if coupling_map:
                # Remap coupling map to logical qubit indices
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

    def _submit_to_device(self, circuit: QuantumCircuit, shots: int) -> dict[str, Any]:  # noqa: PLR6301
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
        allowed_ops = {op.name() for op in self._device.operations()}
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
