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

from typing import TYPE_CHECKING, Any, cast

from qiskit.circuit import Measure, Parameter, QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from mqt.core import fomac

from .capabilities import get_capabilities
from .exceptions import TranslationError, UnsupportedOperationError
from .job import QiskitJob
from .translator import InstructionContext, build_program_ir

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.circuit import Instruction

__all__ = ["QiskitBackend"]

# Maximum number of coupling pairs to generate for two-qubit operations
# This prevents excessive combinations when device has many qubits
MAX_COUPLING_PAIRS = 50


class QiskitBackend(BackendV2):  # type: ignore[misc]
    """A Qiskit BackendV2 adapter for QDMI devices via FoMaC.

    This backend provides neutral QASM3-based program submission to QDMI devices.
    It automatically introspects device capabilities and constructs a Target
    object with supported operations.

    Args:
        device_index: Index of the device to use from fomac.devices() (default: 0).
        use_cache: Whether to use cached device capabilities (default: True).

    Raises:
        RuntimeError: If no FoMaC devices are available.
        IndexError: If device_index is out of range.

    Attributes:
        capabilities_hash: SHA256 hash of the device capabilities snapshot.
    """

    def __init__(self, device_index: int = 0, *, use_cache: bool = True) -> None:
        """Initialize the backend with a FoMaC device.

        Args:
            device_index: Index of the device to use from fomac.devices() (default: 0).
            use_cache: Whether to use cached device capabilities (default: True).

        Raises:
            RuntimeError: If no FoMaC devices are available.
            IndexError: If device_index is out of range.
        """
        super().__init__()

        # Get the device from FoMaC
        devices_list = list(fomac.devices())
        if not devices_list:
            msg = "No FoMaC devices available"
            raise RuntimeError(msg)
        if device_index < 0 or device_index >= len(devices_list):
            msg = f"Device index {device_index} out of range (available: {len(devices_list)})"
            raise IndexError(msg)

        self._device = devices_list[device_index]
        self._capabilities = get_capabilities(self._device, use_cache=use_cache)

        # Build Target from capabilities
        self._target = self._build_target()

        # Set backend options
        self._options = Options(shots=1024)

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

        # Add measurement operation (always supported)
        target.add_instruction(Measure(), {(i,): None for i in range(self._capabilities.num_qubits)})

        # Add operations from device capabilities
        for op_name, op_info in self._capabilities.operations.items():
            # Skip measurement (already added)
            if op_name == "measure":
                continue

            # Map known operations to Qiskit gates
            gate = self._map_operation_to_gate(op_name)
            if gate is None:
                continue  # Skip unsupported operations

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
            if props is not None:
                target.add_instruction(gate, dict.fromkeys(qargs, props))
            else:
                target.add_instruction(gate, dict.fromkeys(qargs))

        return target

    @staticmethod
    def _map_operation_to_gate(op_name: str) -> Instruction | None:
        """Map a device operation name to a Qiskit gate.

        Args:
            op_name: Device operation name.

        Returns:
            Qiskit gate instance or None if not mappable.
        """
        from qiskit.circuit import Parameter
        from qiskit.circuit.library import (
            CHGate,
            CXGate,
            CYGate,
            CZGate,
            DCXGate,
            ECRGate,
            HGate,
            IGate,
            PhaseGate,
            RGate,
            RXGate,
            RXXGate,
            RYGate,
            RYYGate,
            RZGate,
            RZXGate,
            RZZGate,
            SdgGate,
            SGate,
            SwapGate,
            SXdgGate,
            SXGate,
            TdgGate,
            TGate,
            U2Gate,
            U3Gate,
            XGate,
            XXMinusYYGate,
            XXPlusYYGate,
            YGate,
            ZGate,
            iSwapGate,
        )

        # Map known operations to Qiskit gates
        gate_map: dict[str, Instruction] = {
            # Single-qubit Pauli gates
            "x": XGate(),
            "y": YGate(),
            "z": ZGate(),
            "id": IGate(),
            "i": IGate(),
            # Hadamard
            "h": HGate(),
            # Phase gates
            "s": SGate(),
            "sdg": SdgGate(),
            "t": TGate(),
            "tdg": TdgGate(),
            "sx": SXGate(),
            "sxdg": SXdgGate(),
            "p": PhaseGate(Parameter("lambda")),
            "phase": PhaseGate(Parameter("lambda")),
            # Rotation gates (parametric)
            "rx": RXGate(Parameter("theta")),
            "ry": RYGate(Parameter("theta")),
            "rz": RZGate(Parameter("phi")),
            "r": RGate(Parameter("theta"), Parameter("phi")),
            "prx": RGate(Parameter("theta"), Parameter("phi")),
            # Universal gates (parametric)
            "u": U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            "u3": U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            "u2": U2Gate(Parameter("phi"), Parameter("lambda")),
            # Two-qubit gates
            "cx": CXGate(),
            "cnot": CXGate(),
            "cy": CYGate(),
            "cz": CZGate(),
            "ch": CHGate(),
            "swap": SwapGate(),
            "iswap": iSwapGate(),
            "dcx": DCXGate(),
            "ecr": ECRGate(),
            # Two-qubit gates (parametric)
            "rxx": RXXGate(Parameter("theta")),
            "ryy": RYYGate(Parameter("theta")),
            "rzz": RZZGate(Parameter("theta")),
            "rzx": RZXGate(Parameter("theta")),
            "xx_plus_yy": XXPlusYYGate(Parameter("theta"), Parameter("beta")),
            "xx_minus_yy": XXMinusYYGate(Parameter("theta"), Parameter("beta")),
        }

        return gate_map.get(op_name.lower())

    def _get_operation_qargs(self, op_info: Any) -> Sequence[tuple[int, ...]]:  # noqa: ANN401
        """Get the qubit argument tuples for an operation.

        Args:
            op_info: Device operation metadata.

        Returns:
            Sequence of qubit index tuples this operation can act on.
        """
        qubits_num = op_info.qubits_num if op_info.qubits_num is not None else 1
        num_qubits = self._capabilities.num_qubits

        if op_info.sites is not None:
            # Check if sites is a tuple of tuples (local multi-qubit with valid combinations)
            # or a flat tuple (single-qubit or global operations)
            if op_info.sites and isinstance(op_info.sites[0], tuple):
                # Local multi-qubit operation: sites already contains valid combinations
                # Each inner tuple represents a valid combination of qubit indices
                return cast("list[tuple[int, ...]]", list(op_info.sites[:MAX_COUPLING_PAIRS]))

            # Single-qubit or global operation: sites is a flat tuple of individual indices
            if qubits_num == 1:
                return [(site,) for site in op_info.sites]

            # Fallback for multi-qubit operations without proper tuple structure
            # (shouldn't happen, but keep for safety)
            if qubits_num == 2:
                site_indices = list(op_info.sites)
                pairs: list[tuple[int, int]] = [
                    (site_indices[i], site_indices[j])
                    for i in range(len(site_indices))
                    for j in range(i + 1, len(site_indices))
                ]
                cm = self._capabilities.coupling_map
                if cm:
                    cm_set = {(int(a), int(b)) for (a, b) in cm}
                    pairs = [p for p in pairs if p in cm_set or (p[1], p[0]) in cm_set]
                return pairs[:MAX_COUPLING_PAIRS]

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

        # Multi-qubit operations - just use first few valid combinations
        return [tuple(range(qubits_num))]

    def run(self, run_input: QuantumCircuit | list[QuantumCircuit], **options: Any) -> QiskitJob:  # noqa: ANN401
        """Run circuits on the backend.

        Args:
            run_input: Single circuit or list of circuits to execute.
            **options: Backend execution options (e.g., shots).

        Returns:
            QiskitJob object wrapping the execution.

        Raises:
            TranslationError: If circuit translation fails.

        Note:
            This base implementation returns deterministic zero-state counts.
            For actual device execution, subclass this backend and override
            ``_submit_to_device()`` to interface with your QDMI device's
            job submission mechanism.
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
        if shots < 1:
            msg = f"'shots' must be >= 1, got {shots}"
            raise TranslationError(msg)

        # Translate and execute circuits
        try:
            results = []
            for circuit in circuits:
                result = self._execute_circuit(circuit, shots=shots)
                results.append(result)

            # Create and return job
            return QiskitJob(backend=self, circuits=circuits, results=results, shots=shots)

        except Exception as exc:
            if isinstance(exc, (UnsupportedOperationError, TranslationError)):
                raise
            msg = f"Circuit execution failed: {exc}"
            raise TranslationError(msg) from exc

    def _submit_to_device(self, circuit: QuantumCircuit, shots: int) -> dict[str, Any]:
        """Submit circuit to actual QDMI device (extension point).

        This method should be overridden by subclasses to implement actual
        device submission. The base implementation returns deterministic
        zero-state counts for demonstration purposes.

        Args:
            circuit: Circuit to execute.
            shots: Number of shots.

        Returns:
            Result dictionary with counts and metadata.

        Example:
            To implement actual device execution, override this method::

                class MyVendorBackend(QiskitBackend):
                    def _submit_to_device(self, circuit, shots):
                        # Serialize to QASM3 (let Qiskit handle this)
                        from qiskit import qasm3
                        qasm3_str = qasm3.dumps(circuit)

                        # Submit to your device
                        job = self._device.submit_job(
                            program=qasm3_str,
                            program_format=YourProgramFormat.QASM3,
                            num_shots=shots
                        )

                        # Wait for results
                        job.wait()
                        counts = job.get_counts()

                        return {
                            "counts": counts,
                            "shots": shots,
                            "success": True,
                            "metadata": {"job_id": job.id}
                        }
        """
        # Default implementation: return deterministic zero-state counts
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
            TranslationError: If circuit translation fails or contains unbound parameters.
            UnsupportedOperationError: If circuit contains unsupported operations.
        """
        # Build instruction contexts from circuit
        contexts = []
        for instruction in circuit.data:
            op_name = instruction.operation.name
            qubits = [circuit.find_bit(q).index for q in instruction.qubits]
            clbits = [circuit.find_bit(c).index for c in instruction.clbits] if instruction.clbits else None

            # Extract parameters
            params = None
            if hasattr(instruction.operation, "params") and instruction.operation.params:
                # Convert parameters to floats
                param_list = []
                from qiskit.circuit import ParameterExpression

                for p in instruction.operation.params:
                    # Explicitly reject unbound symbols
                    if isinstance(p, Parameter):
                        msg = f"Unbound parameter '{p.name}' in circuit"
                        raise TranslationError(msg)
                    if isinstance(p, ParameterExpression) and getattr(p, "parameters", None):
                        # ParameterExpression with free parameters is unbound
                        names = ", ".join(sorted(par.name for par in p.parameters))
                        msg = f"Unbound parameter expression with symbols: {names}"
                        raise TranslationError(msg)
                    try:
                        param_list.append(float(p))
                    except Exception as conv_exc:
                        msg = f"Non-numeric parameter value: {p!r}"
                        raise TranslationError(msg) from conv_exc
                params = param_list

            ctx = InstructionContext(
                name=op_name,
                qubits=qubits,
                params=params,
                clbits=clbits,
            )
            contexts.append(ctx)

        # Build program IR
        allowed_ops = set(self._capabilities.operations.keys())
        allowed_ops.add("measure")

        try:
            from .types import IRValidationError

            program_ir = build_program_ir(
                name=circuit.name or "circuit",
                instruction_contexts=contexts,
                allowed_operations=allowed_ops,
                num_qubits=self._capabilities.num_qubits,
                pseudo_ops=["barrier"],
            )
        except IRValidationError as exc:
            msg = f"Failed to build program IR: {exc}"
            raise UnsupportedOperationError(msg) from exc

        # Store program_ir in metadata for potential use by subclasses
        result = self._submit_to_device(circuit, shots)
        result.setdefault("metadata", {})
        result["metadata"]["program_name"] = program_ir.name

        return result
