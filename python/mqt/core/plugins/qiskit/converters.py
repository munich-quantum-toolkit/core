# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit converters for various program formats.

This module provides conversion functions from Qiskit QuantumCircuit
to various device-specific program formats.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit.library import Barrier, CZGate, Measure, RGate

from .exceptions import TranslationError, UnsupportedOperationError

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from ... import fomac

__all__ = ["qiskit_to_iqm_json"]


def __dir__() -> list[str]:
    return __all__


def qiskit_to_iqm_json(circuit: QuantumCircuit, device: fomac.Device) -> str:
    """Convert a Qiskit :class:`~qiskit.circuit.QuantumCircuit` to IQM JSON format.

    The IQM JSON format is a device-specific format that encodes quantum operations
    as JSON objects with site names, operation names, and arguments.

    Note:
        The conversion currently supports only operations that are natively
        supported by the IQM hardware. Unsupported operations will raise
        :class:`UnsupportedOperationError`.

    Args:
        circuit: The Qiskit quantum circuit to convert.
        device: The FoMaC device providing site mapping and metadata.

    Returns:
        JSON string representation of the circuit in IQM format.

    Raises:
        UnsupportedOperationError: If the circuit contains operations not supported
            by IQM hardware.
        TranslationError: If the conversion process fails.

    Examples:
        >>> from qiskit import QuantumCircuit
        >>> import numpy as np
        >>> from mqt.core.plugins.qiskit.converters import qiskit_to_iqm_json
        >>> qc = QuantumCircuit(2, 2)
        >>> qc.r(np.pi / 2, 0, 0)
        >>> qc.cz(0, 1)
        >>> qc.measure_all()
        >>> json_str = qiskit_to_iqm_json(qc, device)
    """

    def _raise_error(exception_type: type[Exception], message: str) -> None:
        """Helper to raise exceptions (satisfies TRY301).

        Args:
            exception_type: The type of exception to raise.
            message: The error message.
        """
        raise exception_type(message)

    try:
        # Check for unbound parameters
        if circuit.parameters:
            param_names = ", ".join(sorted(p.name for p in circuit.parameters))
            msg = (
                f"Circuit contains unbound parameters: {param_names}. "
                "All parameters must be bound to numeric values before conversion to IQM JSON. "
                "Use circuit.assign_parameters() to bind parameters."
            )
            _raise_error(UnsupportedOperationError, msg)

        sites = device.sites()
        instructions: list[dict[str, Any]] = []

        for instruction in circuit.data:
            operation, qargs, cargs = instruction.operation, instruction.qubits, instruction.clbits

            # R gate (PRX in IQM terminology)
            if isinstance(operation, RGate):
                angle_t = float(operation.params[0] / (2 * np.pi))
                phase_t = float(operation.params[1] / (2 * np.pi))
                qubit_index = circuit.find_bit(qargs[0]).index
                instructions.append({
                    "name": "prx",
                    "locus": [sites[qubit_index].name()],
                    "args": {
                        "angle_t": angle_t,
                        "phase_t": phase_t,
                    },
                })

            # CZ gate
            elif isinstance(operation, CZGate):
                qubit_index1 = circuit.find_bit(qargs[0]).index
                qubit_index2 = circuit.find_bit(qargs[1]).index
                instructions.append({
                    "name": "cz",
                    "locus": [
                        sites[qubit_index1].name(),
                        sites[qubit_index2].name(),
                    ],
                    "args": {},
                })

            # Barrier
            elif isinstance(operation, Barrier):
                qubit_indices: list[int] = []
                for qubit in qargs:
                    qubit_index = circuit.find_bit(qubit).index
                    qubit_indices.append(qubit_index)
                instructions.append({
                    "name": "barrier",
                    "locus": [sites[i].name() for i in qubit_indices],
                    "args": {},
                })

            # Measure
            elif isinstance(operation, Measure):
                clbit = cargs[0]
                bitloc = circuit.find_bit(clbit)

                # Check if classical bit is part of a register
                if not bitloc.registers:
                    msg = (
                        "Measurement of unregistered classical bit is unsupported by IQM JSON export. "
                        "All classical bits must be part of a ClassicalRegister."
                    )
                    _raise_error(TranslationError, msg)

                creg = bitloc.registers[0][0]
                creg_idx = circuit.cregs.index(creg)
                clbit_index = bitloc.registers[0][1]
                key = f"{creg.name}_{len(creg)}_{creg_idx}_{clbit_index}"
                qubit_index = circuit.find_bit(qargs[0]).index
                instructions.append({
                    "name": "measure",
                    "locus": [sites[qubit_index].name()],
                    "args": {
                        "key": key,
                    },
                })

            # Unsupported operation
            else:
                msg = f"Operation '{operation.name}' is not supported in IQM JSON format"
                _raise_error(UnsupportedOperationError, msg)

        program: dict[str, Any] = {
            "name": circuit.name or "circuit",
            "metadata": {},
            "instructions": instructions,
        }

        return json.dumps(program)

    except UnsupportedOperationError:
        raise
    except Exception as exc:
        msg = f"Failed to convert circuit to IQM JSON: {exc}"
        raise TranslationError(msg) from exc
