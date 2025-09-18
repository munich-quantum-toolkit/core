# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# under Windows, make sure to add the appropriate DLL directory to the PATH
if sys.platform == "win32":

    def _dll_patch() -> None:
        """Add the DLL directory to the PATH."""
        import sysconfig

        bin_dir = Path(sysconfig.get_paths()["purelib"]) / "mqt" / "core" / "bin"
        os.add_dll_directory(str(bin_dir))

    _dll_patch()
    del _dll_patch

from typing import TYPE_CHECKING

from ._version import version as __version__
from ._version import version_tuple as version_info
from .ir import QuantumComputation

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def load(input_circuit: QuantumComputation | str | os.PathLike[str] | QuantumCircuit) -> QuantumComputation:
    """Load a quantum circuit from any supported format as a :class:`~mqt.core.ir.QuantumComputation`.

    Args:
        input_circuit: The input circuit to translate to a :class:`~mqt.core.ir.QuantumComputation`.
                       This can be a :class:`~mqt.core.ir.QuantumComputation` itself,
                       a file name to any of the supported file formats,
                       an OpenQASM (2.0 or 3.0) string, or
                       a Qiskit :class:`~qiskit.circuit.QuantumCircuit`.

    Returns:
        The :class:`~mqt.core.ir.QuantumComputation`.

    Raises:
        FileNotFoundError: If the input circuit is a file name and the file does not exist.
    """
    if isinstance(input_circuit, QuantumComputation):
        return input_circuit

    if isinstance(input_circuit, (str, os.PathLike)):
        input_str = str(input_circuit)
        max_filename_length = 255 if os.name == "nt" else os.pathconf("/", "PC_NAME_MAX")
        if len(input_str) > max_filename_length or not Path(input_circuit).is_file():
            if isinstance(input_circuit, os.PathLike) or "OPENQASM" not in input_circuit:
                msg = f"File {input_circuit} does not exist."
                raise FileNotFoundError(msg)
            # otherwise, we assume that this is a QASM string
            return QuantumComputation.from_qasm_str(input_str)

        return QuantumComputation.from_qasm(input_str)

    # At this point, we know that the input is a Qiskit QuantumCircuit
    from .plugins.qiskit import qiskit_to_mqt

    return qiskit_to_mqt(input_circuit)


__all__ = ["__version__", "load", "version_info"]
