# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

__all__ = ["ClassicalRegister", "QuantumRegister"]

class QuantumRegister:
    """A class to represent a collection of qubits.

    Args:
        start: The starting index of the quantum register.
        size: The number of qubits in the quantum register.
        name: The name of the quantum register. A name will be generated if not provided.
    """

    def __init__(self, start: int, size: int, name: str = "") -> None: ...
    @property
    def start(self) -> int:
        """The index of the first qubit in the quantum register."""

    @property
    def end(self) -> int:
        """Index of the last qubit in the quantum register."""

    @property
    def size(self) -> int:
        """The number of qubits in the quantum register."""

    @property
    def name(self) -> str:
        """The name of the quantum register."""

    def __eq__(self, other: object) -> bool:
        """Check if the quantum register is equal to another quantum register."""

    def __ne__(self, other: object) -> bool:
        """Check if the quantum register is not equal to another quantum register."""

    def __hash__(self) -> int:
        """Return the hash of the quantum register."""

    def __getitem__(self, key: int) -> int:
        """Get the qubit at the specified index."""

    def __contains__(self, qubit: int) -> bool:
        """Check if the quantum register contains a qubit."""

class ClassicalRegister:
    """A class to represent a collection of classical bits.

    Args:
        start: The starting index of the classical register.
        size: The number of bits in the classical register.
        name: The name of the classical register. A name will be generated if not provided.
    """

    def __init__(self, start: int, size: int, name: str = "") -> None: ...
    @property
    def start(self) -> int:
        """The index of the first bit in the classical register."""

    @property
    def end(self) -> int:
        """Index of the last bit in the classical register."""

    @property
    def size(self) -> int:
        """The number of bits in the classical register."""

    @property
    def name(self) -> str:
        """The name of the classical register."""

    def __eq__(self, other: object) -> bool:
        """Check if the classical register is equal to another classical register."""

    def __ne__(self, other: object) -> bool:
        """Check if the classical register is not equal to another classical register."""

    def __hash__(self) -> int:
        """Return the hash of the classical register."""

    def __getitem__(self, key: int) -> int:
        """Get the bit at the specified index."""

    def __contains__(self, bit: int) -> bool:
        """Check if the classical register contains a bit."""
