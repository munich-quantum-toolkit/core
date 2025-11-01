# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Provider for Qiskit integration.

This module provides a provider interface for discovering and accessing QDMI
devices through Qiskit's BackendV2 interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.core import fomac

if TYPE_CHECKING:
    from .backend import QiskitBackend

__all__ = ["QDMIProvider"]


class QDMIProvider:
    """Provider for QDMI devices accessed via FoMaC.

    This provider discovers and manages QDMI devices that are available through
    the FoMaC layer. It provides a Qiskit-idiomatic interface for device
    discovery and backend instantiation.

    Examples:
        List all available backends:

        >>> from mqt.core.qdmi.qiskit import QDMIProvider
        >>> provider = QDMIProvider()
        >>> backends = provider.backends()
        >>> for backend in backends:
        ...     print(f"{backend.name}: {backend.target.num_qubits} qubits")

        Get a specific backend by name:

        >>> backend = provider.get_backend("MQT NA Default QDMI Device")
    """

    def __init__(self) -> None:
        """Initialize the QDMI provider."""

    def backends(self, name: str | None = None) -> list[QiskitBackend]:
        """Return all available backends, optionally filtered by name.

        Args:
            name: If provided, return only the backend with this exact name.

        Returns:
            List of QiskitBackend instances. Empty list if name specified but not found.

        Examples:
            Get all backends:

            >>> provider = QDMIProvider()
            >>> all_backends = provider.backends()

            Get a specific backend by name:

            >>> backends = provider.backends(name="MQT NA Default QDMI Device")
        """
        from .backend import QiskitBackend

        # Get all devices from FoMaC
        devices = list(fomac.devices())

        # Filter by name if specified
        if name is not None:
            devices = [d for d in devices if d.name() == name]

        # Create backend instances
        return [QiskitBackend(device=device, provider=self) for device in devices]

    def get_backend(self, name: str) -> QiskitBackend:
        """Get a single backend by name.

        Args:
            name: Name of the backend to retrieve.

        Returns:
            QiskitBackend instance.

        Raises:
            ValueError: If no matching backend found.

        Examples:
            Get a specific backend:

            >>> provider = QDMIProvider()
            >>> backend = provider.get_backend("MQT NA Default QDMI Device")
        """
        backends = self.backends(name=name)

        if not backends:
            msg = f"No backend found with name '{name}'"
            raise ValueError(msg)

        return backends[0]

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        num_backends = len(list(fomac.devices()))
        return f"<QDMIProvider(backends={num_backends})>"
