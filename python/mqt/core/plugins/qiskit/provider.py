# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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

import warnings
from typing import TYPE_CHECKING

from ...qdmi.driver import registered_device_ids
from .backend import QDMIBackend
from .exceptions import UnsupportedDeviceError, UnsupportedFormatError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Unpack

    from ...typing import QDMISessionParameters

__all__ = ["QDMIProvider"]


def __dir__() -> list[str]:
    return __all__


class QDMIProvider:
    """Provider for registered QDMI devices.

    This provider discovers registered QDMI devices lazily and adapts
    Qiskit-compatible devices as backends.

    Examples:
        List all available backends:

        >>> from mqt.core.plugins.qiskit import QDMIProvider
        >>> provider = QDMIProvider()
        >>> for backend in provider.backends():
        ...     print(f"{backend.name}: {backend.target.num_qubits} qubits")

        Get a specific backend by name:

        >>> backend = provider.get_backend("MQT Core DDSIM QDMI Device")

        Configure credentials through the selected device's persistent
        configuration or provider-specific environment.
    """

    @staticmethod
    def device_ids() -> list[str]:
        """Return the current registered QDMI device IDs without opening them."""
        return registered_device_ids()

    def backends(self, name: str | None = None) -> list[QDMIBackend]:
        """Return all available backends, optionally filtered by name substring.

        Args:
            name: If provided, return only backends whose name contains this substring.

        Returns:
            Compatible QDMI backends. The list is empty when no name matches.

        Examples:
            Get all backends:

            >>> provider = QDMIProvider()
            >>> all_backends = provider.backends()

            Filter backends by name substring:

            >>> ddsim_backends = provider.backends(name="DDSIM")
            >>> qdmi_backends = provider.backends(name="QDMI")
        """
        return list(self._iter_backends(name))

    def _iter_backends(self, name: str | None = None) -> Iterator[QDMIBackend]:
        """Open available registered backends one at a time.

        Args:
            name: If provided, yield only backends whose name contains this substring.

        Yields:
            Each backend whose registered device can be opened.
        """
        for device_id in self.device_ids():
            try:
                backend = self.get_backend_by_device_id(device_id)
            except (UnsupportedDeviceError, UnsupportedFormatError):
                continue
            except (IndexError, RuntimeError, ValueError):
                warnings.warn(
                    f"Could not open registered QDMI device '{device_id}'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue
            if name is None or (backend.name is not None and name in backend.name):
                yield backend

    def get_backend_by_device_id(
        self,
        device_id: str,
        **session_parameters: Unpack[QDMISessionParameters],
    ) -> QDMIBackend:
        """Open one fresh backend for an exact stable QDMI device ID.

        Returns:
            A backend for a fresh device session.
        """
        return QDMIBackend.from_device_id(device_id, provider=self, **session_parameters)

    def get_backend(self, name: str) -> QDMIBackend:
        """Get a single backend by name.

        Args:
            name: Name of the backend to retrieve.

        Returns:
            The matching QDMI backend.

        Raises:
            ValueError: If no matching backend found.

        Examples:
            Get a specific backend:

            >>> provider = QDMIProvider()
            >>> backend = provider.get_backend("MQT Core DDSIM QDMI Device")
        """
        for backend in self._iter_backends(name):
            if backend.name == name:
                return backend

        msg = f"No backend found with name '{name}'"
        raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return f"<QDMIProvider(devices={len(self.device_ids())})>"
