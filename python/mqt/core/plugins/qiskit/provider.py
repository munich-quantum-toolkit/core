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

from ... import fomac
from .backend import QDMIBackend

__all__ = ["QDMIProvider"]


def __dir__() -> list[str]:
    return __all__


class QDMIProvider:
    """Provider for QDMI devices accessed via FoMaC.

    This provider discovers and manages QDMI devices that are available through
    the FoMaC layer. It provides a Qiskit-idiomatic interface for device
    discovery and backend instantiation.

    Examples:
        List all available backends:

        >>> from mqt.core.plugins.qiskit import QDMIProvider
        >>> provider = QDMIProvider()
        >>> for backend in provider.backends():
        ...     print(f"{backend.name}: {backend.target.num_qubits} qubits")

        Get a specific backend by name:

        >>> backend = provider.get_backend("MQT Core DDSIM QDMI Device")

        Create a provider with authentication:

        >>> provider = QDMIProvider(token="my_token")
        >>> # or with username and password
        >>> provider = QDMIProvider(username="user", password="pass")
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        auth_file: str | None = None,
        auth_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        project_id: str | None = None,
        **session_kwargs: str,
    ) -> None:
        """Initialize the QDMI provider.

        Args:
            token: Authentication token for the session.
            auth_file: Path to file containing authentication information.
            auth_url: URL to authentication server.
            username: Username for authentication.
            password: Password for authentication.
            project_id: Project ID for the session.
            session_kwargs: Optional additional keyword arguments for Session initialization.
        """
        kwargs = {
            "token": token,
            "auth_file": auth_file,
            "auth_url": auth_url,
            "username": username,
            "password": password,
            "project_id": project_id,
        }
        if session_kwargs:
            kwargs.update(session_kwargs)

        self._session = fomac.Session(**kwargs)
        self._backends = [
            QDMIBackend(device=d, provider=self) for d in self._session.get_devices() if QDMIBackend.is_convertible(d)
        ]

    def backends(self, name: str | None = None) -> list[QDMIBackend]:
        """Return all available backends, optionally filtered by name substring.

        Args:
            name: If provided, return only backends whose name contains this substring.

        Returns:
            List of QiskitBackend instances. Empty list if name specified but not found.

        Examples:
            Get all backends:

            >>> provider = QDMIProvider()
            >>> all_backends = provider.backends()

            Filter backends by name substring:

            >>> na_backends = provider.backends(name="NA")  # matches "MQT NA Default QDMI Device"
            >>> qdmi_backends = provider.backends(name="QDMI")  # matches all devices with "QDMI" in name
        """
        # Filter by name substring if specified
        if name is not None:
            return [b for b in self._backends if name in b.name]

        return self._backends

    def get_backend(self, name: str) -> QDMIBackend:
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
            >>> backend = provider.get_backend("MQT Core DDSIM QDMI Device")
        """
        for backend in self._backends:
            if backend.name == name:
                return backend

        msg = f"No backend found with name '{name}'"
        raise ValueError(msg)

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return f"<QDMIProvider(backends={len(self._backends)})>"
