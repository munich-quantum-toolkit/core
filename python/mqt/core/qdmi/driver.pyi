# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Register, discover, and open QDMI devices through MQT Core."""

import os
import pathlib
from collections.abc import Sequence
from typing import overload

import mqt.core.qdmi

class DeviceDefinition:
    """A stable QDMI device registration that can be stored before loading."""

    def __init__(
        self,
        device_id: str,
        library_path: str | os.PathLike,
        prefix: str,
        *,
        base_url: str | None = None,
        token: str | None = None,
        auth_file: str | os.PathLike | None = None,
        auth_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        device_config: str | None = None,
        device_config_file: str | os.PathLike | None = None,
        custom1: str | None = None,
        custom2: str | None = None,
        custom3: str | None = None,
        custom4: str | None = None,
        custom5: str | None = None,
    ) -> None:
        """Create a device definition without loading its native library.

        Args:
            device_id: Stable identifier used by :func:`open_device`.
            library_path: Path to the shared QDMI device library.
            prefix: Function prefix used by the library (for example, ``MY_DEVICE``).
            base_url: Optional base URL for the device API endpoint.
            token: Optional authentication token.
            auth_file: Optional path to an authentication file.
            auth_url: Optional authentication server URL.
            username: Optional authentication username.
            password: Optional authentication password.
            device_config: Optional inline JSON device description.
            device_config_file: Optional device-description JSON file.
            custom1: Optional custom configuration parameter 1.
            custom2: Optional custom configuration parameter 2.
            custom3: Optional custom configuration parameter 3.
            custom4: Optional custom configuration parameter 4.
            custom5: Optional custom configuration parameter 5.
        """

    @property
    def device_id(self) -> str:
        """Stable identifier used to open the device."""

    @property
    def library_path(self) -> pathlib.Path:
        """Path to the native QDMI device library."""

    @property
    def prefix(self) -> str:
        """Prefix used for the QDMI device interface functions."""

class DeviceRegistry:
    """Discover or explicitly register QDMI device definitions."""

    @overload
    def __init__(self) -> None:
        """Discover definitions from the standard configuration sources."""

    @overload
    def __init__(self, definitions: Sequence[DeviceDefinition]) -> None:
        """Create an isolated registry from explicit definitions."""

    @property
    def definitions(self) -> list[DeviceDefinition]:
        """Enabled definitions in stable registration order."""

    @property
    def device_ids(self) -> list[str]:
        """Enabled stable device IDs."""

    def register_device(self, definition: DeviceDefinition, *, replace: bool = False) -> None:
        """Register a definition, optionally replacing the same ID."""

    def register_device_if_absent(self, definition: DeviceDefinition) -> bool:
        """Register a fallback unless its ID exists or is disabled."""

class OpenAllResult:
    """Per-ID successes and failures from opening all registered devices."""

    @property
    def devices(self) -> dict[str, mqt.core.qdmi.Device]:
        """Successfully opened devices keyed by stable ID."""

    @property
    def errors(self) -> dict[str, str]:
        """Error messages for devices that could not be opened."""

class DeviceManager:
    """An immutable registry snapshot that opens fresh device sessions."""

    @overload
    def __init__(self) -> None:
        """Snapshot the current process default registry."""

    @overload
    def __init__(self, registry: DeviceRegistry) -> None:
        """Snapshot an explicit registry."""

    @property
    def definitions(self) -> list[DeviceDefinition]:
        """Definitions in this immutable snapshot."""

    @property
    def device_ids(self) -> list[str]:
        """Stable IDs in this immutable snapshot."""

    def open(
        self,
        device_id: str,
        *,
        base_url: str | None = None,
        token: str | None = None,
        auth_file: str | os.PathLike | None = None,
        auth_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        device_config: str | None = None,
        device_config_file: str | os.PathLike | None = None,
        custom1: str | None = None,
        custom2: str | None = None,
        custom3: str | None = None,
        custom4: str | None = None,
        custom5: str | None = None,
    ) -> mqt.core.qdmi.Device:
        """Open a fresh session for one stable device ID."""

    def open_all(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        auth_file: str | os.PathLike | None = None,
        auth_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        device_config: str | None = None,
        device_config_file: str | os.PathLike | None = None,
        custom1: str | None = None,
        custom2: str | None = None,
        custom3: str | None = None,
        custom4: str | None = None,
        custom5: str | None = None,
    ) -> OpenAllResult:
        """Open all devices and isolate failures by stable ID."""

def register_device(definition: DeviceDefinition, *, replace: bool = False) -> None:
    """Register a QDMI device definition without loading its library.

    Args:
        definition: Definition to validate and store.
        replace: Replace an existing definition for future opens.

    Raises:
        ValueError: If the definition is invalid or its ID is already registered.
    """

def register_device_if_absent(definition: DeviceDefinition) -> bool:
    """Register a valid QDMI device definition if its ID is absent.

    Existing and explicitly disabled IDs are not inserted. Invalid definitions
    still raise.

    Args:
        definition: Definition to validate and store.

    Returns:
        bool: Whether the definition was inserted.

    Raises:
        ValueError: If the definition is invalid.
    """

def registered_device_ids() -> list[str]:
    """Return registered, enabled QDMI device IDs in registration order.

    This includes devices registered at runtime and does not load native device
    libraries or expose their definitions.
    """

def open_device(
    device_id: str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    auth_file: str | os.PathLike | None = None,
    auth_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    device_config: str | None = None,
    device_config_file: str | os.PathLike | None = None,
    custom1: str | None = None,
    custom2: str | None = None,
    custom3: str | None = None,
    custom4: str | None = None,
    custom5: str | None = None,
) -> mqt.core.qdmi.Device:
    """Open a registered QDMI device by stable ID.

    Every call creates a fresh device session while keeping the stable registration
    unchanged. Opening the device loads trusted native device code.

    Args:
        device_id: Stable ID of a registered device.
        base_url: Optional base URL override for the device API endpoint.
        token: Optional authentication token override.
        auth_file: Optional authentication-file override.
        auth_url: Optional authentication server URL override.
        username: Optional authentication username override.
        password: Optional authentication password override.
        device_config: Optional inline JSON device-description override.
        device_config_file: Optional device-description JSON file override.
        custom1: Optional custom configuration parameter 1 override.
        custom2: Optional custom configuration parameter 2 override.
        custom3: Optional custom configuration parameter 3 override.
        custom4: Optional custom configuration parameter 4 override.
        custom5: Optional custom configuration parameter 5 override.

    Returns:
        mqt.core.qdmi.Device: The opened device, ready for direct backend construction.

    Raises:
        IndexError: If the ID is not registered.
        RuntimeError: If the device library cannot be loaded or initialized.
    """
