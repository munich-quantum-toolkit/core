# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Device utilities for MQT Catalyst Plugin.

This module provides utilities to configure PennyLane devices for use with the MQT plugin,
preventing Catalyst from decomposing gates into quantum.unitary operations with matrix parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pennylane as qml


def configure_device_for_mqt(device: qml.devices.Device) -> qml.devices.Device:
    """Configure a PennyLane device to work optimally with the MQT plugin.

    This function modifies device capabilities to prevent Catalyst from decomposing
    controlled gates (like qml.ctrl(PauliX)) into quantum.unitary operations with
    explicit matrix parameters. Instead, gates remain as quantum.custom operations
    (like "CNOT") that can be converted by the MQT plugin.

    The modification uses several techniques:
    1. Removes QubitUnitary from device.operations (prevents matrix decomposition path)
    2. Clears _to_matrix_ops (avoids Catalyst validation requiring QubitUnitary support)
    3. Uses qjit_capabilities hook (injects modified capabilities before QJITDevice init)

    Args:
        device: A PennyLane device instance to configure.

    Returns:
        The same device instance with modified capabilities.

    Raises:
        ValueError: If the device does not have a config_filepath attribute set.

    Example:
        >>> import pennylane as qml
        >>> from mqt.core.plugins.catalyst.device import configure_device_for_mqt
        >>> dev = qml.device("lightning.qubit", wires=2)
        >>> dev = configure_device_for_mqt(dev)
        >>> @qml.qnode(dev)
        ... def circuit():
        ...     qml.ctrl(qml.PauliX(wires=0), control=1)  # Will become CNOT, not matrix
        ...     return qml.state()
    """
    from pennylane.devices.capabilities import DeviceCapabilities

    # Load the original capabilities from the device's config file
    if hasattr(device, "config_filepath") and device.config_filepath is not None:
        toml_file = device.config_filepath
    else:
        msg = "Device does not have a config_filepath attribute set."
        raise ValueError(msg)

    caps = DeviceCapabilities.from_toml_file(toml_file, "qjit")

    # Remove QubitUnitary from operations to prevent matrix decomposition
    if "QubitUnitary" in caps.operations:
        del caps.operations["QubitUnitary"]

    # Clear _to_matrix_ops to avoid Catalyst validation at qjit_device.py:322
    # which requires QubitUnitary support if _to_matrix_ops is set
    if hasattr(device, "_to_matrix_ops"):
        device._to_matrix_ops = set()  # noqa: SLF001

    # Set the qjit_capabilities hook so QJITDevice uses our modified capabilities
    # This bypasses the normal TOML loading in _load_device_capabilities
    device.qjit_capabilities = caps

    return device


def get_device(device_name: str, **kwargs: Any) -> qml.devices.Device:  # noqa: ANN401
    """Create and configure a PennyLane device for use with the MQT plugin.

    This is a convenience function that creates a device and automatically configures
    it to work optimally with the MQT plugin, preventing unnecessary decomposition to
    unitary matrices.

    Args:
        device_name: The name of the PennyLane device (e.g., "lightning.qubit").
        **kwargs: Additional keyword arguments passed to qml.device().

    Returns:
        A configured PennyLane device ready for use with MQT conversion passes.

    Example:
        >>> from mqt.core.plugins.catalyst.device import get_device
        >>> dev = get_device("lightning.qubit", wires=2)
        >>> @qml.qnode(dev)
        ... def circuit():
        ...     qml.ctrl(qml.PauliX(wires=0), control=1)
        ...     return qml.state()
    """
    import pennylane as qml

    device = qml.device(device_name, **kwargs)
    return configure_device_for_mqt(device)


__all__ = ["configure_device_for_mqt", "get_device"]
