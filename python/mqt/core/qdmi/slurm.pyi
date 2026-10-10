# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Open a QDMI device named by the Slurm license environment."""

import mqt.core.qdmi

def open_device_from_license() -> mqt.core.qdmi.Device:
    """Open the QDMI device named by the Slurm license environment.

    ``SLURM_JOB_LICENSES`` must contain one local license whose name equals a stable
    ID registered with MQT Core's builtin QDMI driver. The optional count must be
    one. The function opens a fresh session for that device and accepts status
    ``IDLE`` or ``BUSY``. Other devices are not initialized. Configure authentication
    through each device's catalogue entry or documented environment variables.

    Warning:
        ``SLURM_JOB_LICENSES`` is process-mutable. This function uses it only for
        device selection. It does not verify a Slurm allocation, authenticate the
        caller, or authorize device access. The device implementation or operating system must
        enforce access independently.

    Returns:
        mqt.core.qdmi.Device: The selected device.

    Raises:
        RuntimeError: If the license value or named device does not satisfy this
            contract.
    """
