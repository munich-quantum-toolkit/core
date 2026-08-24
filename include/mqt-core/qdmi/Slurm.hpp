/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Slurm.hpp
 * @brief QDMI adapter for selecting a QDMI device from a Slurm license
 * environment value.
 */

#pragma once

#include "qdmi/Client.hpp"

namespace qdmi::slurm {

/**
 * @brief Opens the QDMI device named by the Slurm license environment.
 * @return A fresh Client session for the selected device.
 * @details The @c SLURM_JOB_LICENSES value must contain exactly one local
 * license. Its name must equal a stable ID visible to the selected QDMI Driver.
 * The optional license count must be one. The device must report
 * @c QDMI_DEVICE_STATUS_IDLE or @c QDMI_DEVICE_STATUS_BUSY.
 * @warning This function uses process-mutable environment data for device
 * selection. It does not verify a Slurm allocation, authenticate the caller,
 * or authorize access to the device. The provider or operating system must
 * enforce access independently.
 * @throws std::runtime_error If the license value is missing, malformed,
 * compound, remote, has a non-unit count, names an unknown device, or names a
 * device in another state.
 */
[[nodiscard]] Device openDeviceFromLicense();

} // namespace qdmi::slurm
