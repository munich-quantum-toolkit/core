/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Slurm.hpp"

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace mqt::bindings {

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerSlurm(nb::module_& qdmiModule) {
  auto slurm = qdmiModule.def_submodule(
      "slurm", "Open a QDMI device named by the Slurm license environment.");
  slurm.def("open_device_from_license", &qdmi::slurm::openDeviceFromLicense,
            R"pb(Open the QDMI device named by the Slurm license environment.

``SLURM_JOB_LICENSES`` must contain one local license whose name equals a stable
ID visible to the selected QDMI Driver. The optional count must be one. The
function opens a fresh Client session and accepts device status ``IDLE`` or
``BUSY``. It does not apply job-specific QDMI configuration or credentials.

Warning:
    ``SLURM_JOB_LICENSES`` is process-mutable. This function uses it only for
    device selection. It does not verify a Slurm allocation, authenticate the
    caller, or authorize device access. The provider or operating system must
    enforce access independently.

Returns:
    mqt.core.qdmi.Device: The fresh device session.

Raises:
    RuntimeError: If the license value or named device does not satisfy this
        contract.)pb");
}

} // namespace mqt::bindings
