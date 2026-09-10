/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"

#include "nanobind/nanobind.h"

#include <cstdint>

namespace mqt::bindings {

namespace nb = nanobind;
using namespace nb::literals;

/// Load the optional compiler only when Device.submit is called.
/// NOLINTNEXTLINE(misc-use-internal-linkage): Called by the QDMI module.
void registerCompiler(nb::class_<qdmi::Device>& device) {
  device.def(
      "submit",
      [](const qdmi::Device& self, const nb::object& program, int64_t numShots,
         const nb::kwargs& options) {
        return nb::module_::import_("mqt.core.mlir")
            .attr("submit_program")(
                program, "target"_a = nb::cast(self, nb::rv_policy::reference),
                "num_shots"_a = numShots, **options);
      },
      "program"_a, "num_shots"_a = 1024, "options"_a,
      R"pb(Compile source or submit a compiled program to this device.

Compiled programs must be compatible with this device.

Source inputs accept ``program_format``, ``enable_timing``, and
``enable_statistics``. ``custom1`` through ``custom5`` are passed to the job.
Use ``submit_job`` for raw payloads.)pb");
}

} // namespace mqt::bindings
