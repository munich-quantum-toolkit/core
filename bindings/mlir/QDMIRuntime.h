/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "qdmi/Parameters.hpp"

#include "QDMIContext.h"

#include "nanobind/nanobind.h"

#include <optional>

namespace mqt::bindings {

/// Copy Python parameters retained beyond a submission call.
[[nodiscard]] std::optional<runtime::JobParameter>
ownJobParameter(const std::optional<PythonCustomJobParameter>& parameter);

/// Register the shared native execution context.
void registerQDMIRuntime(nanobind::module_& module);

} // namespace mqt::bindings
