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

#include "mqt_core_native_compiler_export.h"

#include <cstddef>
#include <functional>

namespace qdmi::dd {
struct WorkerRequest;
struct WorkerResponse;

/// Execute OpenQASM and serialize retained DD state inside the compiler
/// library.
extern "C" MQT_CORE_NATIVE_COMPILER_EXPORT void
mqtCoreRunWorkerQasm(const WorkerRequest& request, WorkerResponse& response,
                     const std::function<size_t(size_t)>& requestWorkers);
} // namespace qdmi::dd
