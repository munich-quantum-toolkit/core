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

#include "nanobind/nanobind.h"

namespace mqt {
void registerBench(nanobind::module_& module);
void registerDD(const nanobind::module_& module);
void registerQDMI(nanobind::module_& module);
void registerMLIR(nanobind::module_& module);
} // namespace mqt
