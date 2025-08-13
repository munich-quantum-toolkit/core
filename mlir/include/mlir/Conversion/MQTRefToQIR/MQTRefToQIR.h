/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/Pass/Pass.h> // from @llvm-project

namespace mqt::ir {

#define GEN_PASS_DECL
#include "mlir/Conversion/MQTRefToQIR/MQTRefToQIR.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/MQTRefToQIR/MQTRefToQIR.h.inc"

} // namespace mqt::ir
