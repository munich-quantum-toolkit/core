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

#include "fomac/FoMaC.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

#include <memory>

namespace mlir::qdmi {
void listAvailableDevices(fomac::Session& session,
                          llvm::raw_ostream& os = llvm::outs());

std::shared_ptr<fomac::Device> getDevice(fomac::Session& session,
                                         StringRef name);
} // namespace mlir::qdmi
