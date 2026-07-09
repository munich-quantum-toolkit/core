/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/Qdmi.h"

#include "fomac/FoMaC.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <memory>

void mlir::qdmi::listAvailableDevices(fomac::Session& session,
                                      llvm::raw_ostream& os) {
  os << "Available QDMI devices:\n";
  for (const auto& dev : session.getDevices()) {
    os << '\t' << dev.getName() << '\n';
  }
}

std::shared_ptr<fomac::Device> mlir::qdmi::getDevice(fomac::Session& session,
                                                     StringRef name) {
  const auto devices = session.getDevices();
  const auto it = std::ranges::find_if(
      devices, [&](const auto& dev) { return dev.getName() == name; });
  return it != devices.end() ? std::make_shared<fomac::Device>(*it) : nullptr;
}
