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

#include <memory>

void mlir::qdmi::listAvailableDevices(fomac::Session& session,
                                      llvm::raw_ostream& os) {
  os << "Available QDMI devices:\n";
  for (auto dev : session.getDevices()) {
    os << '\t' << dev.getName() << '\n';
  }
}

std::shared_ptr<fomac::Device> mlir::qdmi::getDevice(fomac::Session& session,
                                                     StringRef name) {
  const auto devices = session.getDevices();

  auto it = devices.begin();
  for (; it != devices.end(); ++it) {
    if (it->getName() == name) {
      break;
    }
  }

  if (it == devices.end()) {
    return nullptr;
  }

  return std::make_shared<fomac::Device>(*it);
}
