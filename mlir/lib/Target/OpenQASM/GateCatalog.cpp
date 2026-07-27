/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/GateCatalog.h"

#include <llvm/ADT/StringMap.h>

#include <array>

namespace mlir::oq3::frontend {
namespace {

using Availability = GateAvailability;
constexpr auto Std = Availability::StandardLibrary;
constexpr auto QELib1 = Availability::QELib1;
constexpr auto Both = Availability::StandardLibraryAndQELib1;
constexpr auto Compat = Availability::Compatibility;

constexpr std::array CATALOG{
    GateCatalogEntry{"gphase", "gphase", 1, 0, 0, Availability::Language},
    GateCatalogEntry{"U", "U", 3, 0, 1, Availability::Language},
    GateCatalogEntry{"id", "id", 0, 0, 1, Both},
    GateCatalogEntry{"x", "x", 0, 0, 1, Both},
    GateCatalogEntry{"y", "y", 0, 0, 1, Both},
    GateCatalogEntry{"z", "z", 0, 0, 1, Both},
    GateCatalogEntry{"h", "h", 0, 0, 1, Both},
    GateCatalogEntry{"s", "s", 0, 0, 1, Both},
    GateCatalogEntry{"sdg", "sdg", 0, 0, 1, Both},
    GateCatalogEntry{"t", "t", 0, 0, 1, Both},
    GateCatalogEntry{"tdg", "tdg", 0, 0, 1, Both},
    GateCatalogEntry{"sx", "sx", 0, 0, 1, Std},
    GateCatalogEntry{"p", "p", 1, 0, 1, Std},
    GateCatalogEntry{"rx", "rx", 1, 0, 1, Both},
    GateCatalogEntry{"ry", "ry", 1, 0, 1, Both},
    GateCatalogEntry{"rz", "rz", 1, 0, 1, Both},
    GateCatalogEntry{"r", "r", 2, 0, 1, Compat},
    GateCatalogEntry{"swap", "swap", 0, 0, 2, Std},
    GateCatalogEntry{"cx", "x", 0, 1, 1, Both},
    GateCatalogEntry{"cy", "y", 0, 1, 1, Both},
    GateCatalogEntry{"cz", "z", 0, 1, 1, Both},
    GateCatalogEntry{"ch", "h", 0, 1, 1, Both},
    GateCatalogEntry{"cp", "p", 1, 1, 1, Std},
    GateCatalogEntry{"crx", "rx", 1, 1, 1, Std},
    GateCatalogEntry{"cry", "ry", 1, 1, 1, Std},
    GateCatalogEntry{"crz", "rz", 1, 1, 1, Both},
    GateCatalogEntry{"ccx", "x", 0, 2, 1, Both},
    GateCatalogEntry{"cswap", "swap", 0, 1, 2, Std},
    GateCatalogEntry{"cu", "U", 4, 1, 1, Std},
    GateCatalogEntry{"u1", "p", 1, 0, 1, Both},
    GateCatalogEntry{"cu1", "p", 1, 1, 1, QELib1},
    GateCatalogEntry{"phase", "p", 1, 0, 1, Std},
    GateCatalogEntry{"cphase", "p", 1, 1, 1, Std},
    GateCatalogEntry{"u2", "u2", 2, 0, 1, Both},
    GateCatalogEntry{"u3", "U", 3, 0, 1, Both},
    GateCatalogEntry{"u", "U", 3, 0, 1, Compat},
    GateCatalogEntry{"cu3", "U", 3, 1, 1, QELib1},
    GateCatalogEntry{"CX", "x", 0, 1, 1, Std},
    GateCatalogEntry{"cnot", "x", 0, 1, 1, Compat},
    GateCatalogEntry{"c3x", "x", 0, 3, 1, Compat},
    GateCatalogEntry{"c4x", "x", 0, 4, 1, Compat},
    GateCatalogEntry{"csx", "sx", 0, 1, 1, Compat},
    GateCatalogEntry{"sxdg", "sxdg", 0, 0, 1, Compat},
    GateCatalogEntry{"c3sqrtx", "sxdg", 0, 3, 1, Compat},
    GateCatalogEntry{"prx", "r", 2, 0, 1, Compat},
    GateCatalogEntry{"cr", "r", 2, 1, 1, Compat},
    GateCatalogEntry{"fredkin", "swap", 0, 1, 2, Compat},
    GateCatalogEntry{"iswap", "iswap", 0, 0, 2, Compat},
    GateCatalogEntry{"iswapdg", "iswap", 0, 0, 2, Compat, false, true},
    GateCatalogEntry{"dcx", "dcx", 0, 0, 2, Compat},
    GateCatalogEntry{"ecr", "ecr", 0, 0, 2, Compat},
    GateCatalogEntry{"rccx", "rccx", 0, 0, 3, Compat},
    GateCatalogEntry{"rxx", "rxx", 1, 0, 2, Compat},
    GateCatalogEntry{"ryy", "ryy", 1, 0, 2, Compat},
    GateCatalogEntry{"rzx", "rzx", 1, 0, 2, Compat},
    GateCatalogEntry{"rzz", "rzz", 1, 0, 2, Compat},
    GateCatalogEntry{"xx_plus_yy", "xx_plus_yy", 2, 0, 2, Compat},
    GateCatalogEntry{"xx_minus_yy", "xx_minus_yy", 2, 0, 2, Compat},
    GateCatalogEntry{"mcx", "x", 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcx_gray", "x", 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcx_vchain", "x", 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcx_recursive", "x", 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcphase", "p", 1, 1, 1, Compat, true},
};

} // namespace

llvm::ArrayRef<GateCatalogEntry> getGateCatalog() { return CATALOG; }

const GateCatalogEntry* lookupGate(const llvm::StringRef name) {
  static const auto index = [] {
    llvm::StringMap<const GateCatalogEntry*> result;
    for (const auto& gate : CATALOG) {
      result.try_emplace(gate.name, &gate);
    }
    return result;
  }();
  return index.lookup(name);
}

} // namespace mlir::oq3::frontend
