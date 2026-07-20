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

#include <llvm/ADT/STLExtras.h>

#include <array>

namespace mlir::oq3::frontend {
namespace {

using Availability = GateAvailability;

constexpr std::array CATALOG{
    GateCatalogEntry{"gphase", "gphase", 1, 0, 0, Availability::Language},
    GateCatalogEntry{"U", "U", 3, 0, 1, Availability::Language},
    GateCatalogEntry{"id", "id", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"x", "x", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"y", "y", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"z", "z", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"h", "h", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"s", "s", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"sdg", "sdg", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"t", "t", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"tdg", "tdg", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"sx", "sx", 0, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"p", "p", 1, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"rx", "rx", 1, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"ry", "ry", 1, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"rz", "rz", 1, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"r", "r", 2, 0, 1, Availability::StandardLibrary},
    GateCatalogEntry{"swap", "swap", 0, 0, 2, Availability::StandardLibrary},
    GateCatalogEntry{"cx", "x", 0, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"cy", "y", 0, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"cz", "z", 0, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"ch", "h", 0, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"cp", "p", 1, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"crx", "rx", 1, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"cry", "ry", 1, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"crz", "rz", 1, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"ccx", "x", 0, 2, 1, Availability::StandardLibrary},
    GateCatalogEntry{"cswap", "swap", 0, 1, 2, Availability::StandardLibrary},
    GateCatalogEntry{"cu", "U", 4, 1, 1, Availability::StandardLibrary},
    GateCatalogEntry{"u1", "p", 1, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"cu1", "p", 1, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"phase", "p", 1, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"cphase", "p", 1, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"u2", "u2", 2, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"u3", "U", 3, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"u", "U", 3, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"cu3", "U", 3, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"CX", "x", 0, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"cnot", "x", 0, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"c3x", "x", 0, 3, 1, Availability::Compatibility},
    GateCatalogEntry{"c4x", "x", 0, 4, 1, Availability::Compatibility},
    GateCatalogEntry{"csx", "sx", 0, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"sxdg", "sxdg", 0, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"c3sqrtx", "sxdg", 0, 3, 1, Availability::Compatibility},
    GateCatalogEntry{"prx", "r", 2, 0, 1, Availability::Compatibility},
    GateCatalogEntry{"cr", "r", 2, 1, 1, Availability::Compatibility},
    GateCatalogEntry{"fredkin", "swap", 0, 1, 2, Availability::Compatibility},
    GateCatalogEntry{"iswap", "iswap", 0, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"iswapdg", "iswap", 0, 0, 2, Availability::Compatibility,
                     false, true},
    GateCatalogEntry{"dcx", "dcx", 0, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"ecr", "ecr", 0, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"rxx", "rxx", 1, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"ryy", "ryy", 1, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"rzx", "rzx", 1, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"rzz", "rzz", 1, 0, 2, Availability::Compatibility},
    GateCatalogEntry{"xx_plus_yy", "xx_plus_yy", 2, 0, 2,
                     Availability::Compatibility},
    GateCatalogEntry{"xx_minus_yy", "xx_minus_yy", 2, 0, 2,
                     Availability::Compatibility},
    GateCatalogEntry{"mcx", "x", 0, 1, 1, Availability::Compatibility, true},
    GateCatalogEntry{"mcx_gray", "x", 0, 1, 1, Availability::Compatibility,
                     true},
    GateCatalogEntry{"mcx_vchain", "x", 0, 1, 1, Availability::Compatibility,
                     true},
    GateCatalogEntry{"mcx_recursive", "x", 0, 1, 1, Availability::Compatibility,
                     true},
    GateCatalogEntry{"mcphase", "p", 1, 1, 1, Availability::Compatibility,
                     true},
};

} // namespace

llvm::ArrayRef<GateCatalogEntry> getGateCatalog() { return CATALOG; }

const GateCatalogEntry* lookupGate(const llvm::StringRef name) {
  const auto iterator = llvm::find_if(
      CATALOG, [&](const GateCatalogEntry& gate) { return gate.name == name; });
  return iterator == CATALOG.end() ? nullptr : &*iterator;
}

} // namespace mlir::oq3::frontend
