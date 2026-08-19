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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>

#include <array>

namespace mlir::oq3::frontend {
namespace {

using Availability = GateAvailability;
constexpr auto STD = Availability::StandardLibrary;
constexpr auto QE_LIB1 = Availability::QELib1;
constexpr auto BOTH = Availability::StandardLibraryAndQELib1;
constexpr auto COMPAT = Availability::Compatibility;
using Lowering = GateLowering;

const std::array CATALOG{
    GateCatalogEntry{"gphase", Lowering::GPhase, 0, Availability::Language},
    GateCatalogEntry{"U", Lowering::BuiltinU, 0, Availability::Language},
    GateCatalogEntry{"id", Lowering::Id, 0, BOTH},
    GateCatalogEntry{"x", Lowering::X, 0, BOTH},
    GateCatalogEntry{"y", Lowering::Y, 0, BOTH},
    GateCatalogEntry{"z", Lowering::Z, 0, BOTH},
    GateCatalogEntry{"h", Lowering::H, 0, BOTH},
    GateCatalogEntry{"s", Lowering::S, 0, BOTH},
    GateCatalogEntry{"sdg", Lowering::Sdg, 0, BOTH},
    GateCatalogEntry{"t", Lowering::T, 0, BOTH},
    GateCatalogEntry{"tdg", Lowering::Tdg, 0, BOTH},
    GateCatalogEntry{"sx", Lowering::SX, 0, STD},
    GateCatalogEntry{"p", Lowering::P, 0, STD},
    GateCatalogEntry{"rx", Lowering::RX, 0, BOTH},
    GateCatalogEntry{"ry", Lowering::RY, 0, BOTH},
    GateCatalogEntry{"rz", Lowering::RZ, 0, BOTH},
    GateCatalogEntry{"r", Lowering::R, 0, COMPAT},
    GateCatalogEntry{"swap", Lowering::SWAP, 0, STD},
    GateCatalogEntry{"cx", Lowering::X, 1, BOTH},
    GateCatalogEntry{"cy", Lowering::Y, 1, BOTH},
    GateCatalogEntry{"cz", Lowering::Z, 1, BOTH},
    GateCatalogEntry{"ch", Lowering::H, 1, BOTH},
    GateCatalogEntry{"cp", Lowering::P, 1, STD},
    GateCatalogEntry{"crx", Lowering::RX, 1, STD},
    GateCatalogEntry{"cry", Lowering::RY, 1, STD},
    GateCatalogEntry{"crz", Lowering::RZ, 1, BOTH},
    GateCatalogEntry{"ccx", Lowering::X, 2, BOTH},
    GateCatalogEntry{"cswap", Lowering::SWAP, 1, STD},
    GateCatalogEntry{"cu", Lowering::CU, 0, STD},
    GateCatalogEntry{"u1", Lowering::P, 0, BOTH},
    GateCatalogEntry{"cu1", Lowering::P, 1, QE_LIB1},
    GateCatalogEntry{"phase", Lowering::P, 0, STD},
    GateCatalogEntry{"cphase", Lowering::P, 1, STD},
    GateCatalogEntry{"u2", Lowering::U2, 0, BOTH},
    GateCatalogEntry{"u3", Lowering::U3, 0, BOTH},
    GateCatalogEntry{"u", Lowering::U3, 0, COMPAT},
    GateCatalogEntry{"cu3", Lowering::U3, 1, QE_LIB1},
    GateCatalogEntry{"CX", Lowering::X, 1, STD},
    GateCatalogEntry{"cnot", Lowering::X, 1, COMPAT},
    GateCatalogEntry{"c3x", Lowering::X, 3, COMPAT},
    GateCatalogEntry{"c4x", Lowering::X, 4, COMPAT},
    GateCatalogEntry{"csx", Lowering::SX, 1, COMPAT},
    GateCatalogEntry{"sxdg", Lowering::SXdg, 0, COMPAT},
    GateCatalogEntry{"c3sqrtx", Lowering::SX, 3, COMPAT},
    GateCatalogEntry{"prx", Lowering::R, 0, COMPAT},
    GateCatalogEntry{"cr", Lowering::R, 1, COMPAT},
    GateCatalogEntry{"fredkin", Lowering::SWAP, 1, COMPAT},
    GateCatalogEntry{"iswap", Lowering::ISWAP, 0, COMPAT},
    GateCatalogEntry{"iswapdg", Lowering::ISWAP, 0, COMPAT, false, true},
    GateCatalogEntry{"dcx", Lowering::DCX, 0, COMPAT},
    GateCatalogEntry{"ecr", Lowering::ECR, 0, COMPAT},
    GateCatalogEntry{"rccx", Lowering::RCCX, 0, COMPAT},
    GateCatalogEntry{"rxx", Lowering::RXX, 0, COMPAT},
    GateCatalogEntry{"ryy", Lowering::RYY, 0, COMPAT},
    GateCatalogEntry{"rzx", Lowering::RZX, 0, COMPAT},
    GateCatalogEntry{"rzz", Lowering::RZZ, 0, COMPAT},
    GateCatalogEntry{"xx_plus_yy", Lowering::XXPlusYY, 0, COMPAT},
    GateCatalogEntry{"xx_minus_yy", Lowering::XXMinusYY, 0, COMPAT},
    GateCatalogEntry{"mcx", Lowering::X, 1, COMPAT, true},
    GateCatalogEntry{"mcx_gray", Lowering::X, 1, COMPAT, true},
    GateCatalogEntry{"mcx_vchain", Lowering::X, 1, COMPAT, true},
    GateCatalogEntry{"mcx_recursive", Lowering::X, 1, COMPAT, true},
    GateCatalogEntry{"mcphase", Lowering::P, 1, COMPAT, true},
};

} // namespace

llvm::ArrayRef<GateCatalogEntry> getGateCatalog() { return CATALOG; }

const GateCatalogEntry* lookupGate(const llvm::StringRef name) {
  static const auto INDEX = [] {
    llvm::StringMap<const GateCatalogEntry*> result;
    for (const auto& gate : CATALOG) {
      result.try_emplace(gate.name, &gate);
    }
    return result;
  }();
  return INDEX.lookup(name);
}

llvm::StringRef canonicalGateName(const GateLowering lowering) {
  switch (lowering) {
  case GateLowering::GPhase:
    return "gphase";
  case GateLowering::Id:
    return "id";
  case GateLowering::X:
    return "x";
  case GateLowering::Y:
    return "y";
  case GateLowering::Z:
    return "z";
  case GateLowering::H:
    return "h";
  case GateLowering::S:
    return "s";
  case GateLowering::Sdg:
    return "sdg";
  case GateLowering::T:
    return "t";
  case GateLowering::Tdg:
    return "tdg";
  case GateLowering::SX:
    return "sx";
  case GateLowering::SXdg:
    return "sxdg";
  case GateLowering::P:
    return "p";
  case GateLowering::RX:
    return "rx";
  case GateLowering::RY:
    return "ry";
  case GateLowering::RZ:
    return "rz";
  case GateLowering::R:
    return "r";
  case GateLowering::U2:
    return "u2";
  case GateLowering::U3:
    return "u3";
  case GateLowering::BuiltinU:
    return "U";
  case GateLowering::CU:
    return "cu";
  case GateLowering::SWAP:
    return "swap";
  case GateLowering::ISWAP:
    return "iswap";
  case GateLowering::DCX:
    return "dcx";
  case GateLowering::ECR:
    return "ecr";
  case GateLowering::RCCX:
    return "rccx";
  case GateLowering::RXX:
    return "rxx";
  case GateLowering::RYY:
    return "ryy";
  case GateLowering::RZX:
    return "rzx";
  case GateLowering::RZZ:
    return "rzz";
  case GateLowering::XXPlusYY:
    return "xx_plus_yy";
  case GateLowering::XXMinusYY:
    return "xx_minus_yy";
  }
  llvm_unreachable("unknown OpenQASM gate lowering");
}

} // namespace mlir::oq3::frontend
