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
using Lowering = GateLowering;

constexpr std::array CATALOG{
    GateCatalogEntry{"gphase", Lowering::GPhase, 1, 0, 0,
                     Availability::Language},
    GateCatalogEntry{"U", Lowering::BuiltinU, 3, 0, 1, Availability::Language},
    GateCatalogEntry{"id", Lowering::Id, 0, 0, 1, Both},
    GateCatalogEntry{"x", Lowering::X, 0, 0, 1, Both},
    GateCatalogEntry{"y", Lowering::Y, 0, 0, 1, Both},
    GateCatalogEntry{"z", Lowering::Z, 0, 0, 1, Both},
    GateCatalogEntry{"h", Lowering::H, 0, 0, 1, Both},
    GateCatalogEntry{"s", Lowering::S, 0, 0, 1, Both},
    GateCatalogEntry{"sdg", Lowering::Sdg, 0, 0, 1, Both},
    GateCatalogEntry{"t", Lowering::T, 0, 0, 1, Both},
    GateCatalogEntry{"tdg", Lowering::Tdg, 0, 0, 1, Both},
    GateCatalogEntry{"sx", Lowering::SX, 0, 0, 1, Std},
    GateCatalogEntry{"p", Lowering::P, 1, 0, 1, Std},
    GateCatalogEntry{"rx", Lowering::RX, 1, 0, 1, Both},
    GateCatalogEntry{"ry", Lowering::RY, 1, 0, 1, Both},
    GateCatalogEntry{"rz", Lowering::RZ, 1, 0, 1, Both},
    GateCatalogEntry{"r", Lowering::R, 2, 0, 1, Compat},
    GateCatalogEntry{"swap", Lowering::SWAP, 0, 0, 2, Std},
    GateCatalogEntry{"cx", Lowering::X, 0, 1, 1, Both},
    GateCatalogEntry{"cy", Lowering::Y, 0, 1, 1, Both},
    GateCatalogEntry{"cz", Lowering::Z, 0, 1, 1, Both},
    GateCatalogEntry{"ch", Lowering::H, 0, 1, 1, Both},
    GateCatalogEntry{"cp", Lowering::P, 1, 1, 1, Std},
    GateCatalogEntry{"crx", Lowering::RX, 1, 1, 1, Std},
    GateCatalogEntry{"cry", Lowering::RY, 1, 1, 1, Std},
    GateCatalogEntry{"crz", Lowering::RZ, 1, 1, 1, Both},
    GateCatalogEntry{"ccx", Lowering::X, 0, 2, 1, Both},
    GateCatalogEntry{"cswap", Lowering::SWAP, 0, 1, 2, Std},
    GateCatalogEntry{"cu", Lowering::CU, 4, 1, 1, Std},
    GateCatalogEntry{"u1", Lowering::P, 1, 0, 1, Both},
    GateCatalogEntry{"cu1", Lowering::P, 1, 1, 1, QELib1},
    GateCatalogEntry{"phase", Lowering::P, 1, 0, 1, Std},
    GateCatalogEntry{"cphase", Lowering::P, 1, 1, 1, Std},
    GateCatalogEntry{"u2", Lowering::U2, 2, 0, 1, Both},
    GateCatalogEntry{"u3", Lowering::U3, 3, 0, 1, Both},
    GateCatalogEntry{"u", Lowering::U3, 3, 0, 1, Compat},
    GateCatalogEntry{"cu3", Lowering::U3, 3, 1, 1, QELib1},
    GateCatalogEntry{"CX", Lowering::X, 0, 1, 1, Std},
    GateCatalogEntry{"cnot", Lowering::X, 0, 1, 1, Compat},
    GateCatalogEntry{"c3x", Lowering::X, 0, 3, 1, Compat},
    GateCatalogEntry{"c4x", Lowering::X, 0, 4, 1, Compat},
    GateCatalogEntry{"csx", Lowering::SX, 0, 1, 1, Compat},
    GateCatalogEntry{"sxdg", Lowering::SXdg, 0, 0, 1, Compat},
    GateCatalogEntry{"c3sqrtx", Lowering::SXdg, 0, 3, 1, Compat},
    GateCatalogEntry{"prx", Lowering::R, 2, 0, 1, Compat},
    GateCatalogEntry{"cr", Lowering::R, 2, 1, 1, Compat},
    GateCatalogEntry{"fredkin", Lowering::SWAP, 0, 1, 2, Compat},
    GateCatalogEntry{"iswap", Lowering::ISWAP, 0, 0, 2, Compat},
    GateCatalogEntry{"iswapdg", Lowering::ISWAP, 0, 0, 2, Compat, false, true},
    GateCatalogEntry{"dcx", Lowering::DCX, 0, 0, 2, Compat},
    GateCatalogEntry{"ecr", Lowering::ECR, 0, 0, 2, Compat},
    GateCatalogEntry{"rccx", Lowering::RCCX, 0, 0, 3, Compat},
    GateCatalogEntry{"rxx", Lowering::RXX, 1, 0, 2, Compat},
    GateCatalogEntry{"ryy", Lowering::RYY, 1, 0, 2, Compat},
    GateCatalogEntry{"rzx", Lowering::RZX, 1, 0, 2, Compat},
    GateCatalogEntry{"rzz", Lowering::RZZ, 1, 0, 2, Compat},
    GateCatalogEntry{"xx_plus_yy", Lowering::XXPlusYY, 2, 0, 2, Compat},
    GateCatalogEntry{"xx_minus_yy", Lowering::XXMinusYY, 2, 0, 2, Compat},
    GateCatalogEntry{"mcx", Lowering::X, 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcx_gray", Lowering::X, 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcx_vchain", Lowering::X, 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcx_recursive", Lowering::X, 0, 1, 1, Compat, true},
    GateCatalogEntry{"mcphase", Lowering::P, 1, 1, 1, Compat, true},
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
