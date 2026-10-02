/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Target/OpenQASM/GateCatalog.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <array>

namespace mlir::openqasm::frontend {
namespace {

using Availability = GateAvailability;
constexpr auto STD = Availability::StandardLibrary;
constexpr auto QE_LIB1 = Availability::QELib1;
constexpr auto BOTH = Availability::StandardLibraryAndQELib1;
constexpr auto COMPAT = Availability::Compatibility;

const std::array CATALOG{
    GateCatalogEntry{"gphase", qc::StandardGate::GPhase, 0,
                     Availability::Language},
    GateCatalogEntry{"U", qc::StandardGate::BuiltinU, 0,
                     Availability::Language},
    GateCatalogEntry{"id", qc::StandardGate::Id, 0, BOTH},
    GateCatalogEntry{"x", qc::StandardGate::X, 0, BOTH},
    GateCatalogEntry{"y", qc::StandardGate::Y, 0, BOTH},
    GateCatalogEntry{"z", qc::StandardGate::Z, 0, BOTH},
    GateCatalogEntry{"h", qc::StandardGate::H, 0, BOTH},
    GateCatalogEntry{"s", qc::StandardGate::S, 0, BOTH},
    GateCatalogEntry{"sdg", qc::StandardGate::Sdg, 0, BOTH},
    GateCatalogEntry{"t", qc::StandardGate::T, 0, BOTH},
    GateCatalogEntry{"tdg", qc::StandardGate::Tdg, 0, BOTH},
    GateCatalogEntry{"sx", qc::StandardGate::SX, 0, STD},
    GateCatalogEntry{"p", qc::StandardGate::P, 0, STD},
    GateCatalogEntry{"rx", qc::StandardGate::RX, 0, BOTH},
    GateCatalogEntry{"ry", qc::StandardGate::RY, 0, BOTH},
    GateCatalogEntry{"rz", qc::StandardGate::RZ, 0, BOTH},
    GateCatalogEntry{"r", qc::StandardGate::R, 0, COMPAT},
    GateCatalogEntry{"swap", qc::StandardGate::SWAP, 0, STD},
    GateCatalogEntry{"cx", qc::StandardGate::X, 1, BOTH},
    GateCatalogEntry{"cy", qc::StandardGate::Y, 1, BOTH},
    GateCatalogEntry{"cz", qc::StandardGate::Z, 1, BOTH},
    GateCatalogEntry{"ch", qc::StandardGate::H, 1, BOTH},
    GateCatalogEntry{"cp", qc::StandardGate::P, 1, STD},
    GateCatalogEntry{"crx", qc::StandardGate::RX, 1, STD},
    GateCatalogEntry{"cry", qc::StandardGate::RY, 1, STD},
    GateCatalogEntry{"crz", qc::StandardGate::RZ, 1, BOTH},
    GateCatalogEntry{"ccx", qc::StandardGate::X, 2, BOTH},
    GateCatalogEntry{"cswap", qc::StandardGate::SWAP, 1, STD},
    GateCatalogEntry{"cu", qc::StandardGate::CU, 0, STD},
    GateCatalogEntry{"u1", qc::StandardGate::P, 0, BOTH},
    GateCatalogEntry{"cu1", qc::StandardGate::P, 1, QE_LIB1},
    GateCatalogEntry{"phase", qc::StandardGate::P, 0, STD},
    GateCatalogEntry{"cphase", qc::StandardGate::P, 1, STD},
    GateCatalogEntry{"u2", qc::StandardGate::U2, 0, BOTH},
    GateCatalogEntry{"u3", qc::StandardGate::U3, 0, BOTH},
    GateCatalogEntry{"u", qc::StandardGate::U3, 0, COMPAT},
    GateCatalogEntry{"cu3", qc::StandardGate::U3, 1, QE_LIB1},
    GateCatalogEntry{"CX", qc::StandardGate::X, 1, STD},
    GateCatalogEntry{"cnot", qc::StandardGate::X, 1, COMPAT},
    GateCatalogEntry{"c3x", qc::StandardGate::X, 3, COMPAT},
    GateCatalogEntry{"c4x", qc::StandardGate::X, 4, COMPAT},
    GateCatalogEntry{"csx", qc::StandardGate::SX, 1, COMPAT},
    GateCatalogEntry{"sxdg", qc::StandardGate::SXdg, 0, COMPAT},
    GateCatalogEntry{"c3sqrtx", qc::StandardGate::SX, 3, COMPAT},
    GateCatalogEntry{"prx", qc::StandardGate::R, 0, COMPAT},
    GateCatalogEntry{"cr", qc::StandardGate::R, 1, COMPAT},
    GateCatalogEntry{"fredkin", qc::StandardGate::SWAP, 1, COMPAT},
    GateCatalogEntry{"iswap", qc::StandardGate::ISWAP, 0, COMPAT},
    GateCatalogEntry{"iswapdg", qc::StandardGate::ISWAP, 0, COMPAT, false,
                     true},
    GateCatalogEntry{"dcx", qc::StandardGate::DCX, 0, COMPAT},
    GateCatalogEntry{"ecr", qc::StandardGate::ECR, 0, COMPAT},
    GateCatalogEntry{"rccx", qc::StandardGate::RCCX, 0, COMPAT},
    GateCatalogEntry{"rxx", qc::StandardGate::RXX, 0, COMPAT},
    GateCatalogEntry{"ryy", qc::StandardGate::RYY, 0, COMPAT},
    GateCatalogEntry{"rzx", qc::StandardGate::RZX, 0, COMPAT},
    GateCatalogEntry{"rzz", qc::StandardGate::RZZ, 0, COMPAT},
    GateCatalogEntry{"xx_plus_yy", qc::StandardGate::XXPlusYY, 0, COMPAT},
    GateCatalogEntry{"xx_minus_yy", qc::StandardGate::XXMinusYY, 0, COMPAT},
    GateCatalogEntry{"mcx", qc::StandardGate::X, 1, COMPAT, true},
    GateCatalogEntry{"mcx_gray", qc::StandardGate::X, 1, COMPAT, true},
    GateCatalogEntry{"mcx_vchain", qc::StandardGate::X, 1, COMPAT, true},
    GateCatalogEntry{"mcx_recursive", qc::StandardGate::X, 1, COMPAT, true},
    GateCatalogEntry{"mcphase", qc::StandardGate::P, 1, COMPAT, true},
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

llvm::StringRef canonicalGateName(const qc::StandardGate gate) {
  return gate == qc::StandardGate::U3
             ? "u3"
             : qc::getStandardGateDescriptor(gate).operationSymbol;
}

} // namespace mlir::openqasm::frontend
