/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/OpType.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qc {
std::string toString(const OpType opType) {
  static const std::unordered_map<OpType, std::string_view> OP_NAMES{
#define HANDLE_OP_TYPE(N, id, flags, repr) {id, {repr}},
#define LAST_OP_TYPE(N)
#include "ir/operations/OpType.inc"

#undef HANDLE_OP_TYPE
#undef LAST_OP_TYPE
  };

  if (const auto it = OP_NAMES.find(opType); it != OP_NAMES.end()) {
    return std::string(it->second);
  }
  throw std::invalid_argument("Invalid OpType!");
}

std::string shortName(const OpType opType) {
  switch (opType) {
  case GPhase:
    return "GPh";
  case SXdg:
    return "sxd";
  case SWAP:
    return "sw";
  case iSWAP:
    return "isw";
  case iSWAPdg:
    return "isd";
  case Peres:
    return "pr";
  case Peresdg:
    return "prd";
  case XXminusYY:
    return "x-y";
  case XXplusYY:
    return "x+y";
  case Barrier:
    return "====";
  case Measure:
    return "msr";
  case Reset:
    return "rst";
  default:
    return toString(opType);
  }
}
OpType opTypeFromString(const std::string& opType) {
  static const std::unordered_map<std::string, OpType> OP_NAME_TO_TYPE = {
      {"none", OpType::None},
      {"gphase", OpType::GPhase},
      {"i", OpType::I},
      {"id", OpType::I},
      {"h", OpType::H},
      {"ch", OpType::H},
      {"x", OpType::X},
      {"cnot", OpType::X},
      {"cx", OpType::X},
      {"mcx", OpType::X},
      {"y", OpType::Y},
      {"cy", OpType::Y},
      {"z", OpType::Z},
      {"cz", OpType::Z},
      {"s", OpType::S},
      {"cs", OpType::S},
      {"sdg", OpType::Sdg},
      {"csdg", OpType::Sdg},
      {"t", OpType::T},
      {"ct", OpType::T},
      {"tdg", OpType::Tdg},
      {"ctdg", OpType::Tdg},
      {"v", OpType::V},
      {"vdg", OpType::Vdg},
      {"u", OpType::U},
      {"cu", OpType::U},
      {"u3", OpType::U},
      {"cu3", OpType::U},
      {"u2", OpType::U2},
      {"cu2", OpType::U2},
      {"p", OpType::P},
      {"cp", OpType::P},
      {"mcp", OpType::P},
      {"phase", OpType::P},
      {"cphase", OpType::P},
      {"mcphase", OpType::P},
      {"u1", OpType::P},
      {"cu1", OpType::P},
      {"sx", OpType::SX},
      {"csx", OpType::SX},
      {"sxdg", OpType::SXdg},
      {"csxdg", OpType::SXdg},
      {"rx", OpType::RX},
      {"crx", OpType::RX},
      {"ry", OpType::RY},
      {"cry", OpType::RY},
      {"rz", OpType::RZ},
      {"crz", OpType::RZ},
      {"swap", OpType::SWAP},
      {"cswap", OpType::SWAP},
      {"iswap", OpType::iSWAP},
      {"iswapdg", OpType::iSWAPdg},
      {"peres", OpType::Peres},
      {"peresdg", OpType::Peresdg},
      {"dcx", OpType::DCX},
      {"ecr", OpType::ECR},
      {"rxx", OpType::RXX},
      {"ryy", OpType::RYY},
      {"rzz", OpType::RZZ},
      {"rzx", OpType::RZX},
      {"xx_minus_yy", OpType::XXminusYY},
      {"xx_plus_yy", OpType::XXplusYY},
      {"measure", OpType::Measure},
      {"reset", OpType::Reset},
      {"barrier", OpType::Barrier},
      {"classic_controlled", OpType::ClassicControlled},
      {"compound", OpType::Compound},
      {"move", OpType::Move},
      {"aod_activate", OpType::AodActivate},
      {"aod_deactivate", OpType::AodDeactivate},
      {"aod_move", OpType::AodMove},
  };

  // try to find the operation type in the map of known operation types and
  // return it if found or throw an exception otherwise.
  if (const auto it = OP_NAME_TO_TYPE.find(opType);
      it != OP_NAME_TO_TYPE.end()) {
    return OP_NAME_TO_TYPE.at(opType);
  }
  throw std::invalid_argument("Unsupported operation type: " + opType);
}
} // namespace qc
