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

#include <algorithm>
#include <array>
#include <ranges>
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
  case IfElse:
    return "if";
  default:
    return toString(opType);
  }
}

namespace {
// Sorted lexicographically by `name`
constexpr std::array<std::pair<std::string_view, OpType>, 74> OP_NAME_TO_TYPE{{
    {"aod_activate", AodActivate},
    {"aod_deactivate", AodDeactivate},
    {"aod_move", AodMove},
    {"barrier", Barrier},
    {"ch", H},
    {"cnot", X},
    {"compound", Compound},
    {"cp", P},
    {"cphase", P},
    {"cr", R},
    {"crx", RX},
    {"cry", RY},
    {"crz", RZ},
    {"cs", S},
    {"csdg", Sdg},
    {"cswap", SWAP},
    {"csx", SX},
    {"csxdg", SXdg},
    {"ct", T},
    {"ctdg", Tdg},
    {"cu", U},
    {"cu1", P},
    {"cu2", U2},
    {"cu3", U},
    {"cx", X},
    {"cy", Y},
    {"cz", Z},
    {"dcx", DCX},
    {"ecr", ECR},
    {"gphase", GPhase},
    {"h", H},
    {"i", I},
    {"id", I},
    {"if_else", IfElse},
    {"iswap", iSWAP},
    {"iswapdg", iSWAPdg},
    {"mcp", P},
    {"mcphase", P},
    {"mcx", X},
    {"measure", Measure},
    {"move", Move},
    {"none", None},
    {"p", P},
    {"peres", Peres},
    {"peresdg", Peresdg},
    {"phase", P},
    {"prx", R},
    {"r", R},
    {"reset", Reset},
    {"rx", RX},
    {"rxx", RXX},
    {"ry", RY},
    {"ryy", RYY},
    {"rz", RZ},
    {"rzx", RZX},
    {"rzz", RZZ},
    {"s", S},
    {"sdg", Sdg},
    {"swap", SWAP},
    {"sx", SX},
    {"sxdg", SXdg},
    {"t", T},
    {"tdg", Tdg},
    {"u", U},
    {"u1", P},
    {"u2", U2},
    {"u3", U},
    {"v", V},
    {"vdg", Vdg},
    {"x", X},
    {"xx_minus_yy", XXminusYY},
    {"xx_plus_yy", XXplusYY},
    {"y", Y},
    {"z", Z},
}};
static_assert(std::ranges::is_sorted(OP_NAME_TO_TYPE.cbegin(),
                                     OP_NAME_TO_TYPE.cend(),
                                     [](const auto& lhs, const auto& rhs) {
                                       return lhs.first < rhs.first;
                                     }));
} // namespace

OpType opTypeFromString(std::string_view opType) {
  const auto* const it = std::ranges::lower_bound(
      OP_NAME_TO_TYPE, opType, {}, &std::pair<std::string_view, OpType>::first);
  if (it != OP_NAME_TO_TYPE.end() && it->first == opType) {
    return it->second;
  }
  throw std::invalid_argument("Unsupported operation type: " +
                              std::string(opType));
}
} // namespace qc
