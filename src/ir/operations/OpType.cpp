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
struct NameToType {
  std::string_view name;
  OpType type;
};

// Sorted lexicographically by `name`
constexpr std::array OP_NAME_TO_TYPE{
    NameToType{.name = "aod_activate", .type = AodActivate},
    NameToType{.name = "aod_deactivate", .type = AodDeactivate},
    NameToType{.name = "aod_move", .type = AodMove},
    NameToType{.name = "barrier", .type = Barrier},
    NameToType{.name = "ch", .type = H},
    NameToType{.name = "cnot", .type = X},
    NameToType{.name = "compound", .type = Compound},
    NameToType{.name = "cp", .type = P},
    NameToType{.name = "cphase", .type = P},
    NameToType{.name = "cr", .type = R},
    NameToType{.name = "crx", .type = RX},
    NameToType{.name = "cry", .type = RY},
    NameToType{.name = "crz", .type = RZ},
    NameToType{.name = "cs", .type = S},
    NameToType{.name = "csdg", .type = Sdg},
    NameToType{.name = "cswap", .type = SWAP},
    NameToType{.name = "csx", .type = SX},
    NameToType{.name = "csxdg", .type = SXdg},
    NameToType{.name = "ct", .type = T},
    NameToType{.name = "ctdg", .type = Tdg},
    NameToType{.name = "cu", .type = U},
    NameToType{.name = "cu1", .type = P},
    NameToType{.name = "cu2", .type = U2},
    NameToType{.name = "cu3", .type = U},
    NameToType{.name = "cx", .type = X},
    NameToType{.name = "cy", .type = Y},
    NameToType{.name = "cz", .type = Z},
    NameToType{.name = "dcx", .type = DCX},
    NameToType{.name = "ecr", .type = ECR},
    NameToType{.name = "gphase", .type = GPhase},
    NameToType{.name = "h", .type = H},
    NameToType{.name = "i", .type = I},
    NameToType{.name = "id", .type = I},
    NameToType{.name = "if_else", .type = IfElse},
    NameToType{.name = "iswap", .type = iSWAP},
    NameToType{.name = "iswapdg", .type = iSWAPdg},
    NameToType{.name = "mcp", .type = P},
    NameToType{.name = "mcphase", .type = P},
    NameToType{.name = "mcx", .type = X},
    NameToType{.name = "measure", .type = Measure},
    NameToType{.name = "move", .type = Move},
    NameToType{.name = "none", .type = None},
    NameToType{.name = "p", .type = P},
    NameToType{.name = "peres", .type = Peres},
    NameToType{.name = "peresdg", .type = Peresdg},
    NameToType{.name = "phase", .type = P},
    NameToType{.name = "prx", .type = R},
    NameToType{.name = "r", .type = R},
    NameToType{.name = "reset", .type = Reset},
    NameToType{.name = "rx", .type = RX},
    NameToType{.name = "rxx", .type = RXX},
    NameToType{.name = "ry", .type = RY},
    NameToType{.name = "ryy", .type = RYY},
    NameToType{.name = "rz", .type = RZ},
    NameToType{.name = "rzx", .type = RZX},
    NameToType{.name = "rzz", .type = RZZ},
    NameToType{.name = "s", .type = S},
    NameToType{.name = "sdg", .type = Sdg},
    NameToType{.name = "swap", .type = SWAP},
    NameToType{.name = "sx", .type = SX},
    NameToType{.name = "sxdg", .type = SXdg},
    NameToType{.name = "t", .type = T},
    NameToType{.name = "tdg", .type = Tdg},
    NameToType{.name = "u", .type = U},
    NameToType{.name = "u1", .type = P},
    NameToType{.name = "u2", .type = U2},
    NameToType{.name = "u3", .type = U},
    NameToType{.name = "v", .type = V},
    NameToType{.name = "vdg", .type = Vdg},
    NameToType{.name = "x", .type = X},
    NameToType{.name = "xx_minus_yy", .type = XXminusYY},
    NameToType{.name = "xx_plus_yy", .type = XXplusYY},
    NameToType{.name = "y", .type = Y},
    NameToType{.name = "z", .type = Z},
};
static_assert(std::ranges::is_sorted(OP_NAME_TO_TYPE.cbegin(),
                                     OP_NAME_TO_TYPE.cend(),
                                     [](const auto& lhs, const auto& rhs) {
                                       return lhs.name < rhs.name;
                                     }));
} // namespace

OpType opTypeFromString(const std::string_view opType) {
  // clang-tidy produces a false-positive that produces a Windows compile error
  // when accepted. NOLINTNEXTLINE(*-qualified-auto)
  const auto it =
      std::ranges::lower_bound(OP_NAME_TO_TYPE, opType, {}, &NameToType::name);
  if (it != OP_NAME_TO_TYPE.end() && it->name == opType) {
    return it->type;
  }
  throw std::invalid_argument("Unsupported operation type: " +
                              std::string(opType));
}
} // namespace qc
