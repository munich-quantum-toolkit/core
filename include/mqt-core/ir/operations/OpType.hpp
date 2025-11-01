/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>

namespace qc {

enum OpTypeFlags : uint8_t {
  OpTypeNone = 0b00,
  OpTypeInv = 0b01,
  OpTypeDiag = 0b10,
};

constexpr static unsigned NUM_OP_TYPE_FLAG_BITS = 2;

// Natively supported operations of the MQT Core library
enum OpType : std::uint8_t {
#define HANDLE_OP_TYPE(N, id, flags, repr)                                     \
  id = ((N) << (NUM_OP_TYPE_FLAG_BITS)) | (flags),
#define LAST_OP_TYPE(N) OpTypeEnd = (N) << (NUM_OP_TYPE_FLAG_BITS),
#include "OpType.inc"

#undef HANDLE_OP_TYPE
#undef LAST_OP_TYPE
};

std::string toString(OpType opType);

/**
 * @brief Gives a short name for the given OpType (at most 3 characters)
 * @param opType OpType to get the short name for
 * @return Short name for the given OpType
 */
std::string shortName(OpType opType);

[[nodiscard]] constexpr bool isTwoQubitGate(const OpType opType) {
  switch (opType) {
  case SWAP:
  case iSWAP:
  case iSWAPdg:
  case Peres:
  case Peresdg:
  case DCX:
  case ECR:
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
  case XXminusYY:
  case XXplusYY:
    return true;
  default:
    return false;
  }
}

/**
 * @brief Checks if given OpType is a single qubit gate
 */
[[nodiscard]] constexpr bool isSingleQubitGate(const OpType type) {
  switch (type) {
  case I:
  case U:
  case U2:
  case P:
  case X:
  case Y:
  case Z:
  case H:
  case S:
  case Sdg:
  case T:
  case SX:
  case SXdg:
  case Tdg:
  case V:
  case Vdg:
  case RX:
  case RY:
  case RZ:
  case R:
    return true;
  default:
    return false;
  }
}

inline std::ostream& operator<<(std::ostream& out, const OpType opType) {
  return out << toString(opType);
}

[[nodiscard]] OpType opTypeFromString(std::string_view opType);

inline std::istream& operator>>(std::istream& in, OpType& opType) {
  std::string opTypeStr;
  in >> opTypeStr;

  if (opTypeStr.empty()) {
    in.setstate(std::istream::failbit);
    return in;
  }

  opType = opTypeFromString(opTypeStr);
  return in;
}

} // namespace qc
