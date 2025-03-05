/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/operations/LoadOp.hpp"

#include <cstddef>
#include <sstream>
#include <string>

namespace na {
auto LoadOp::toString() const -> std::string {
  std::stringstream ss;
  ss << "@+ load";
  if (atoms.size() == 1) {
    if (targetLocations_) {
      ss << " " << targetLocations_->front();
    }
    ss << " " << *(atoms.front());
  } else {
    ss << " [\n";
    for (std::size_t i = 0; i < atoms.size(); ++i) {
      ss << "    ";
      if (targetLocations_) {
        ss << (*targetLocations_)[i] << " ";
      }
      ss << *(atoms[i]) << "\n";
    }
    ss << "]";
  }
  return ss.str();
}
} // namespace na
