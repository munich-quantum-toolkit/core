/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Expression.hpp"

#include <ostream>
#include <string>

namespace sym {

Variable::Variable(const std::string& name) {
  if (const auto it = registered.find(name); it != registered.end()) {
    id = it->second;
  } else {
    registered[name] = nextId;
    names[nextId] = name;
    id = nextId;
    ++nextId;
  }
}

std::string Variable::getName() const noexcept { return names[id]; }

std::ostream& operator<<(std::ostream& os, const Variable& var) {
  os << var.getName();
  return os;
}
} // namespace sym
