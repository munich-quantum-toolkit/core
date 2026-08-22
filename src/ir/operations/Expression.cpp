/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/Expression.hpp"

#include <mutex>
#include <ostream>
#include <string>

namespace sym {

Variable::Variable(const std::string& name) {
  const std::lock_guard lock(registryMutex);
  if (const auto it = registered.find(name); it != registered.end()) {
    id = it->second;
  } else {
    registered[name] = nextId;
    names[nextId] = name;
    id = nextId;
    ++nextId;
  }
}

std::string Variable::getName() const {
  const std::lock_guard lock(registryMutex);
  return names.at(id);
}

std::ostream& operator<<(std::ostream& os, const Variable& var) {
  os << var.getName();
  return os;
}
} // namespace sym
