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
#include <memory>

namespace qasm3 {
class Expression;

template <typename T> class Type;
using TypeExpr = Type<std::shared_ptr<Expression>>;
using ResolvedType = Type<uint64_t>;
} // namespace qasm3
