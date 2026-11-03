/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Pauli.hpp
/// Sparse real sums of Pauli strings.

#pragma once

#include "dd/DDDefinitions.hpp"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace dd {

/// One real coefficient multiplying a Pauli string.
///
/// Each character is I, X, Y, or Z. The rightmost character acts on qubit zero.
/// The label contains one character per qubit, including identities.
struct PauliTerm {
  std::string paulis;
  fp coefficient = 1.;
};

/// A sparse sum of Pauli terms; an empty sum is the zero observable.
using PauliObservable = llvm::SmallVector<PauliTerm, 1>;

} // namespace dd
