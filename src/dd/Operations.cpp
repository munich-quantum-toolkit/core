/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Operations.hpp"

#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"

#include <complex>

namespace dd {

VectorDD applyGlobalPhase(VectorDD& in, const fp& phase, Package& dd) {
  in.w = dd.cn.lookup(in.w * ComplexValue{std::polar(1.0, phase)});
  return in;
}

} // namespace dd
