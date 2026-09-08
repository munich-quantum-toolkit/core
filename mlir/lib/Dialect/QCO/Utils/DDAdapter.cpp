/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/DDAdapter.h"

#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>

#include <cstddef>
#include <span>

namespace mlir::qco {

auto makeGateDD(dd::Package& package, std::span<const Complex> matrix,
                size_t /*numQubits*/, llvm::ArrayRef<dd::Qubit> targets,
                const dd::Controls& controls) -> dd::MatrixDD {
  return package.makeGateDD(matrix, {targets.data(), targets.size()}, controls);
}

} // namespace mlir::qco
