/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Support/PrettyPrinting.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>

#include <cmath>
#include <complex> // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace mqt::test {

/**
 * Check whether two unitary matrices are equal up to a single unit-modulus
 * global phase factor.
 *
 * The comparison is symmetric and numerically stable in the sense that a near
 * zero overlap (``|trace(rhs^H * lhs)| <= atol``) is treated as "not
 * equivalent" to avoid division by a tiny number.
 */
template <typename Matrix>
[[nodiscard]] bool isEquivalentUpToGlobalPhase(const Matrix& lhs,
                                               const Matrix& rhs,
                                               double atol = 1e-10) {
  const auto overlap = (rhs.adjoint() * lhs).trace();
  if (std::abs(overlap) <= atol) {
    return false;
  }
  const auto factor = overlap / std::abs(overlap);
  return lhs.isApprox(factor * rhs, atol);
}

template <typename BuilderT> struct NamedBuilder {
  const char* name = nullptr;
  void (*fn)(BuilderT&) = nullptr;

  constexpr NamedBuilder(const char* nameIn, void (*fnIn)(BuilderT&)) noexcept
      : name(nameIn), fn(fnIn) {}

  // NOLINTNEXTLINE(*-explicit-constructor)
  constexpr NamedBuilder(std::nullptr_t) noexcept : fn(nullptr) {}

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return fn != nullptr;
  }
};

template <typename BuilderT>
[[nodiscard]] constexpr NamedBuilder<BuilderT>
namedBuilder(const char* name, void (*fn)(BuilderT&)) noexcept {
  return NamedBuilder<BuilderT>{name, fn};
}

[[nodiscard]] constexpr const char* displayName(const char* name) noexcept {
  return name != nullptr ? name : "<null>";
}

/**
 * @brief Returns true when IR printing is always enabled via the environment.
 *
 * Set the environment variable @c MQT_MLIR_TEST_PRINT_IR to any non-empty
 * value to unconditionally print IR in every test.
 */
[[nodiscard]] inline bool irPrintingForced() noexcept {
  const char* const env = std::getenv("MQT_MLIR_TEST_PRINT_IR");
  return env != nullptr && *env != '\0';
}

/**
 * @brief RAII helper that defers IR printing until the end of a test.
 *
 * @details
 * Each call to @c record() eagerly renders the given @c ModuleOp to a string
 * and stores it alongside its header. When the @c DeferredPrinter is destroyed
 * (i.e., at the end of the test body) it either:
 *   - flushes all captured IR to @c llvm::errs() when the test has already
 *     recorded a failure (@c ::testing::Test::HasFailure()), or
 *   - flushes unconditionally when the @c MQT_MLIR_TEST_PRINT_IR environment
 *     variable is set to a non-empty value.
 *
 * In all other cases the captured strings are simply discarded, avoiding the
 * significant I/O overhead of box-printing on every passing test.
 *
 * Usage:
 * @code
 *   TEST_P(MyTest, MyCase) {
 *     DeferredPrinter printer;
 *     auto module = build(...);
 *     printer.record(module.get(), "My IR");
 *     EXPECT_TRUE(someCheck(module.get()));
 *   }
 * @endcode
 */
class DeferredPrinter {
public:
  DeferredPrinter() = default;

  // Non-copyable, non-movable — lives exactly as long as the test scope.
  DeferredPrinter(const DeferredPrinter&) = delete;
  DeferredPrinter& operator=(const DeferredPrinter&) = delete;
  DeferredPrinter(DeferredPrinter&&) = delete;
  DeferredPrinter& operator=(DeferredPrinter&&) = delete;

  /**
   * @brief Capture the current state of @p module under the given @p header.
   *
   * The IR is rendered to a string immediately (so later mutations to the
   * module do not affect already-captured snapshots). The output is deferred
   * until the printer is destroyed.
   */
  void record(mlir::ModuleOp module, llvm::StringRef header) {
    llvm::SmallString<4096> irString;
    llvm::raw_svector_ostream irStream(irString);
    module.print(irStream);
    entries_.emplace_back(header.str(), irString.str());
  }

  /**
   * @brief Flush all captured entries to @c llvm::errs() if needed.
   *
   * Triggered by a test failure or the @c MQT_MLIR_TEST_PRINT_IR env var.
   */
  ~DeferredPrinter() {
    if (!irPrintingForced() && !::testing::Test::HasFailure()) {
      return;
    }
    for (const auto& [header, ir] : entries_) {
      mlir::printBoxTop();
      mlir::printBoxLine(header);
      mlir::printBoxMiddle();
      mlir::printBoxText(ir);
      mlir::printBoxBottom();
      llvm::errs().flush();
    }
  }

private:
  std::vector<std::pair<std::string, std::string>> entries_;
};

// ---------------------------------------------------------------------------
// Gate-matrix factories and two-qubit layout helpers shared by the
// decomposition / native-synthesis unit tests. They build reference unitaries
// for equivalence checks and are intentionally test-only, so they live here
// rather than in the production decomposition headers.
// ---------------------------------------------------------------------------

/// Hadamard gate (2x2).
[[nodiscard]] inline const mlir::qco::Matrix2x2& hGate() {
  constexpr double frac1Sqrt2 =
      0.707106781186547524400844362104849039284835937688474036588L;
  static const mlir::qco::Matrix2x2 matrix = mlir::qco::Matrix2x2::fromElements(
      frac1Sqrt2, frac1Sqrt2, frac1Sqrt2, -frac1Sqrt2);
  return matrix;
}

/// Logical qubit index within a two-qubit reference matrix.
using QubitId = std::size_t;

/// `SWAP` gate in MQT operand order (qubit 0 = MSB).
[[nodiscard]] inline const mlir::qco::Matrix4x4& swapGate() {
  static const mlir::qco::Matrix4x4 matrix =
      mlir::qco::Matrix4x4::fromElements(1, 0, 0, 0, //
                                         0, 0, 1, 0, //
                                         0, 1, 0, 0, //
                                         0, 0, 0, 1);
  return matrix;
}

/// Embed a single-qubit matrix into the two-qubit space acting on `qubitId`.
[[nodiscard]] inline mlir::qco::Matrix4x4
expandToTwoQubits(const mlir::qco::Matrix2x2& singleQubitMatrix,
                  QubitId qubitId) {
  if (qubitId == 0) {
    return kron(singleQubitMatrix, mlir::qco::Matrix2x2::identity());
  }
  if (qubitId == 1) {
    return kron(mlir::qco::Matrix2x2::identity(), singleQubitMatrix);
  }
  llvm::reportFatalInternalError("Invalid qubit id for single-qubit expansion");
}

/// Reorder a two-qubit matrix so it acts on qubits `{0, 1}` in MQT order.
[[nodiscard]] inline mlir::qco::Matrix4x4
fixTwoQubitMatrixQubitOrder(const mlir::qco::Matrix4x4& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds) {
  if (qubitIds == llvm::SmallVector<QubitId, 2>{1, 0}) {
    return swapGate() * twoQubitMatrix * swapGate();
  }
  if (qubitIds == llvm::SmallVector<QubitId, 2>{0, 1}) {
    return twoQubitMatrix;
  }
  llvm::reportFatalInternalError(
      "Invalid qubit IDs for fixing two-qubit matrix");
}

} // namespace mqt::test

#define MQT_NAMED_BUILDER(fn) ::mqt::test::namedBuilder(#fn, fn)
