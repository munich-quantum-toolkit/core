/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "qc_programs.h"

namespace mlir::qco::native_synth {
bool allowsSingleQubitOp(UnitaryOpInterface op,
                         const decomposition::NativeProfileSpec& spec);
bool getBlockTwoQubitMatrix(Operation* op, Matrix4x4& matrix);
} // namespace mlir::qco::native_synth

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/QC/Builder/QCProgramBuilder.h>
#include <mlir/Dialect/QC/IR/QCDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::decomposition;
using namespace mlir::qco::native_synth;

namespace mlir::qco::native_synth_test {

using mqt::test::isEquivalentUpToGlobalPhase;

/// Minimal dense, row-major, square complex matrix with runtime dimension.
///
/// Used by the multi-qubit equivalence checks (the synthesized circuits may
/// span more than two wires, so the fixed-size `Matrix2x2`/`Matrix4x4` are not
/// enough). Provides exactly the surface
/// `mqt::test::isEquivalentUpToGlobalPhase` needs: `adjoint()`, `operator*`,
/// scalar multiply, `trace()`, and `isApprox()`.
class TestMatrix {
public:
  TestMatrix() = default;
  explicit TestMatrix(std::size_t dim)
      : dim_(dim), data_(dim * dim, std::complex<double>{0.0, 0.0}) {}

  /// Identity matrix of dimension @p dim.
  [[nodiscard]] static TestMatrix identity(std::size_t dim);
  /// Promote a fixed `2×2` matrix to a `TestMatrix`.
  [[nodiscard]] static TestMatrix fromMatrix2x2(const Matrix2x2& matrix);
  /// Promote a fixed `4×4` matrix to a `TestMatrix`.
  [[nodiscard]] static TestMatrix fromMatrix4x4(const Matrix4x4& matrix);

  [[nodiscard]] std::size_t dim() const { return dim_; }

  [[nodiscard]] std::complex<double>& operator()(std::size_t row,
                                                 std::size_t col) {
    return data_[(row * dim_) + col];
  }
  [[nodiscard]] std::complex<double> operator()(std::size_t row,
                                                std::size_t col) const {
    return data_[(row * dim_) + col];
  }

  /// Matrix product (dimensions must match).
  [[nodiscard]] TestMatrix operator*(const TestMatrix& rhs) const;
  /// Element-wise scaling by a complex scalar.
  [[nodiscard]] TestMatrix operator*(std::complex<double> scalar) const;
  /// Conjugate transpose.
  [[nodiscard]] TestMatrix adjoint() const;
  /// Sum of diagonal entries.
  [[nodiscard]] std::complex<double> trace() const;
  /// Entry-wise approximate equality (false on dimension mismatch).
  [[nodiscard]] bool isApprox(const TestMatrix& other,
                              double tol = 1e-10) const;

private:
  std::size_t dim_ = 0;
  std::vector<std::complex<double>> data_;
};

/// Left scalar multiply, mirroring the right multiply above.
[[nodiscard]] inline TestMatrix operator*(std::complex<double> scalar,
                                          const TestMatrix& matrix) {
  return matrix * scalar;
}

bool extractSingleQubitMatrix(qco::UnitaryOpInterface op, Matrix2x2& out);
bool extractTwoQubitMatrix(qco::UnitaryOpInterface op, Matrix4x4& out);
[[nodiscard]] std::optional<Matrix4x4>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp);
[[nodiscard]] TestMatrix expandOneQToN(const Matrix2x2& matrix, std::size_t q,
                                       std::size_t numQubits);
[[nodiscard]] TestMatrix expandTwoQToN(const Matrix4x4& matrix, std::size_t q0,
                                       std::size_t q1, std::size_t numQubits);
[[nodiscard]] std::optional<TestMatrix>
computeNQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp,
                               std::size_t maxQubits = 6);

/// One row of the standard multi-profile equivalence sweeps in tests.
struct NativeSynthesisProfileSweepCase {
  const char* nativeGates;
  bool (*isNative)(mlir::OwningOpRef<mlir::ModuleOp>&);
};

/// Shared gtest fixture for native-gate synthesis pass tests.
class NativeSynthesisPassTest : public testing::Test {
protected:
  void SetUp() override {
    mlir::DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
    context = std::make_unique<mlir::MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  template <typename... Allowed1QOps>
  static bool onlyTheseOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp,
                           const bool allowCx, const bool allowCz) {
    bool ok = true;
    std::ignore = moduleOp->walk([&](mlir::qco::UnitaryOpInterface op) {
      mlir::Operation* raw = op.getOperation();
      if (llvm::isa_and_present<mlir::qco::CtrlOp>(raw->getParentOp())) {
        return mlir::WalkResult::advance();
      }
      if (llvm::isa<mlir::qco::BarrierOp, mlir::qco::GPhaseOp>(raw)) {
        return mlir::WalkResult::advance();
      }
      if (auto ctrl = llvm::dyn_cast<mlir::qco::CtrlOp>(raw)) {
        if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
          ok = false;
          return mlir::WalkResult::interrupt();
        }
        mlir::Operation* body = ctrl.getBodyUnitary(0).getOperation();
        const bool isCx = llvm::isa<mlir::qco::XOp>(body);
        const bool isCz = llvm::isa<mlir::qco::ZOp>(body);
        if ((isCx && allowCx) || (isCz && allowCz)) {
          return mlir::WalkResult::advance();
        }
        ok = false;
        return mlir::WalkResult::interrupt();
      }

      if (!llvm::isa<Allowed1QOps...>(raw)) {
        ok = false;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return ok;
  }

  static bool onlyIbmBasicCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::XOp, mlir::qco::SXOp, mlir::qco::RZOp,
                        mlir::qco::POp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/false);
  }

  static bool onlyIbmBasicCzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::XOp, mlir::qco::SXOp, mlir::qco::RZOp,
                        mlir::qco::POp>(moduleOp, /*allowCx=*/false,
                                        /*allowCz=*/true);
  }

  static bool onlyGenericU3CxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/false);
  }

  static bool onlyGenericU3CzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp>(moduleOp, /*allowCx=*/false,
                                        /*allowCz=*/true);
  }

  static bool onlyIqmDefaultOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::ROp>(moduleOp, /*allowCx=*/false,
                                        /*allowCz=*/true);
  }

  static bool
  onlyIbmFractionalOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::XOp, mlir::qco::SXOp, mlir::qco::RZOp,
                        mlir::qco::POp, mlir::qco::RXOp, mlir::qco::RZZOp>(
        moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool
  onlyAxisPairRxRzCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::RXOp, mlir::qco::RZOp, mlir::qco::POp>(
        moduleOp, /*allowCx=*/true, /*allowCz=*/false);
  }

  static bool
  onlyAxisPairRxRyCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::RXOp, mlir::qco::RYOp>(
        moduleOp, /*allowCx=*/true, /*allowCz=*/false);
  }

  static bool
  onlyAxisPairRyRzCzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::RYOp, mlir::qco::RZOp, mlir::qco::POp>(
        moduleOp, /*allowCx=*/false, /*allowCz=*/true);
  }

  static bool
  onlyUOrAxisPairRxRzCxOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp, mlir::qco::RXOp, mlir::qco::RZOp,
                        mlir::qco::POp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/false);
  }

  static bool
  onlyGenericU3CxOrCzOps(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    return onlyTheseOps<mlir::qco::UOp>(moduleOp, /*allowCx=*/true,
                                        /*allowCz=*/true);
  }

  static std::array<NativeSynthesisProfileSweepCase, 3>
  coreEquivalenceProfiles() {
    return {{{.nativeGates = "x,sx,rz,cx", .isNative = &onlyIbmBasicCxOps},
             {.nativeGates = "u,cx", .isNative = &onlyGenericU3CxOps},
             {.nativeGates = "r,cz", .isNative = &onlyIqmDefaultOps}}};
  }

  static void runNativeSynthesis(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp,
                                 const std::string& nativeGates) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    pm.addPass(mlir::qco::createFuseTwoQubitUnitaryRuns(
        mlir::qco::FuseTwoQubitUnitaryRunsOptions{
            .nativeGates = nativeGates,
        }));
    ASSERT_TRUE(mlir::succeeded(pm.run(*moduleOp)));
  }

  static void runQcToQco(mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    ASSERT_TRUE(mlir::succeeded(pm.run(*moduleOp)));
  }

  static std::string
  moduleToString(const mlir::OwningOpRef<mlir::ModuleOp>& moduleOp) {
    std::string text;
    llvm::raw_string_ostream os(text);
    moduleOp.get()->print(os);
    return text;
  }

  template <typename BuildFn, typename PredicateFn>
  void expectNativeAfterSynthesis(BuildFn buildFn,
                                  const std::string& nativeGates,
                                  PredicateFn isNative) {
    auto moduleOp = buildFn();
    runNativeSynthesis(moduleOp, nativeGates);
    EXPECT_TRUE(isNative(moduleOp));
  }

  template <typename BuildFn>
  void expectSynthesisFailure(BuildFn buildFn, const std::string& nativeGates) {
    auto moduleOp = buildFn();
    mlir::PassManager pm(moduleOp->getContext());
    pm.addPass(mlir::createQCToQCO());
    pm.addPass(mlir::qco::createFuseTwoQubitUnitaryRuns(
        mlir::qco::FuseTwoQubitUnitaryRunsOptions{
            .nativeGates = nativeGates,
        }));
    EXPECT_TRUE(mlir::failed(pm.run(*moduleOp)));
  }

  template <typename BuildFn, typename PredicateFn, typename UnitaryFn>
  void expectEquivalentAndNativeAfterSynthesis(BuildFn buildFn,
                                               const std::string& nativeGates,
                                               PredicateFn isNative,
                                               UnitaryFn computeUnitary) {
    auto expectedModule = buildFn();
    runQcToQco(expectedModule);
    const auto expectedUnitary = computeUnitary(expectedModule);
    ASSERT_TRUE(expectedUnitary.has_value());

    auto synthesizedModule = buildFn();
    runNativeSynthesis(synthesizedModule, nativeGates);
    EXPECT_TRUE(isNative(synthesizedModule));
    const auto synthesizedUnitary = computeUnitary(synthesizedModule);
    ASSERT_TRUE(synthesizedUnitary.has_value());
    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary));
  }

  std::unique_ptr<mlir::MLIRContext> context;
};

TestMatrix TestMatrix::identity(std::size_t dim) {
  TestMatrix result(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    result(i, i) = std::complex<double>{1.0, 0.0};
  }
  return result;
}

TestMatrix TestMatrix::fromMatrix2x2(const Matrix2x2& matrix) {
  TestMatrix result(2);
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t col = 0; col < 2; ++col) {
      result(row, col) = matrix(row, col);
    }
  }
  return result;
}

TestMatrix TestMatrix::fromMatrix4x4(const Matrix4x4& matrix) {
  TestMatrix result(4);
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      result(row, col) = matrix(row, col);
    }
  }
  return result;
}

TestMatrix TestMatrix::operator*(const TestMatrix& rhs) const {
  TestMatrix result(dim_);
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t k = 0; k < dim_; ++k) {
      const std::complex<double> a = (*this)(row, k);
      if (a == std::complex<double>{0.0, 0.0}) {
        continue;
      }
      for (std::size_t col = 0; col < dim_; ++col) {
        result(row, col) += a * rhs(k, col);
      }
    }
  }
  return result;
}

TestMatrix TestMatrix::operator*(std::complex<double> scalar) const {
  TestMatrix result(dim_);
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t col = 0; col < dim_; ++col) {
      result(row, col) = (*this)(row, col) * scalar;
    }
  }
  return result;
}

TestMatrix TestMatrix::adjoint() const {
  TestMatrix result(dim_);
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t col = 0; col < dim_; ++col) {
      result(col, row) = std::conj((*this)(row, col));
    }
  }
  return result;
}

std::complex<double> TestMatrix::trace() const {
  std::complex<double> sum{0.0, 0.0};
  for (std::size_t i = 0; i < dim_; ++i) {
    sum += (*this)(i, i);
  }
  return sum;
}

bool TestMatrix::isApprox(const TestMatrix& other, double tol) const {
  if (dim_ != other.dim_) {
    return false;
  }
  for (std::size_t row = 0; row < dim_; ++row) {
    for (std::size_t col = 0; col < dim_; ++col) {
      if (std::abs((*this)(row, col) - other(row, col)) > tol) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] static std::optional<Value>
getUnitaryQubitOperand(qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getOperand(index);
  if (!llvm::isa<qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

[[nodiscard]] static std::optional<Value>
getUnitaryQubitResult(qco::UnitaryOpInterface op, std::size_t index) {
  if (index >= op.getNumQubits()) {
    return std::nullopt;
  }
  Value v = op->getResult(index);
  if (!llvm::isa<qco::QubitType>(v.getType())) {
    return std::nullopt;
  }
  return v;
}

/// Extract the 2x2 unitary matrix associated with a single-qubit op.
bool extractSingleQubitMatrix(qco::UnitaryOpInterface op, Matrix2x2& out) {
  if (op.getUnitaryMatrix2x2(out)) {
    return true;
  }
  qco::DynamicMatrix dynamic;
  if (!op.getUnitaryMatrixDynamic(dynamic) || dynamic.rows() != 2 ||
      dynamic.cols() != 2) {
    return false;
  }
  out = Matrix2x2::fromElements(dynamic(0, 0), dynamic(0, 1), dynamic(1, 0),
                                dynamic(1, 1));
  return true;
}

/// 4×4 unitary for a two-qubit op (same layout as ``getUnitaryMatrix4x4``).
bool extractTwoQubitMatrix(qco::UnitaryOpInterface op, Matrix4x4& out) {
  if (native_synth::getBlockTwoQubitMatrix(op.getOperation(), out)) {
    return true;
  }
  return op.getUnitaryMatrix4x4(out);
}

using mqt::test::expandToTwoQubits;
using mqt::test::fixTwoQubitMatrixQubitOrder;
using mqt::test::QubitId;

std::optional<Matrix4x4>
computeTwoQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }
  Matrix4x4 unitary = Matrix4x4::identity();
  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t nextQubitId = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = llvm::dyn_cast<qco::AllocOp>(&rawOp)) {
          if (nextQubitId >= 2) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), nextQubitId++);
        }
      }
    }
  }

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary =
              expandToTwoQubits(oneQ, static_cast<QubitId>(*qid)) * unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          // Reorder the gate's (operand0, operand1) layout into the canonical
          // (qubit 0, qubit 1) order used by `unitary`.
          const llvm::SmallVector<QubitId, 2> ids{static_cast<QubitId>(*q0id),
                                                  static_cast<QubitId>(*q1id)};
          unitary = fixTwoQubitMatrixQubitOrder(twoQ, ids) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }
      }
    }
  }

  if (nextQubitId != 2) {
    return std::nullopt;
  }
  return unitary;
}

/// Kronecker-embed ``matrix`` on wire ``q`` into a ``2^N``-dim unitary (same
/// index bit order as QCO 4×4 matrices: wire 0 is the high bit).
TestMatrix expandOneQToN(const Matrix2x2& matrix, std::size_t q,
                         std::size_t numQubits) {
  const std::size_t dim = 1ULL << numQubits;
  TestMatrix full(dim);
  const auto bit = numQubits - 1 - q;
  const std::size_t mask = 1ULL << bit;
  for (std::size_t col = 0; col < dim; ++col) {
    const std::size_t sIn = (col >> bit) & 1ULL;
    const std::size_t rest = col & ~mask;
    for (std::size_t sOut = 0; sOut < 2; ++sOut) {
      const std::size_t row = rest | (sOut << bit);
      full(row, col) = matrix(sOut, sIn);
    }
  }
  return full;
}

/// Embed ``matrix`` on wires ``q0``, ``q1`` into a ``2^N``-dim unitary.
TestMatrix expandTwoQToN(const Matrix4x4& matrix, std::size_t q0,
                         std::size_t q1, std::size_t numQubits) {
  const std::size_t dim = 1ULL << numQubits;
  TestMatrix full(dim);
  const auto bit0 = numQubits - 1 - q0;
  const auto bit1 = numQubits - 1 - q1;
  const std::size_t mask0 = 1ULL << bit0;
  const std::size_t mask1 = 1ULL << bit1;
  const std::size_t maskBoth = mask0 | mask1;
  for (std::size_t col = 0; col < dim; ++col) {
    const std::size_t s0In = (col >> bit0) & 1ULL;
    const std::size_t s1In = (col >> bit1) & 1ULL;
    // 2-bit index for the pair matches QCO 4×4 row/column layout.
    const std::size_t smallIn = (s0In << 1) | s1In;
    const std::size_t rest = col & ~maskBoth;
    for (std::size_t smallOut = 0; smallOut < 4; ++smallOut) {
      const std::size_t s0Out = (smallOut >> 1) & 1ULL;
      const std::size_t s1Out = smallOut & 1ULL;
      const std::size_t row = rest | (s0Out << bit0) | (s1Out << bit1);
      full(row, col) = matrix(smallOut, smallIn);
    }
  }
  return full;
}

/// Full ``2^N`` unitary from a QCO module (``alloc`` / ``static``, 1q/2q
/// unitaries, ``ctrl`` with X/Z body). ``std::nullopt`` on unsupported ops or
/// if ``N`` exceeds ``maxQubits``.
std::optional<TestMatrix>
computeNQubitUnitaryFromModule(const OwningOpRef<ModuleOp>& moduleOp,
                               std::size_t maxQubits) {
  ModuleOp module = moduleOp.get();
  if (!module) {
    return std::nullopt;
  }

  llvm::DenseMap<Value, std::size_t> qubitIds;
  std::size_t numQubits = 0;

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        if (auto alloc = llvm::dyn_cast<qco::AllocOp>(&rawOp)) {
          if (numQubits >= maxQubits) {
            return std::nullopt;
          }
          qubitIds.try_emplace(alloc.getResult(), numQubits++);
        } else if (auto staticOp = llvm::dyn_cast<qco::StaticOp>(&rawOp)) {
          const auto idx = static_cast<std::size_t>(staticOp.getIndex());
          if (idx >= maxQubits) {
            return std::nullopt;
          }
          qubitIds.try_emplace(staticOp.getResult(), idx);
          numQubits = std::max(numQubits, idx + 1);
        }
      }
    }
  }

  if (numQubits == 0) {
    return std::nullopt;
  }

  TestMatrix unitary = TestMatrix::identity(1ULL << numQubits);

  auto getQubitId = [&](Value qubit) -> std::optional<std::size_t> {
    auto it = qubitIds.find(qubit);
    if (it == qubitIds.end()) {
      return std::nullopt;
    }
    return it->second;
  };

  for (auto func : module.getOps<func::FuncOp>()) {
    for (auto& block : func.getBlocks()) {
      for (auto& rawOp : block.getOperations()) {
        auto op = llvm::dyn_cast<qco::UnitaryOpInterface>(&rawOp);
        if (!op) {
          continue;
        }
        if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op.getOperation())) {
          continue;
        }

        if (op.isSingleQubit()) {
          const auto qIn = getUnitaryQubitOperand(op, 0);
          if (!qIn) {
            return std::nullopt;
          }
          auto qid = getQubitId(*qIn);
          if (!qid) {
            return std::nullopt;
          }
          Matrix2x2 oneQ;
          if (!extractSingleQubitMatrix(op, oneQ)) {
            return std::nullopt;
          }
          unitary = expandOneQToN(oneQ, *qid, numQubits) * unitary;
          const auto qOut = getUnitaryQubitResult(op, 0);
          if (!qOut) {
            return std::nullopt;
          }
          qubitIds[*qOut] = *qid;
          continue;
        }

        if (op.isTwoQubit()) {
          const auto q0In = getUnitaryQubitOperand(op, 0);
          const auto q1In = getUnitaryQubitOperand(op, 1);
          if (!q0In || !q1In) {
            return std::nullopt;
          }
          auto q0id = getQubitId(*q0In);
          auto q1id = getQubitId(*q1In);
          if (!q0id || !q1id) {
            return std::nullopt;
          }
          Matrix4x4 twoQ;
          if (!extractTwoQubitMatrix(op, twoQ)) {
            return std::nullopt;
          }
          unitary = expandTwoQToN(twoQ, *q0id, *q1id, numQubits) * unitary;
          const auto q0Out = getUnitaryQubitResult(op, 0);
          const auto q1Out = getUnitaryQubitResult(op, 1);
          if (!q0Out || !q1Out) {
            return std::nullopt;
          }
          qubitIds[*q0Out] = *q0id;
          qubitIds[*q1Out] = *q1id;
          continue;
        }
      }
    }
  }

  return unitary;
}

} // namespace mlir::qco::native_synth_test

using namespace mlir::qco::native_synth_test;

namespace {

struct NativeSynthMenuRow {
  const char* name;
  const char* nativeGates;
  bool (*isNative)(OwningOpRef<ModuleOp>&);
};

// --- Inline circuit builders ---

static void broadOneQThenCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.y(q1);
  b.h(q0);
  b.sx(q1);
  b.rx(0.13, q0);
  b.ry(-0.47, q1);
  b.rz(0.29, q0);
  b.cz(q0, q1);
}

static void zeroAngleThenCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.rx(0.0, q0);
  b.ry(0.0, q1);
  b.rz(0.0, q0);
  b.p(0.0, q1);
  b.cz(q0, q1);
}

static void ibmFractionalGateFamilies(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.rx(0.13, q1);
  b.cx(q0, q1);
  b.cz(q1, q0);
  b.swap(q0, q1);
  b.rzz(-0.33, q0, q1);
  b.rzx(0.41, q0, q1);
}

static void hstycxTwoQ(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.s(q0);
  b.t(q0);
  b.y(q0);
  b.cx(q0, q1);
}

static void cxYOnQ1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.cx(q0, q1);
  b.y(q1);
}

static void hCxTOnQ1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q1);
  b.cx(q0, q1);
  b.t(q1);
}

static void xYSXCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.y(q0);
  b.sx(q0);
  b.cz(q0, q1);
}

static void hYCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q0);
  b.cx(q0, q1);
}

static void zCx(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.z(q0);
  b.cx(q0, q1);
}

static void xHCz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.x(q0);
  b.h(q0);
  b.cz(q0, q1);
}

static void hq0Yq1CxSq0(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.y(q1);
  b.cx(q0, q1);
  b.s(q0);
}

static void hCxSq1(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.s(q1);
}

static void threeQGhz(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  const auto q2 = b.allocQubit();
  b.h(q0);
  b.cx(q0, q1);
  b.cx(q1, q2);
}

static void determinismSwap(mlir::qc::QCProgramBuilder& b) {
  const auto q0 = b.allocQubit();
  const auto q1 = b.allocQubit();
  b.swap(q0, q1);
  b.dealloc(q0);
  b.dealloc(q1);
}

} // namespace

// --- NativeSpec / NativePolicy ---

TEST(NativeSpecTest, ResolveIbmBasicCx) {
  const auto spec = decomposition::parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  EXPECT_TRUE(spec->allowedGates.contains(decomposition::NativeGateKind::Cx));
  EXPECT_TRUE(spec->allowedGates.contains(decomposition::NativeGateKind::X));
  EXPECT_FALSE(spec->allowRzz);
}

TEST(NativeSpecTest, ResolveRejectsUnknownToken) {
  EXPECT_FALSE(
      decomposition::parseNativeSpec("x,sx,rz,not-a-gate").has_value());
}

TEST(NativeSpecTest, PhaseAliasPMatchesRzInIbmStyleMenu) {
  const auto pMenu = decomposition::parseNativeSpec("x,sx,p,cx");
  const auto rzMenu = decomposition::parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(pMenu);
  ASSERT_TRUE(rzMenu);
  EXPECT_EQ(pMenu->allowedGates, rzMenu->allowedGates);
}

TEST(NativePolicyTest, UsesCxAndCzFromResolvedSpec) {
  const auto cxOnly = decomposition::parseNativeSpec("u,cx");
  ASSERT_TRUE(cxOnly);
  EXPECT_TRUE(llvm::is_contained(cxOnly->entanglerBases,
                                 decomposition::EntanglerBasis::Cx));
  EXPECT_FALSE(llvm::is_contained(cxOnly->entanglerBases,
                                  decomposition::EntanglerBasis::Cz));

  const auto both = decomposition::parseNativeSpec("u,cx,cz");
  ASSERT_TRUE(both);
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases,
                                 decomposition::EntanglerBasis::Cx));
  EXPECT_TRUE(llvm::is_contained(both->entanglerBases,
                                 decomposition::EntanglerBasis::Cz));
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
class NativePolicyAllowsOpTest : public ::testing::Test {
protected:
  MLIRContext context;
  QCOProgramBuilder builder{&context};

  void SetUp() override {
    context.loadDialect<QCODialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<scf::SCFDialect>();
    builder.initialize();
  }
};

TEST_F(NativePolicyAllowsOpTest, AllowsSingleQubitOpRespectsMenu) {
  const auto spec = decomposition::parseNativeSpec("x,sx,rz,cx");
  ASSERT_TRUE(spec);
  Value q = builder.staticQubit(0);
  q = builder.x(q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  XOp xop;
  mod->walk([&](XOp op) {
    xop = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(xop);
  EXPECT_TRUE(allowsSingleQubitOp(
      llvm::cast<UnitaryOpInterface>(xop.getOperation()), *spec));
}

TEST_F(NativePolicyAllowsOpTest, RejectsSingleQubitOpNotInMenu) {
  const auto spec = decomposition::parseNativeSpec("u,cx");
  ASSERT_TRUE(spec);
  Value q = builder.staticQubit(0);
  q = builder.x(q);
  auto mod = builder.finalize();
  ASSERT_TRUE(mod);
  XOp xop;
  mod->walk([&](XOp op) {
    xop = op;
    return WalkResult::interrupt();
  });
  ASSERT_TRUE(xop);
  EXPECT_FALSE(allowsSingleQubitOp(
      llvm::cast<UnitaryOpInterface>(xop.getOperation()), *spec));
}

// --- Pass profile coverage ---

// NOLINTNEXTLINE(misc-use-internal-linkage)
class NativeSynthesisSwapProfileTest
    : public NativeSynthesisPassTest,
      public testing::WithParamInterface<NativeSynthMenuRow> {
public:
  using NativeSynthesisPassTest::onlyGenericU3CxOps;
  using NativeSynthesisPassTest::onlyIbmBasicCxOps;
  using NativeSynthesisPassTest::onlyIqmDefaultOps;
};

TEST_P(NativeSynthesisSwapProfileTest, DecomposesSwapToProfile) {
  const NativeSynthMenuRow& param = GetParam();
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::swap);
      },
      param.nativeGates, param.isNative);
}

INSTANTIATE_TEST_SUITE_P(
    SwapMenuMatrix, NativeSynthesisSwapProfileTest,
    testing::Values(
        NativeSynthMenuRow{"IbmBasicCx", "x,sx,rz,cx",
                           &NativeSynthesisSwapProfileTest::onlyIbmBasicCxOps},
        NativeSynthMenuRow{"GenericU3Cx", "u,cx",
                           &NativeSynthesisSwapProfileTest::onlyGenericU3CxOps},
        NativeSynthMenuRow{"IqmDefault", "r,cz",
                           &NativeSynthesisSwapProfileTest::onlyIqmDefaultOps}),
    [](const testing::TestParamInfo<NativeSynthMenuRow>& info) {
      return info.param.name;
    });

TEST_F(NativeSynthesisPassTest, DecomposesHstycxToIbmBasicCx) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hstycxTwoQ);
      },
      "x,sx,rz,cx", &NativeSynthesisPassTest::onlyIbmBasicCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesCxYOnQ1ToIqmDefault) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), cxYOnQ1); },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, BroadOneQCanonicalizationOnIqmDefault) {
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), broadOneQThenCz);
  runNativeSynthesis(moduleOp, "r,cz");
  EXPECT_TRUE(onlyIqmDefaultOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, ZeroAngleCanonicalizationOnRyRzCz) {
  auto moduleOp =
      mlir::qc::QCProgramBuilder::build(context.get(), zeroAngleThenCz);
  runNativeSynthesis(moduleOp, "ry,rz,cz");
  EXPECT_TRUE(onlyAxisPairRyRzCzOps(moduleOp));
}

TEST_F(NativeSynthesisPassTest, DecomposesCxToCzForIbmBasicCzProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hCxTOnQ1);
      },
      "x,sx,rz,cz", &NativeSynthesisPassTest::onlyIbmBasicCzOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIqmDefaultProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), xYSXCz); },
      "r,cz", &NativeSynthesisPassTest::onlyIqmDefaultOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToIbmFractionalProfile) {
  expectNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 ibmFractionalGateFamilies);
      },
      "x,sx,rz,rx,rzz,cz", &NativeSynthesisPassTest::onlyIbmFractionalOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesToAxisPairRxRzCxProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), hYCx); },
      "rx,rz,cx", &NativeSynthesisPassTest::onlyAxisPairRxRzCxOps);
}

TEST_F(NativeSynthesisPassTest, DecomposesRzToAxisPairRxRyCxProfile) {
  expectNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), zCx); },
      "rx,ry,cx", &NativeSynthesisPassTest::onlyAxisPairRxRyCxOps);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesGenericU3CxBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), hq0Yq1CxSq0);
      },
      "u,cx", &NativeSynthesisPassTest::onlyGenericU3CxOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, GenericProfileMatchesAxisPairRyRzCzBehavior) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), xHCz); },
      "ry,rz,cz", &NativeSynthesisPassTest::onlyAxisPairRyRzCzOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, CustomProfileAcceptsMultipleEntanglersMenu) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return mlir::qc::QCProgramBuilder::build(context.get(), hCxSq1); },
      "u,cx,cz", &NativeSynthesisPassTest::onlyGenericU3CxOrCzOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, FailsForUnsupportedNativeGateMenu) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(), mlir::qc::h);
      },
      "not-a-gate");
}

TEST_F(NativeSynthesisPassTest, FailsForNativeGateMenuWithoutSingleQEmitter) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::singleControlledX);
      },
      "cx,cz");
}

TEST_F(NativeSynthesisPassTest, FailsForMultiControlledGateStructure) {
  expectSynthesisFailure(
      [&] {
        return mlir::qc::QCProgramBuilder::build(context.get(),
                                                 mlir::qc::multipleControlledX);
      },
      "x,sx,rz,cx");
}

TEST_F(NativeSynthesisPassTest, CandidateSelectionIsDeterministicAcrossRuns) {
  auto buildFn = [&] {
    return mlir::qc::QCProgramBuilder::build(context.get(), determinismSwap);
  };
  auto firstModule = buildFn();
  runNativeSynthesis(firstModule, "u,cx");
  auto secondModule = buildFn();
  runNativeSynthesis(secondModule, "u,cx");
  EXPECT_EQ(moduleToString(firstModule), moduleToString(secondModule));
}

TEST_F(NativeSynthesisPassTest, ThreeQubitGhzEquivalentOnCoreProfiles) {
  for (const auto& profileCase : coreEquivalenceProfiles()) {
    auto expected = mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runQcToQco(expected);
    const auto expectedUnitary = computeNQubitUnitaryFromModule(expected);
    ASSERT_TRUE(expectedUnitary.has_value());

    auto synthesized =
        mlir::qc::QCProgramBuilder::build(context.get(), threeQGhz);
    runNativeSynthesis(synthesized, profileCase.nativeGates);
    EXPECT_TRUE(profileCase.isNative(synthesized));
    const auto synthesizedUnitary = computeNQubitUnitaryFromModule(synthesized);
    ASSERT_TRUE(synthesizedUnitary.has_value());
    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary));
  }
}
