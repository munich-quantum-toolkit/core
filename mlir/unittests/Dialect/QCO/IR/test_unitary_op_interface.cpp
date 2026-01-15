/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include <complex>
#include <functional>
#include <gtest/gtest.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <string>

namespace {

using namespace mlir;
using namespace std::complex_literals;

class QcoUnitaryOpInterfaceTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<qco::QCODialect, arith::ArithDialect, cf::ControlFlowDialect,
                func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  /**
   * @brief Build expected QCO IR programmatically and run canonicalization
   */
  [[nodiscard]] OwningOpRef<ModuleOp> buildQCOIR(
      const std::function<void(qco::QCOProgramBuilder&)>& buildFunc) const {
    qco::QCOProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    return builder.finalize();
  }

  /**
   * @brief Return list of unitary matrices in given function
   */
  template <typename MatrixType>
  [[nodiscard]] llvm::SmallVector<llvm::Expected<MatrixType>, 0>
  getMatricesInFunction(func::FuncOp funcOp) {
    llvm::SmallVector<llvm::Expected<MatrixType>, 0> matrices;
    for (auto&& op : funcOp.getOps()) {
      auto unitaryOp = llvm::dyn_cast_if_present<qco::UnitaryOpInterface>(op);
      if (unitaryOp) {
        if (auto matrix = unitaryOp.getUnitaryMatrix<MatrixType>()) {
          matrices.push_back(*matrix);
        } else {
          matrices.push_back(llvm::createStringError(
              "Failed to get matrix of gate '%d' (%s)", matrices.size(),
              unitaryOp.getBaseSymbol().data()));
        }
      }
    }
    return matrices;
  }

  /**
   * @brief Get text representation of given module.
   */
  [[nodiscard]] static std::string toString(ModuleOp moduleOp) {
    std::string buffer;
    llvm::raw_string_ostream serializeStream{buffer};
    moduleOp->print(serializeStream);
    return serializeStream.str();
  }

private:
  std::unique_ptr<MLIRContext> context;
};

} // namespace

TEST_F(QcoUnitaryOpInterfaceTest, getUnitaryMatrix2x2) {
  const auto expectedValues = std::array{
      Eigen::Matrix2cd{{1, 0}, {0, 1}},
      Eigen::Matrix2cd{{0.87758256, -0.47942554i}, {-0.47942554i, 0.87758256}},
      Eigen::Matrix2cd{{0.99500417, -0.09195267 - 0.03887696i},
                       {0.09537451 + 0.02950279i, 0.76102116 + 0.64099928i}}};
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    reg[0] = builder.id(reg[0]);
    reg[0] = builder.rx(1.0, reg[0]);
    reg[0] = builder.u(0.2, 0.3, 0.4, reg[0]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(*moduleOps.begin());

  llvm::SmallVector<Eigen::Matrix2cd> actualValues;
  for (auto&& op : funcOp.getOps()) {
    auto unitaryOp = llvm::dyn_cast<qco::UnitaryOpInterface>(op);
    if (unitaryOp) {
      auto matrix = unitaryOp.getUnitaryMatrix<Eigen::Matrix2cd>();
      ASSERT_TRUE(matrix) << toString(*moduleOp)
                          << "\nFailed to get matrix of gate "
                          << actualValues.size();
      actualValues.push_back(*matrix);
    }
  }

  ASSERT_EQ(actualValues.size(), expectedValues.size());
  for (std::size_t i = 0; i < actualValues.size(); ++i) {
    EXPECT_TRUE(actualValues[i].isApprox(expectedValues.at(i), 1e-8))
        << "Wrong matrix at gate " << i;
  }
}

TEST_F(QcoUnitaryOpInterfaceTest, combine2x2UnitaryMatrices) {
  // use Qiskit to build same circuit (`qc`) and get unitary using
  // `qiskit.quantum_info.Operator(qc).data`
  const auto expectedValue =
      Eigen::Matrix2cd{{-0.57126014 - 0.1499036i, -0.10275332 - 0.80039522i},
                       {0.6908436 - 0.41704421i, -0.47252452 - 0.35430187i}};
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(1, "q");
    reg[0] = builder.id(reg[0]);
    reg[0] = builder.h(reg[0]);
    reg[0] = builder.rx(1.2, reg[0]);
    reg[0] = builder.ry(1.3, reg[0]);
    reg[0] = builder.rz(1.4, reg[0]);
    reg[0] = builder.u(0.5, 0.4, 0.3, reg[0]);
    reg[0] = builder.x(reg[0]);
    reg[0] = builder.y(reg[0]);
    reg[0] = builder.z(reg[0]);
    reg[0] = builder.sx(reg[0]);
    reg[0] = builder.s(reg[0]);
    reg[0] = builder.p(0.2, reg[0]);
    reg[0] = builder.sdg(reg[0]);
    reg[0] = builder.t(reg[0]);
    reg[0] = builder.r(1.0, 0.9, reg[0]);
    reg[0] = builder.tdg(reg[0]);
    reg[0] = builder.sxdg(reg[0]);

    // not supported by Qiskit
    // reg[0] = builder.cgphase(2.7, reg[0]);
    // reg[0] = builder.u2(2.2, 0.5, reg[0]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(*moduleOps.begin());
  ASSERT_TRUE(funcOp);

  auto matrices = getMatricesInFunction<Eigen::Matrix2cd>(funcOp);

  Eigen::Matrix2cd combinedMatrix = Eigen::Matrix2cd::Identity();
  for (auto&& matrix : matrices) {
    ASSERT_TRUE(static_cast<bool>(matrix))
        << llvm::toString(matrix.takeError());
    combinedMatrix = *matrix * combinedMatrix;
  }

  EXPECT_TRUE(combinedMatrix.isApprox(expectedValue, 1e-8))
      << "Combination of matrices does not match expected matrix";
}

TEST_F(QcoUnitaryOpInterfaceTest, combine4x4UnitaryMatrices) {
  // use Qiskit to build same circuit (`qc`) and get unitary using
  // `qiskit.quantum_info.Operator(qc).data`:
  // qc = QuantumCircuit(2, 0)
  // qc.rxx(1.1, 0, 1);
  // qc.ryy(1.2, 0, 1);
  // qc.swap(0, 1);
  // qc.rzz(1.3, 0, 1);
  // qc.rzx(1.4, 0, 1);
  // qc.iswap(0, 1);
  // qc.ecr(0, 1);
  // qc.dcx(0, 1);
  // qc.append(XXMinusYYGate(2.0, 2.1), [0, 1]);
  // qc.append(XXPlusYYGate(2.2, 2.3), [0, 1]);
  // qc.cx(0, 1);
  // qc.crz(0.2, 0, 1);
  // print(Operator(qc).data)

  const auto expectedValue =
      Eigen::Matrix4cd{{-0.19081581 - 0.2947213i, -0.37097846 + 0.25410653i,
                        0.20121632 - 0.46723087i, 0.01790367 + 0.64453107i},
                       {0.52886313 + 0.11168466i, -0.38256759 + 0.30793566i,
                        -0.06979467 + 0.54125643i, 0.30827692 + 0.2716312i},
                       {0.22651995 - 0.64212905i, 0.23878723 + 0.02364174i,
                        -0.62742798 + 0.05455965i, -0.24786847 + 0.14387256i},
                       {-0.26601581 + 0.22394997i, 0.08948183 + 0.7007405i,
                        -0.02323066 + 0.21493068i, -0.57690346 - 0.02202962i}};
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");

    std::tie(reg[0], reg[1]) = builder.rxx(1.1, reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.ryy(1.2, reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.swap(reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.rzz(1.3, reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.rzx(1.4, reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.iswap(reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.ecr(reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.dcx(reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.xx_minus_yy(2.0, 2.1, reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.xx_plus_yy(2.2, 2.3, reg[0], reg[1]);

    // implicit conversion of dynamic matrix to fixed-size matrix, if size matches
    std::tie(reg[0], reg[1]) = builder.cx(reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.crz(0.2, reg[0], reg[1]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(*moduleOps.begin());
  ASSERT_TRUE(funcOp);

  auto matrices = getMatricesInFunction<Eigen::Matrix4cd>(funcOp);

  Eigen::Matrix4cd combinedMatrix = Eigen::Matrix4cd::Identity();
  for (auto&& matrix : matrices) {
    ASSERT_TRUE(static_cast<bool>(matrix))
        << llvm::toString(matrix.takeError());
    combinedMatrix = *matrix * combinedMatrix;
  }

  EXPECT_TRUE(combinedMatrix.isApprox(expectedValue, 1e-8))
      << "Combination of matrices does not match expected matrix";
}

TEST_F(QcoUnitaryOpInterfaceTest, getDynamicUnitaryMatrix) {
  const auto expectedValues = std::array{
      Eigen::MatrixXcd{{1, 0}, {0, 1}},
      Eigen::MatrixXcd{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}},
      Eigen::MatrixXcd{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}};
  auto moduleOp = buildQCOIR([](qco::QCOProgramBuilder& builder) {
    auto reg = builder.allocQubitRegister(2, "q");
    reg[0] = builder.id(reg[0]);
    std::tie(reg[0], reg[1]) = builder.swap(reg[0], reg[1]);
    std::tie(reg[0], reg[1]) = builder.cx(reg[0], reg[1]);
  });

  auto&& moduleOps = moduleOp->getBody()->getOperations();
  ASSERT_FALSE(moduleOps.empty());
  auto funcOp = llvm::dyn_cast<func::FuncOp>(*moduleOps.begin());
  ASSERT_TRUE(funcOp);

  auto actualValues = getMatricesInFunction<Eigen::MatrixXcd>(funcOp);

  ASSERT_EQ(actualValues.size(), expectedValues.size());
  for (std::size_t i = 0; i < actualValues.size(); ++i) {
    ASSERT_TRUE(static_cast<bool>(actualValues[i]))
        << llvm::toString(actualValues[i].takeError());
    EXPECT_TRUE(actualValues[i]->isApprox(expectedValues.at(i), 1e-8))
        << "Wrong matrix at gate " << i;
  }
}
