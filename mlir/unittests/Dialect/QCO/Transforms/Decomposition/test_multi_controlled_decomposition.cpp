/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::qco;

/// Full unitary verification via DD (k <= 8 only; cost grows quickly above
/// that).
constexpr std::array<std::size_t, 7> K_DD_CONTROL_COUNTS = {2, 3, 4, 5,
                                                            6, 7, 8};
/// Pass-only checks above k = 8; includes k = 22–24 for the one- vs
/// two-ancilla boundary at wire count n = k + 1 >= 23.
constexpr std::array<std::size_t, 11> K_SMOKE_CONTROL_COUNTS = {
    10, 12, 15, 18, 20, 22, 23, 24, 25, 28, 30,
};

enum class ControlledPauli : std::uint8_t { X, Z };

namespace {

class McxDecompositionTest : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, arith::ArithDialect, func::FuncDialect>();
    context_ = std::make_unique<MLIRContext>();
    context_->appendDialectRegistry(registry);
    context_->loadAllAvailableDialects();
  }

public:
  [[nodiscard]] MLIRContext* context() const { return context_.get(); }

private:
  std::unique_ptr<MLIRContext> context_;
};

class McxDdTest : public McxDecompositionTest,
                  public testing::WithParamInterface<std::size_t> {};

class MczDdTest : public McxDecompositionTest,
                  public testing::WithParamInterface<std::size_t> {};

class LargeMcxTest : public McxDecompositionTest,
                     public testing::WithParamInterface<std::size_t> {};

class LargeMczTest : public McxDecompositionTest,
                     public testing::WithParamInterface<std::size_t> {};

} // namespace

[[nodiscard]] static OwningOpRef<ModuleOp>
buildControlledPauliModule(MLIRContext* context, std::size_t numControls,
                           ControlledPauli pauli) {
  return QCOProgramBuilder::build(
      context, [numControls, pauli](QCOProgramBuilder& b) {
        SmallVector<Value> wires;
        wires.reserve(numControls + 1);
        for (std::size_t i = 0; i <= numControls; ++i) {
          wires.push_back(b.staticQubit(i));
        }
        const auto controls = ValueRange(wires).drop_back();
        const auto target = wires.back();
        if (pauli == ControlledPauli::X) {
          b.mcx(controls, target);
        } else {
          b.mcz(controls, target);
        }
      });
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildMcxModule(MLIRContext* context, std::size_t numControls) {
  return buildControlledPauliModule(context, numControls, ControlledPauli::X);
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildMczModule(MLIRContext* context, std::size_t numControls) {
  return buildControlledPauliModule(context, numControls, ControlledPauli::Z);
}

[[nodiscard]] static qc::QuantumComputation
funcOpToQuantumComputation(func::FuncOp funcOp, std::size_t& numQubits) {
  DenseMap<Value, std::size_t> qubitIndex;
  numQubits = 0;
  for (StaticOp staticOp : funcOp.getOps<StaticOp>()) {
    const auto index = static_cast<std::size_t>(staticOp.getIndex());
    qubitIndex[staticOp.getQubit()] = index;
    numQubits = std::max(numQubits, index + 1);
  }

  const auto mapQubit = [&qubitIndex](Value in, Value out) {
    const std::size_t q = qubitIndex.at(in);
    qubitIndex[out] = q;
    return static_cast<qc::Qubit>(q);
  };

  qc::QuantumComputation qc(numQubits);
  for (Operation& op : funcOp.getBody().front()) {
    if (isa<StaticOp, func::ReturnOp, SinkOp, arith::ConstantOp>(op)) {
      continue;
    }
    if (auto hOp = dyn_cast<HOp>(op)) {
      qc.h(mapQubit(hOp.getInputQubit(0), hOp.getOutputQubit(0)));
      continue;
    }
    if (auto xOp = dyn_cast<XOp>(op)) {
      qc.x(mapQubit(xOp.getInputQubit(0), xOp.getOutputQubit(0)));
      continue;
    }
    if (auto tOp = dyn_cast<TOp>(op)) {
      qc.t(mapQubit(tOp.getInputQubit(0), tOp.getOutputQubit(0)));
      continue;
    }
    if (auto tdgOp = dyn_cast<TdgOp>(op)) {
      qc.tdg(mapQubit(tdgOp.getInputQubit(0), tdgOp.getOutputQubit(0)));
      continue;
    }
    if (auto pOp = dyn_cast<POp>(op)) {
      const qc::Qubit q = mapQubit(pOp.getInputQubit(0), pOp.getOutputQubit(0));
      const auto theta = mlir::utils::valueToDouble(pOp.getTheta());
      EXPECT_TRUE(theta.has_value());
      qc.p(theta.value_or(0.0), q);
      continue;
    }
    if (auto ctrlOp = dyn_cast<CtrlOp>(op)) {
      EXPECT_EQ(ctrlOp.getNumControls(), 1U)
          << "decomposition must not leave multi-controlled gates";
      EXPECT_EQ(ctrlOp.getNumTargets(), 1U);
      const qc::Qubit control =
          mapQubit(ctrlOp.getControlsIn()[0], ctrlOp.getControlsOut()[0]);
      const qc::Qubit target =
          mapQubit(ctrlOp.getInputTarget(0), ctrlOp.getTargetsOut()[0]);
      qc.cx(control, target);
      continue;
    }
    ADD_FAILURE() << "unexpected op in decomposed circuit: "
                  << op.getName().getStringRef().str();
  }

  return qc;
}

static void expectImplementsControlledPauli(func::FuncOp funcOp,
                                            std::size_t numControls,
                                            ControlledPauli pauli) {
  std::size_t numQubits = 0;
  auto decomposedQc = funcOpToQuantumComputation(funcOp, numQubits);
  ASSERT_EQ(numQubits, numControls + 1);

  const auto dd = std::make_unique<dd::Package>(numQubits);

  qc::QuantumComputation referenceQc(numQubits);
  qc::Controls controls;
  for (std::size_t i = 0; i < numControls; ++i) {
    controls.emplace(static_cast<qc::Qubit>(i));
  }
  const auto target = static_cast<qc::Qubit>(numControls);
  if (pauli == ControlledPauli::X) {
    referenceQc.mcx(controls, target);
  } else {
    referenceQc.mcz(controls, target);
  }

  const dd::MatrixDD decomposedDD = dd::buildFunctionality(decomposedQc, *dd);
  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_TRUE(dd->multiply(decomposedDD, dd->conjugateTranspose(referenceDD))
                  .isIdentity(/*upToGlobalPhase=*/false));
}

static void expectImplementsMcx(func::FuncOp funcOp, std::size_t numControls) {
  expectImplementsControlledPauli(funcOp, numControls, ControlledPauli::X);
}

static void expectImplementsMcz(func::FuncOp funcOp, std::size_t numControls) {
  expectImplementsControlledPauli(funcOp, numControls, ControlledPauli::Z);
}

[[nodiscard]] static std::size_t countMultiControlledOps(ModuleOp moduleOp) {
  std::size_t count = 0;
  moduleOp.walk([&count](CtrlOp op) {
    if (op.getNumControls() >= 2) {
      ++count;
    }
  });
  return count;
}

static LogicalResult
runDecomposePass(ModuleOp moduleOp,
                 const DecomposeMultiControlledOptions& options = {}) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createDecomposeMultiControlled(options));
  return pm.run(moduleOp);
}

TEST_P(McxDdTest, ImplementsMcx) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMcxModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcx(funcOp, numControls);
}

INSTANTIATE_TEST_SUITE_P(McxDd, McxDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<std::size_t>& info) {
                           return "controls" + std::to_string(info.param);
                         });

TEST_P(MczDdTest, ImplementsMcz) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMczModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcz(funcOp, numControls);
}

INSTANTIATE_TEST_SUITE_P(MczDd, MczDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<std::size_t>& info) {
                           return "controls" + std::to_string(info.param);
                         });

TEST_P(LargeMcxTest, DecomposesWithoutMultiControlledGates) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMcxModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);
}

INSTANTIATE_TEST_SUITE_P(LargeMcx, LargeMcxTest,
                         testing::ValuesIn(K_SMOKE_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<std::size_t>& info) {
                           return "controls" + std::to_string(info.param);
                         });

TEST_P(LargeMczTest, DecomposesWithoutMultiControlledGates) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMczModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);
}

INSTANTIATE_TEST_SUITE_P(LargeMcz, LargeMczTest,
                         testing::ValuesIn(K_SMOKE_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<std::size_t>& info) {
                           return "controls" + std::to_string(info.param);
                         });

TEST_F(McxDecompositionTest, LeavesSingleControlledXUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.cx(builder.staticQubit(0), builder.staticQubit(1));
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);

  std::size_t singleControlledCount = 0;
  moduleOp->walk([&singleControlledCount](CtrlOp op) {
    if (op.getNumControls() == 1) {
      ++singleControlledCount;
    }
  });
  EXPECT_EQ(singleControlledCount, 1U);
}

TEST_F(McxDecompositionTest, LeavesSingleControlledZUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.cz(builder.staticQubit(0), builder.staticQubit(1));
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);

  std::size_t singleControlledCount = 0;
  moduleOp->walk([&singleControlledCount](CtrlOp op) {
    if (op.getNumControls() == 1) {
      ++singleControlledCount;
    }
  });
  EXPECT_EQ(singleControlledCount, 1U);
}

TEST_F(McxDecompositionTest, MinControlsKeepsToffoli) {
  auto moduleOp = buildMcxModule(context(), 2);
  ASSERT_TRUE(moduleOp);

  DecomposeMultiControlledOptions options;
  options.minControls = 3;
  ASSERT_TRUE(runDecomposePass(moduleOp.get(), options).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
}

TEST_F(McxDecompositionTest, MinControlsKeepsCcZ) {
  auto moduleOp = buildMczModule(context(), 2);
  ASSERT_TRUE(moduleOp);

  DecomposeMultiControlledOptions options;
  options.minControls = 3;
  ASSERT_TRUE(runDecomposePass(moduleOp.get(), options).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
}

TEST_F(McxDecompositionTest, DecomposesMcxAndMcz) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.mcx({builder.staticQubit(0), builder.staticQubit(1),
                     builder.staticQubit(2)},
                    builder.staticQubit(3));
        builder.mcz({builder.staticQubit(4), builder.staticQubit(5)},
                    builder.staticQubit(6));
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runDecomposePass(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);
}
