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
#include "mlir/Dialect/QCO/Transforms/Decomposition/MultiControlled.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"
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
#include <optional>
#include <string>
#include <tuple>

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

namespace {

enum class ControlledPauli : std::uint8_t { X, Z };

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
        return SmallVector<Value>{};
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

[[nodiscard]] static OwningOpRef<ModuleOp>
buildTwoControlledXModule(MLIRContext* context) {
  return buildMcxModule(context, 2);
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildRCCXModule(MLIRContext* context) {
  return QCOProgramBuilder::build(context, [](QCOProgramBuilder& b) {
    std::ignore = b.rccx(b.staticQubit(0), b.staticQubit(1), b.staticQubit(2));
    return SmallVector<Value>{};
  });
}

[[nodiscard]] static OwningOpRef<ModuleOp>
buildTwoControlledPhaseModule(MLIRContext* context, double theta) {
  return QCOProgramBuilder::build(context, [theta](QCOProgramBuilder& b) {
    b.mcp(theta, {b.staticQubit(0), b.staticQubit(1)}, b.staticQubit(2));
    return SmallVector<Value>{};
  });
}

static void appendRCCXElementaryOnQubits(qc::QuantumComputation& qc,
                                         qc::Qubit control0, qc::Qubit control1,
                                         qc::Qubit target) {
  qc.h(target);
  qc.t(target);
  qc.cx(control1, target);
  qc.tdg(target);
  qc.cx(control0, target);
  qc.t(target);
  qc.cx(control1, target);
  qc.tdg(target);
  qc.h(target);
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
    if (auto rzOp = dyn_cast<RZOp>(op)) {
      const qc::Qubit q =
          mapQubit(rzOp.getInputQubit(0), rzOp.getOutputQubit(0));
      const auto theta = mlir::utils::valueToDouble(rzOp.getTheta());
      EXPECT_TRUE(theta.has_value());
      qc.rz(theta.value_or(0.0), q);
      continue;
    }
    if (auto ctrlOp = dyn_cast<CtrlOp>(op)) {
      EXPECT_LE(ctrlOp.getNumControls(), 2U)
          << "decomposition must not leave gates with three or more controls";
      EXPECT_EQ(ctrlOp.getNumTargets(), 1U);
      const qc::Qubit target =
          mapQubit(ctrlOp.getInputTarget(0), ctrlOp.getTargetsOut()[0]);
      auto inner =
          utils::getSoleBodyUnitary<UnitaryOpInterface>(*ctrlOp.getBody());
      if (!inner) {
        ADD_FAILURE() << "ctrl body must contain a single unitary gate";
        continue;
      }
      if (ctrlOp.getNumControls() == 1U) {
        const qc::Qubit control =
            mapQubit(ctrlOp.getControlsIn()[0], ctrlOp.getControlsOut()[0]);
        if (isa<XOp>(inner.getOperation())) {
          qc.cx(control, target);
          continue;
        }
        if (isa<ZOp>(inner.getOperation())) {
          qc.cz(control, target);
          continue;
        }
        if (auto pOp = dyn_cast<POp>(inner.getOperation())) {
          const auto theta = mlir::utils::valueToDouble(pOp.getTheta());
          EXPECT_TRUE(theta.has_value());
          qc.cp(theta.value_or(0.0), control, target);
          continue;
        }
      } else {
        const qc::Qubit control0 =
            mapQubit(ctrlOp.getControlsIn()[0], ctrlOp.getControlsOut()[0]);
        const qc::Qubit control1 =
            mapQubit(ctrlOp.getControlsIn()[1], ctrlOp.getControlsOut()[1]);
        qc::Controls controls;
        controls.emplace(control0);
        controls.emplace(control1);
        if (isa<XOp>(inner.getOperation())) {
          qc.mcx(controls, target);
          continue;
        }
        if (isa<ZOp>(inner.getOperation())) {
          qc.mcz(controls, target);
          continue;
        }
        if (auto pOp = dyn_cast<POp>(inner.getOperation())) {
          const auto theta = mlir::utils::valueToDouble(pOp.getTheta());
          EXPECT_TRUE(theta.has_value());
          qc.mcp(theta.value_or(0.0), controls, target);
          continue;
        }
      }
      ADD_FAILURE() << "unexpected controlled gate in decomposed circuit: "
                    << inner.getOperation()->getName().getStringRef().str();
      continue;
    }
    if (auto rccxOp = dyn_cast<RCCXOp>(op)) {
      const qc::Qubit control0 =
          mapQubit(rccxOp.getQubit0In(), rccxOp.getQubit0Out());
      const qc::Qubit control1 =
          mapQubit(rccxOp.getQubit1In(), rccxOp.getQubit1Out());
      const qc::Qubit target =
          mapQubit(rccxOp.getQubit2In(), rccxOp.getQubit2Out());
      appendRCCXElementaryOnQubits(qc, control0, control1, target);
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
  EXPECT_EQ(decomposedDD, referenceDD);
}

static void expectImplementsMcx(func::FuncOp funcOp, std::size_t numControls) {
  expectImplementsControlledPauli(funcOp, numControls, ControlledPauli::X);
}

static void expectImplementsMcz(func::FuncOp funcOp, std::size_t numControls) {
  expectImplementsControlledPauli(funcOp, numControls, ControlledPauli::Z);
}

static void expectImplementsRCCX(func::FuncOp funcOp) {
  std::size_t numQubits = 0;
  auto decomposedQc = funcOpToQuantumComputation(funcOp, numQubits);
  ASSERT_EQ(numQubits, 3U);

  const auto dd = std::make_unique<dd::Package>(numQubits);

  qc::QuantumComputation referenceQc(numQubits);
  appendRCCXElementaryOnQubits(referenceQc, static_cast<qc::Qubit>(0),
                               static_cast<qc::Qubit>(1),
                               static_cast<qc::Qubit>(2));

  const dd::MatrixDD decomposedDD = dd::buildFunctionality(decomposedQc, *dd);
  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_EQ(decomposedDD, referenceDD);
}

static void expectImplementsTwoControlledX(func::FuncOp funcOp) {
  expectImplementsMcx(funcOp, 2);
}

static void expectImplementsTwoControlledPhase(func::FuncOp funcOp,
                                               double theta) {
  std::size_t numQubits = 0;
  auto decomposedQc = funcOpToQuantumComputation(funcOp, numQubits);
  ASSERT_EQ(numQubits, 3U);

  const auto dd = std::make_unique<dd::Package>(numQubits);

  qc::QuantumComputation referenceQc(numQubits);
  qc::Controls controls;
  controls.emplace(static_cast<qc::Qubit>(0));
  controls.emplace(static_cast<qc::Qubit>(1));
  referenceQc.mcp(theta, controls, static_cast<qc::Qubit>(2));

  const dd::MatrixDD decomposedDD = dd::buildFunctionality(decomposedQc, *dd);
  const dd::MatrixDD referenceDD = dd::buildFunctionality(referenceQc, *dd);
  EXPECT_EQ(decomposedDD, referenceDD);
}

[[nodiscard]] static std::size_t
countMultiControlledOps(ModuleOp moduleOp, std::size_t minControls = 2) {
  std::size_t count = 0;
  moduleOp.walk([&count, minControls](CtrlOp op) {
    if (op.getNumControls() >= minControls) {
      ++count;
    }
  });
  return count;
}

[[nodiscard]] static std::optional<CtrlOp>
findSoleMultiControlledCtrlOp(ModuleOp moduleOp) {
  CtrlOp found;
  std::size_t count = 0;
  moduleOp.walk([&](CtrlOp op) {
    if (op.getNumControls() >= 2) {
      found = op;
      ++count;
    }
  });
  if (count != 1) {
    return std::nullopt;
  }
  return found;
}

static LogicalResult runMultiAndThreeControlledPasses(
    ModuleOp moduleOp, const DecomposeMultiControlledOptions& options = {}) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createDecomposeMultiControlled(options));
  pm.addPass(createDecomposeThreeControlled());
  return pm.run(moduleOp);
}

static LogicalResult runThreeControlledPass(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createDecomposeThreeControlled());
  return pm.run(moduleOp);
}

static LogicalResult runTwoControlledPass(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createDecomposeTwoControlled());
  return pm.run(moduleOp);
}

static LogicalResult runFullDecompositionPipeline(
    ModuleOp moduleOp, const DecomposeMultiControlledOptions& options = {}) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createDecomposeMultiControlled(options));
  pm.addPass(createDecomposeThreeControlled());
  pm.addPass(createDecomposeTwoControlled());
  return pm.run(moduleOp);
}

TEST_P(McxDdTest, ImplementsMcx) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMcxModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

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
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcz(funcOp, numControls);
}

INSTANTIATE_TEST_SUITE_P(MczDd, MczDdTest,
                         testing::ValuesIn(K_DD_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<std::size_t>& info) {
                           return "controls" + std::to_string(info.param);
                         });

TEST_P(LargeMcxTest, DecomposesWithoutThreeOrMoreControlledGates) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMcxModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
}

INSTANTIATE_TEST_SUITE_P(LargeMcx, LargeMcxTest,
                         testing::ValuesIn(K_SMOKE_CONTROL_COUNTS),
                         [](const testing::TestParamInfo<std::size_t>& info) {
                           return "controls" + std::to_string(info.param);
                         });

TEST_P(LargeMczTest, DecomposesWithoutThreeOrMoreControlledGates) {
  const std::size_t numControls = GetParam();
  auto moduleOp = buildMczModule(context(), numControls);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
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
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

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
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 0U);

  std::size_t singleControlledCount = 0;
  moduleOp->walk([&singleControlledCount](CtrlOp op) {
    if (op.getNumControls() == 1) {
      ++singleControlledCount;
    }
  });
  EXPECT_EQ(singleControlledCount, 1U);
}

TEST_F(McxDecompositionTest, FullyDecomposesFourControlMcx) {
  auto moduleOp = buildMcxModule(context(), 4);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runFullDecompositionPipeline(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 0U);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcx(funcOp, 4);
}

TEST_F(McxDecompositionTest, DecomposesRCCXToElementaryGates) {
  auto moduleOp = buildRCCXModule(context());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runTwoControlledPass(moduleOp.get()).succeeded());

  bool foundRCCX = false;
  moduleOp->walk([&foundRCCX](RCCXOp) { foundRCCX = true; });
  EXPECT_FALSE(foundRCCX);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsRCCX(funcOp);
}

TEST_F(McxDecompositionTest, DecomposesCcxViaTwoControlledPass) {
  auto moduleOp = buildTwoControlledXModule(context());
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runTwoControlledPass(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 0U);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsTwoControlledX(funcOp);
}

TEST_F(McxDecompositionTest, DecomposesCcPhaseViaTwoControlledPass) {
  constexpr double theta = 0.37;
  auto moduleOp = buildTwoControlledPhaseModule(context(), theta);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runTwoControlledPass(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 0U);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsTwoControlledPhase(funcOp, theta);
}

TEST_F(McxDecompositionTest, DecomposesCczViaTwoControlledPass) {
  auto moduleOp = buildMczModule(context(), 2);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runTwoControlledPass(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 0U);

  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcz(funcOp, 2);
}

TEST_F(McxDecompositionTest,
       LeavesSingleControlledXUntouchedByTwoControlledPass) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.cx(builder.staticQubit(0), builder.staticQubit(1));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runTwoControlledPass(moduleOp.get()).succeeded());

  std::size_t singleControlledCount = 0;
  moduleOp->walk([&singleControlledCount](CtrlOp op) {
    if (op.getNumControls() == 1) {
      ++singleControlledCount;
    }
  });
  EXPECT_EQ(singleControlledCount, 1U);
}

TEST_F(McxDecompositionTest, LeavesTwoControlledXUntouched) {
  auto moduleOp = buildMcxModule(context(), 2);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 1U);
}

TEST_F(McxDecompositionTest, LeavesTwoControlledZUntouched) {
  auto moduleOp = buildMczModule(context(), 2);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 2), 1U);
}

TEST_F(McxDecompositionTest, DecomposesThreeControlledXViaThreeControlledPass) {
  auto moduleOp = buildMcxModule(context(), 3);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runThreeControlledPass(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcx(funcOp, 3);
}

TEST_F(McxDecompositionTest, DecomposesThreeControlledZViaThreeControlledPass) {
  auto moduleOp = buildMczModule(context(), 3);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runThreeControlledPass(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
  auto funcOp = *moduleOp->getBody()->getOps<func::FuncOp>().begin();
  expectImplementsMcz(funcOp, 3);
}

TEST_F(McxDecompositionTest, MinControlsKeepsToffoli) {
  auto moduleOp = buildMcxModule(context(), 2);
  ASSERT_TRUE(moduleOp);

  DecomposeMultiControlledOptions options;
  options.minControls = 3;
  PassManager pm(moduleOp->getContext());
  pm.addPass(createDecomposeMultiControlled(options));
  ASSERT_TRUE(pm.run(moduleOp.get()).succeeded());
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
}

TEST_F(McxDecompositionTest, MinControlsKeepsCcZ) {
  auto moduleOp = buildMczModule(context(), 2);
  ASSERT_TRUE(moduleOp);

  DecomposeMultiControlledOptions options;
  options.minControls = 3;
  PassManager pm(moduleOp->getContext());
  pm.addPass(createDecomposeMultiControlled(options));
  ASSERT_TRUE(pm.run(moduleOp.get()).succeeded());
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
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  EXPECT_EQ(countMultiControlledOps(moduleOp.get(), 3), 0U);
  EXPECT_GE(countMultiControlledOps(moduleOp.get(), 2), 1U);
}

TEST_F(McxDecompositionTest, PassFailsWhenMinControlsBelowTwo) {
  auto moduleOp = buildMcxModule(context(), 3);
  ASSERT_TRUE(moduleOp);

  DecomposeMultiControlledOptions options;
  options.minControls = 1;
  EXPECT_FALSE(
      runMultiAndThreeControlledPasses(moduleOp.get(), options).succeeded());
}

TEST_F(McxDecompositionTest, LeavesMultiOpCtrlUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        const Value c0 = builder.staticQubit(0);
        const Value c1 = builder.staticQubit(1);
        const Value target = builder.staticQubit(2);
        builder.ctrl({c0, c1}, target, [&](Value targetArg) -> Value {
          const Value afterX = builder.x(targetArg);
          return builder.y(afterX);
        });
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  const auto ctrlOpOpt = findSoleMultiControlledCtrlOp(moduleOp.get());
  ASSERT_TRUE(ctrlOpOpt.has_value());
  CtrlOp ctrlOp = *ctrlOpOpt;
  EXPECT_EQ(ctrlOp.getNumControls(), 2U);
  EXPECT_EQ(ctrlOp.getNumTargets(), 1U);
  EXPECT_EQ(ctrlOp.getNumBodyUnitaries(), 2U);
  EXPECT_TRUE(isa<XOp>(ctrlOp.getBodyUnitary(0).getOperation()));
  EXPECT_TRUE(isa<YOp>(ctrlOp.getBodyUnitary(1).getOperation()));
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
}

TEST_F(McxDecompositionTest, LeavesMultiControlledHUntouched) {
  auto moduleOp =
      QCOProgramBuilder::build(context(), [](QCOProgramBuilder& builder) {
        builder.mch({builder.staticQubit(0), builder.staticQubit(1)},
                    builder.staticQubit(2));
        return SmallVector<Value>{};
      });
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(runMultiAndThreeControlledPasses(moduleOp.get()).succeeded());

  const auto ctrlOpOpt = findSoleMultiControlledCtrlOp(moduleOp.get());
  ASSERT_TRUE(ctrlOpOpt.has_value());
  CtrlOp ctrlOp = *ctrlOpOpt;
  EXPECT_EQ(ctrlOp.getNumControls(), 2U);
  EXPECT_EQ(ctrlOp.getNumTargets(), 1U);
  ASSERT_EQ(ctrlOp.getNumBodyUnitaries(), 1U);
  EXPECT_TRUE(isa<HOp>(ctrlOp.getBodyUnitary(0).getOperation()));
  EXPECT_EQ(countMultiControlledOps(moduleOp.get()), 1U);
}

TEST_F(McxDecompositionTest, SynthesizeTwoControlledRequiresEightByEight) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context());
        builder.initialize();
        const Value q0 = builder.staticQubit(0);
        const Value q1 = builder.staticQubit(1);
        const Value q2 = builder.staticQubit(2);
        decomposition::synthesizeTwoControlled(
            builder, builder.getLoc(), q0, q1, q2, DynamicMatrix::identity(4));
      },
      "synthesizeTwoControlled requires an 8x8 unitary matrix");
}

TEST_F(McxDecompositionTest, SynthesizeMultiControlledRequiresThreeControls) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context());
        builder.initialize();
        const Value q0 = builder.staticQubit(0);
        const Value q1 = builder.staticQubit(1);
        decomposition::synthesizeMultiControlled(
            builder, builder.getLoc(), ValueRange{q0}, q1, 2,
            decomposition::ControlledTarget::X);
      },
      "synthesizeMultiControlled requires at least 3 control qubits");
}

TEST_F(McxDecompositionTest, SynthesizeThreeControlledRequiresThreeControls) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context());
        builder.initialize();
        const Value q0 = builder.staticQubit(0);
        const Value q1 = builder.staticQubit(1);
        const Value q2 = builder.staticQubit(2);
        decomposition::synthesizeThreeControlled(
            builder, builder.getLoc(), ValueRange{q0, q1}, q2,
            decomposition::ControlledTarget::X);
      },
      "three-controlled synthesis requires exactly 3 control qubits");
}

TEST_F(McxDecompositionTest, SynthesizeThreeControlledZRequiresThreeControls) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context());
        builder.initialize();
        const Value q0 = builder.staticQubit(0);
        const Value q1 = builder.staticQubit(1);
        const Value q2 = builder.staticQubit(2);
        decomposition::synthesizeThreeControlled(
            builder, builder.getLoc(), ValueRange{q0, q1}, q2,
            decomposition::ControlledTarget::Z);
      },
      "three-controlled synthesis requires exactly 3 control qubits");
}
