/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"
#include "qc_programs.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>

using namespace mlir;

namespace {

struct QCToQCOTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<qco::QCOProgramBuilder> referenceBuilder;
  bool expectsCompleteTensorState = false;
  bool skipReferenceComparison = false;

  friend std::ostream& operator<<(std::ostream& os,
                                  const QCToQCOTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QCToQCOTestCase& info) {
  return os << "QCToQCO{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QCToQCOTest : public testing::TestWithParam<QCToQCOTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry
        .insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                qtensor::QTensorDialect, arith::ArithDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static LogicalResult runQCToQCOConversion(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createQCToQCO());
  return pm.run(module);
}

namespace {

class QCToQCORegressionTest : public testing::Test {
protected:
  MLIRContext context;

  QCToQCORegressionTest() {
    DialectRegistry registry;
    registry
        .insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                qtensor::QTensorDialect, arith::ArithDialect, func::FuncDialect,
                memref::MemRefDialect, scf::SCFDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  void expectNoQCOperations(ModuleOp module) {
    bool retainsQCOperations = false;
    module.walk([&](Operation* operation) {
      retainsQCOperations |=
          operation->getDialect() == context.getLoadedDialect<qc::QCDialect>();
    });
    EXPECT_FALSE(retainsQCOperations);
  }

public:
  static void expectOperationLocalRegisterAccesses(ModuleOp moduleOp) {
    std::size_t extracts = 0;
    std::size_t inserts = 0;
    llvm::DenseSet<Operation*> committedExtracts;
    moduleOp.walk([&](qtensor::ExtractOp extract) {
      ++extracts;
      ASSERT_TRUE(extract.getResult().hasOneUse());
      auto* user = *extract.getResult().getUsers().begin();
      EXPECT_EQ(user->getDialect(),
                moduleOp.getContext()->getOrLoadDialect<qco::QCODialect>());
    });
    moduleOp.walk([&](qtensor::InsertOp insert) {
      ++inserts;
      auto* producer = insert.getScalar().getDefiningOp();
      ASSERT_NE(producer, nullptr);
      EXPECT_EQ(producer->getDialect(),
                moduleOp.getContext()->getOrLoadDialect<qco::QCODialect>());

      Value input;
      if (auto unitary = dyn_cast<qco::UnitaryOpInterface>(producer)) {
        input = unitary.getInputForOutput(insert.getScalar());
      } else if (auto measure = dyn_cast<qco::MeasureOp>(producer)) {
        EXPECT_EQ(insert.getScalar(), measure.getQubitOut());
        input = measure.getQubitIn();
      } else if (auto reset = dyn_cast<qco::ResetOp>(producer)) {
        EXPECT_EQ(insert.getScalar(), reset.getQubitOut());
        input = reset.getQubitIn();
      }
      ASSERT_TRUE(input);

      auto extract = input.getDefiningOp<qtensor::ExtractOp>();
      ASSERT_TRUE(extract);
      EXPECT_TRUE(
          isEqualConstantIntOrValue(extract.getIndex(), insert.getIndex()));
      EXPECT_EQ(extract->getBlock(), producer->getBlock());
      EXPECT_EQ(producer->getBlock(), insert->getBlock());
      EXPECT_TRUE(extract->isBeforeInBlock(producer));
      EXPECT_TRUE(producer->isBeforeInBlock(insert));
      committedExtracts.insert(extract);
    });
    EXPECT_GT(extracts, 0U);
    EXPECT_EQ(extracts, inserts);
    EXPECT_EQ(committedExtracts.size(), extracts);

    moduleOp.walk([&](memref::LoadOp load) {
      EXPECT_FALSE(isa<qc::QubitType>(load.getMemRefType().getElementType()));
    });
  }

  static void expectStructuredStateUsesCompleteTensors(ModuleOp moduleOp) {
    bool sawStructuredQuantumState = false;
    moduleOp.walk([&](Operation* operation) {
      if (!isa<qco::IfOp, qco::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
              operation)) {
        return;
      }
      const auto isQubitTensor = [](Type type) {
        const auto tensor = dyn_cast<RankedTensorType>(type);
        return tensor && isa<qco::QubitType>(tensor.getElementType());
      };
      const bool hasTensorOperand =
          llvm::any_of(operation->getOperandTypes(), isQubitTensor);
      const bool hasTensorResult =
          llvm::any_of(operation->getResultTypes(), isQubitTensor);
      EXPECT_TRUE(hasTensorOperand);
      EXPECT_TRUE(hasTensorResult);
      const auto tensorOperands = llvm::to_vector(
          llvm::make_filter_range(operation->getOperandTypes(), isQubitTensor));
      const auto tensorResults = llvm::to_vector(
          llvm::make_filter_range(operation->getResultTypes(), isQubitTensor));
      EXPECT_EQ(tensorOperands, tensorResults);
      for (Region& region : operation->getRegions()) {
        if (region.empty()) {
          continue;
        }
        const auto tensorArguments = llvm::to_vector(llvm::make_filter_range(
            region.front().getArgumentTypes(), isQubitTensor));
        EXPECT_EQ(tensorArguments, tensorResults);
      }
      sawStructuredQuantumState = true;
    });
    EXPECT_TRUE(sawStructuredQuantumState);
  }
};

} // namespace

TEST_F(QCToQCORegressionTest, PreservesForResultsWithQuantumState) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i1 attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %loop = scf.for %i = %c0 to %c1 step %c1
        iter_args(%flag = %true) -> (i1) {
      qc.h %qc : !qc.qubit
      scf.yield %flag : i1
    }
    qc.dealloc %qc : !qc.qubit
    return %loop : i1
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  bool sawLoop = false;
  module->walk([&](scf::ForOp loop) {
    sawLoop = true;
    EXPECT_EQ(loop.getNumResults(), 2);
    EXPECT_TRUE(loop.getResult(0).getType().isInteger(1));
    EXPECT_TRUE(isa<qco::QubitType>(loop.getResult(1).getType()));
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    EXPECT_EQ(yield.getNumOperands(), loop.getNumResults());
    EXPECT_TRUE(llvm::equal(yield.getOperandTypes(), loop.getResultTypes()));
  });
  EXPECT_TRUE(sawLoop);

  expectNoQCOperations(*module);
}

TEST_F(QCToQCORegressionTest, PreservesWhileConditionArgumentsAndOrdering) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i1 attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    %true = arith.constant true
    %zero = arith.constant 0 : i64
    %result:2 = scf.while (%flag = %true, %count = %zero)
        : (i1, i64) -> (i64, i1) {
      qc.h %qc : !qc.qubit
      %false = arith.constant false
      scf.condition(%false) %count, %flag : i64, i1
    } do {
    ^bb0(%count: i64, %flag: i1):
      qc.x %qc : !qc.qubit
      scf.yield %flag, %count : i1, i64
    }
    qc.dealloc %qc : !qc.qubit
    return %result#1 : i1
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  bool sawWhile = false;
  module->walk([&](scf::WhileOp loop) {
    sawWhile = true;
    ASSERT_EQ(loop.getNumResults(), 3);
    EXPECT_TRUE(loop.getResult(0).getType().isInteger(64));
    EXPECT_TRUE(loop.getResult(1).getType().isInteger(1));
    EXPECT_TRUE(isa<qco::QubitType>(loop.getResult(2).getType()));
    auto condition =
        cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
    EXPECT_TRUE(
        llvm::equal(condition.getArgs().getTypes(), loop.getResultTypes()));
    auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
    EXPECT_TRUE(
        llvm::equal(yield.getOperandTypes(), loop.getInits().getTypes()));
  });
  EXPECT_TRUE(sawWhile);
  expectNoQCOperations(*module);
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(*module)));
  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  APInt result;
  ASSERT_TRUE(matchPattern(returnOp.getOperand(0), m_ConstantInt(&result)));
  EXPECT_TRUE(result.isOne());
}

TEST_F(QCToQCORegressionTest, IgnoresClassicalRegisterLoadsInWhileState) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> memref<1xi1>
      attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    %c = memref.alloc() : memref<1xi1>
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    memref.store %true, %c[%c0] : memref<1xi1>
    %result = scf.while (%running = %true) : (i1) -> i1 {
      %condition = memref.load %c[%c0] : memref<1xi1>
      scf.condition(%condition) %running : i1
    } do {
    ^bb0(%running: i1):
      qc.x %qc : !qc.qubit
      %false = arith.constant false
      memref.store %false, %c[%c0] : memref<1xi1>
      scf.yield %false : i1
    }
    qc.dealloc %qc : !qc.qubit
    return %c : memref<1xi1>
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  expectNoQCOperations(*module);

  bool retainsClassicalRegister = false;
  module->walk([&](memref::LoadOp op) {
    retainsClassicalRegister |=
        op.getMemRefType().getElementType().isInteger(1);
  });
  EXPECT_TRUE(retainsClassicalRegister);
}

TEST_F(QCToQCORegressionTest, ConvertsTypeChangingWhileWithQuantumState) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    %initial = arith.constant 1.0 : f32
    %result = scf.while (%input = %initial) : (f32) -> i64 {
      qc.h %qc : !qc.qubit
      %condition = arith.constant true
      %next = arith.constant 7 : i64
      scf.condition(%condition) %next : i64
    } do {
    ^bb0(%input: i64):
      qc.x %qc : !qc.qubit
      %next = arith.sitofp %input : i64 to f32
      scf.yield %next : f32
    }
    qc.dealloc %qc : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  scf::WhileOp loop;
  module->walk([&](scf::WhileOp candidate) { loop = candidate; });
  ASSERT_TRUE(loop);
  ASSERT_EQ(loop.getInits().size(), 2);
  EXPECT_TRUE(loop.getInits().front().getType().isF32());
  EXPECT_TRUE(isa<qco::QubitType>(loop.getInits().back().getType()));
  ASSERT_EQ(loop.getNumResults(), 2);
  EXPECT_TRUE(loop.getResult(0).getType().isInteger(64));
  EXPECT_TRUE(isa<qco::QubitType>(loop.getResult(1).getType()));

  auto condition =
      cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
  EXPECT_TRUE(
      llvm::equal(condition.getArgs().getTypes(), loop.getResultTypes()));
  auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
  EXPECT_TRUE(llvm::equal(yield.getOperandTypes(), loop.getInits().getTypes()));
  expectNoQCOperations(*module);
}

TEST_F(QCToQCORegressionTest, LeavesUnrelatedSCFTerminatorsUntouched) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i1 attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    qc.h %qc : !qc.qubit
    %result = scf.execute_region -> i1 {
      %true = arith.constant true
      scf.yield %true : i1
    }
    qc.dealloc %qc : !qc.qubit
    return %result : i1
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  bool sawExecuteRegion = false;
  module->walk([&](scf::ExecuteRegionOp) { sawExecuteRegion = true; });
  EXPECT_TRUE(sawExecuteRegion);
  expectNoQCOperations(*module);
}

TEST_F(QCToQCORegressionTest, PreservesIfClassicalResultsWithoutScratch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%condition: i1) -> i64
      attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    %result = scf.if %condition -> i64 {
      qc.h %qc : !qc.qubit
      %then = arith.constant 1 : i64
      scf.yield %then : i64
    } else {
      qc.x %qc : !qc.qubit
      %else = arith.constant 2 : i64
      scf.yield %else : i64
    }
    qc.dealloc %qc : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  qco::IfOp ifOp;
  module->walk([&](qco::IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getClassicalResults().size(), 1);
  EXPECT_TRUE(ifOp.getClassicalResults().front().getType().isInteger(64));
  ASSERT_EQ(ifOp.getLinearResults().size(), 1);
  EXPECT_TRUE(isa<qco::QubitType>(ifOp.getLinearResults().front().getType()));
  for (qco::YieldOp yield : {ifOp.thenYield(), ifOp.elseYield()}) {
    ASSERT_EQ(yield.getNumOperands(), 2);
    EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
    EXPECT_TRUE(isa<qco::QubitType>(yield.getOperand(1).getType()));
  }

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), ifOp.getClassicalResults().front());

  bool containsScratchStorage = false;
  module->walk([&](Operation* operation) {
    containsScratchStorage |=
        isa<memref::AllocaOp, memref::LoadOp, memref::StoreOp>(operation);
  });
  EXPECT_FALSE(containsScratchStorage);
  expectNoQCOperations(*module);
}

TEST_F(QCToQCORegressionTest,
       PreservesIndexSwitchClassicalResultsWithoutScratch) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%index: index) -> i64
      attributes {mqt.entry_point} {
    %qc = qc.alloc : !qc.qubit
    %result = scf.index_switch %index -> i64
    case 0 {
      qc.h %qc : !qc.qubit
      %case = arith.constant 1 : i64
      scf.yield %case : i64
    }
    default {
      qc.x %qc : !qc.qubit
      %default = arith.constant 2 : i64
      scf.yield %default : i64
    }
    qc.dealloc %qc : !qc.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  qco::IndexSwitchOp switchOp;
  module->walk([&](qco::IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getClassicalResults().size(), 1);
  EXPECT_TRUE(switchOp.getClassicalResults().front().getType().isInteger(64));
  ASSERT_EQ(switchOp.getLinearResults().size(), 1);
  EXPECT_TRUE(
      isa<qco::QubitType>(switchOp.getLinearResults().front().getType()));
  for (Region* region : switchOp.getRegions()) {
    auto yield = cast<qco::YieldOp>(region->front().getTerminator());
    ASSERT_EQ(yield.getNumOperands(), 2);
    EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
    EXPECT_TRUE(isa<qco::QubitType>(yield.getOperand(1).getType()));
  }

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), switchOp.getClassicalResults().front());

  bool containsScratchStorage = false;
  module->walk([&](Operation* operation) {
    containsScratchStorage |=
        isa<memref::AllocaOp, memref::LoadOp, memref::StoreOp>(operation);
  });
  EXPECT_FALSE(containsScratchStorage);
  expectNoQCOperations(*module);
}

TEST_F(QCToQCORegressionTest,
       MaterializesSequentialPotentialAliasesAtEachOperation) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%i: index) attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<2x!qc.qubit>
    %c0 = arith.constant 0 : index
    %q0 = memref.load %reg[%c0] : memref<2x!qc.qubit>
    qc.h %q0 : !qc.qubit
    %q1 = memref.load %reg[%c0] : memref<2x!qc.qubit>
    qc.x %q1 : !qc.qubit
    %q2 = memref.load %reg[%i] : memref<2x!qc.qubit>
    qc.y %q2 : !qc.qubit
    %q3 = memref.load %reg[%c0] : memref<2x!qc.qubit>
    %unused = qc.measure %q3 : !qc.qubit -> i1
    %q4 = memref.load %reg[%i] : memref<2x!qc.qubit>
    qc.reset %q4 : !qc.qubit
    %q5 = memref.load %reg[%c0] : memref<2x!qc.qubit>
    qc.barrier %q5 : !qc.qubit
    %q6 = memref.load %reg[%i] : memref<2x!qc.qubit>
    qc.inv (%arg0 = %q6) {
      qc.z %arg0 : !qc.qubit
      qc.yield
    } : !qc.qubit
    memref.dealloc %reg : memref<2x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  expectOperationLocalRegisterAccesses(*moduleOp);
  expectNoQCOperations(*moduleOp);

  std::size_t allocations = 0;
  std::size_t deallocations = 0;
  moduleOp->walk([&](qtensor::AllocOp) { ++allocations; });
  moduleOp->walk([&](qtensor::DeallocOp) { ++deallocations; });
  EXPECT_EQ(allocations, 1U);
  EXPECT_EQ(deallocations, 1U);
}

TEST_F(QCToQCORegressionTest, RetainsQubitRegisterName) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  std::ignore = builder.allocQubitRegisterStorage(2, "named_qubits");
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));

  qtensor::AllocOp allocation;
  moduleOp->walk([&](qtensor::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  const auto name = allocation->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

TEST_F(QCToQCORegressionTest, RetainsDynamicQubitRegisterName) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%size: index) attributes {mqt.entry_point} {
    %reg = memref.alloc(%size) {mqt.register_name = "named_qubits"} : memref<?x!qc.qubit>
    memref.dealloc %reg : memref<?x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));

  qtensor::AllocOp allocation;
  moduleOp->walk([&](qtensor::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_TRUE(allocation.getResult().getType().isDynamicDim(0));
  EXPECT_EQ(allocation.getSize(), allocation->getBlock()->getArgument(0));
  const auto name = allocation->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

TEST_F(QCToQCORegressionTest, RejectsRegisterBackedReferenceEscapes) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func private @escape(!qc.qubit)
  func.func @main() attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<1x!qc.qubit>
    %c0 = arith.constant 0 : index
    %q = memref.load %reg[%c0] : memref<1x!qc.qubit>
    func.call @escape(%q) : (!qc.qubit) -> ()
    memref.dealloc %reg : memref<1x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str())
            .contains("cannot consume a register-backed qubit reference");
    return success();
  });
  EXPECT_TRUE(failed(runQCToQCOConversion(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST_F(QCToQCORegressionTest, PreflightRejectsNonOneDimensionalQubitRegisters) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<!qc.qubit>
    %q = memref.load %reg[] : memref<!qc.qubit>
    qc.x %q : !qc.qubit
    memref.dealloc %reg : memref<!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str())
            .contains("requires one-dimensional qubit register storage");
    return success();
  });

  PassManager pm(&context);
  pm.enableVerifier(false);
  pm.addPass(createQCToQCO());
  EXPECT_TRUE(failed(pm.run(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST_F(QCToQCORegressionTest, PreflightRejectsDerivedQubitRegisterValues) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<1x!qc.qubit>
    %cast = memref.cast %reg : memref<1x!qc.qubit> to memref<?x!qc.qubit>
    memref.dealloc %cast : memref<?x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str())
            .contains("requires a directly allocated qubit register");
    return success();
  });

  PassManager pm(&context);
  pm.enableVerifier(false);
  pm.addPass(createQCToQCO());
  EXPECT_TRUE(failed(pm.run(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST_F(QCToQCORegressionTest,
       PreflightRejectsUnsupportedQuantumBlockArguments) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"mlir(
module {
  func.func @main(%q: !qc.qubit)
      attributes {mqt.entry_point} {
    qc.x %q : !qc.qubit
    return
  }
}
)mlir",
      R"mlir(
module {
  func.func @main(%reg: memref<1x!qc.qubit>)
      attributes {mqt.entry_point} {
    return
  }
}
)mlir",
      R"mlir(
module {
  func.func @main(%reg: memref<*x!qc.qubit>)
      attributes {mqt.entry_point} {
    return
  }
}
)mlir",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    bool sawExpectedDiagnostic = false;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      sawExpectedDiagnostic |=
          StringRef(diagnostic.str())
              .contains("cannot convert arbitrary qubit or qubit-register "
                        "block arguments; only QC modifier qubit arguments are "
                        "supported");
      return success();
    });

    PassManager pm(&context);
    pm.enableVerifier(false);
    pm.addPass(createQCToQCO());
    EXPECT_TRUE(failed(pm.run(*moduleOp)));
    EXPECT_TRUE(sawExpectedDiagnostic);
  }
}

TEST_F(QCToQCORegressionTest,
       PreflightRejectsUnsupportedQuantumRegionCaptures) {
  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    scf.execute_region {
      qc.x %q : !qc.qubit
      scf.yield
    }
    qc.dealloc %q : !qc.qubit
    return
  }
}
)mlir",
      R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<1x!qc.qubit>
    %c0 = arith.constant 0 : index
    %q = memref.load %reg[%c0] : memref<1x!qc.qubit>
    scf.execute_region {
      qc.x %q : !qc.qubit
      scf.yield
    }
    memref.dealloc %reg : memref<1x!qc.qubit>
    return
  }
}
)mlir",
  });

  for (const auto source : sources) {
    SCOPED_TRACE(source.str());
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    bool sawExpectedDiagnostic = false;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      sawExpectedDiagnostic |=
          StringRef(diagnostic.str())
              .contains("cannot capture quantum values in an unsupported "
                        "region-bearing operation");
      return success();
    });

    PassManager pm(&context);
    pm.enableVerifier(false);
    pm.addPass(createQCToQCO());
    EXPECT_TRUE(failed(pm.run(*moduleOp)));
    EXPECT_TRUE(sawExpectedDiagnostic);
  }
}

TEST_F(QCToQCORegressionTest, CapturesQubitsUsedByPowInsideFor) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %exponent = arith.constant 2.0 : f64
    scf.for %i = %c0 to %c1 step %c1 {
      qc.pow(%exponent) (%arg0 = %q) {
        qc.x %arg0 : !qc.qubit
        qc.yield
      } : !qc.qubit
    }
    qc.dealloc %q : !qc.qubit
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::ForOp loop;
  moduleOp->walk([&](scf::ForOp candidate) { loop = candidate; });
  ASSERT_TRUE(loop);
  ASSERT_EQ(loop.getNumResults(), 1);
  EXPECT_TRUE(isa<qco::QubitType>(loop.getResult(0).getType()));
  EXPECT_FALSE(loop.getBody()->getOps<qco::PowOp>().empty());
  expectNoQCOperations(*moduleOp);
}

namespace {

enum class ModifierKind : std::uint8_t { Inv, Ctrl, Pow };
enum class StructuredKind : std::uint8_t { For, While, If, IndexSwitch };

struct NestedModifierCase {
  std::string name;
  ModifierKind modifier;
  StructuredKind structured;
};

} // namespace

static void emitStructuredQubitUse(qc::QCProgramBuilder& builder,
                                   const StructuredKind kind, Value qubit) {
  switch (kind) {
  case StructuredKind::For:
    builder.scfFor(0, 1, 1, [&](Value) { builder.x(qubit); });
    return;
  case StructuredKind::While:
    builder.scfWhile(
        [&] {
          builder.x(qubit);
          builder.scfCondition(
              arith::ConstantOp::create(builder, builder.getBoolAttr(false)));
        },
        [&] { builder.y(qubit); });
    return;
  case StructuredKind::If:
    builder.scfIf(true, [&] { builder.x(qubit); }, [&] { builder.y(qubit); });
    return;
  case StructuredKind::IndexSwitch: {
    const auto caseBody = [&] { builder.x(qubit); };
    const auto defaultBody = [&] { builder.y(qubit); };
    const SmallVector<int64_t> cases{0};
    const SmallVector<llvm::function_ref<void()>> caseBodies{caseBody};
    builder.scfIndexSwitch(0, cases, caseBodies, defaultBody);
    return;
  }
  }
  llvm_unreachable("unknown structured operation");
}

static OwningOpRef<ModuleOp>
buildNestedModifierProgram(MLIRContext* context,
                           const NestedModifierCase& testCase,
                           const bool registerBacked) {
  qc::QCProgramBuilder builder(context);
  builder.initialize();
  Value target;
  if (registerBacked) {
    auto reg = builder.allocQubitRegisterStorage(1);
    auto index = arith::ConstantIndexOp::create(builder, 0);
    target = builder.loadQubit(reg, index.getResult());
  } else {
    target = builder.allocQubit();
  }
  const auto body = [&](Value argument) {
    emitStructuredQubitUse(builder, testCase.structured, argument);
  };

  switch (testCase.modifier) {
  case ModifierKind::Inv:
    builder.inv(target, body);
    break;
  case ModifierKind::Ctrl:
    builder.ctrl(builder.allocQubit(), target, body);
    break;
  case ModifierKind::Pow:
    builder.pow(2.0, target, body);
    break;
  }
  return builder.finalize();
}

namespace {

class NestedModifierConversionTest
    : public QCToQCORegressionTest,
      public testing::WithParamInterface<NestedModifierCase> {};

} // namespace

TEST_P(NestedModifierConversionTest, CarriesQubitThroughStructuredOperation) {
  for (const bool registerBacked : {false, true}) {
    SCOPED_TRACE(testing::Message() << "register_backed=" << registerBacked);
    auto moduleOp =
        buildNestedModifierProgram(&context, GetParam(), registerBacked);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    qco::YieldOp modifierYield;
    moduleOp->walk([&](qco::YieldOp yield) {
      if (isa<qco::InvOp, qco::CtrlOp, qco::PowOp>(yield->getParentOp())) {
        modifierYield = yield;
      }
    });
    ASSERT_TRUE(modifierYield);
    ASSERT_EQ(modifierYield.getNumOperands(), 1);

    Value structuredResult;
    switch (GetParam().structured) {
    case StructuredKind::For:
      moduleOp->walk(
          [&](scf::ForOp op) { structuredResult = op.getResults().back(); });
      break;
    case StructuredKind::While:
      moduleOp->walk(
          [&](scf::WhileOp op) { structuredResult = op.getResults().back(); });
      break;
    case StructuredKind::If:
      moduleOp->walk([&](qco::IfOp op) {
        structuredResult = op.getLinearResults().back();
      });
      break;
    case StructuredKind::IndexSwitch:
      moduleOp->walk([&](qco::IndexSwitchOp op) {
        structuredResult = op.getLinearResults().back();
      });
      break;
    }

    ASSERT_TRUE(structuredResult);
    EXPECT_EQ(modifierYield.getOperand(0), structuredResult);
    if (registerBacked) {
      expectOperationLocalRegisterAccesses(*moduleOp);
    }
    expectNoQCOperations(*moduleOp);
  }
}

static StringRef modifierName(const ModifierKind modifier) {
  switch (modifier) {
  case ModifierKind::Inv:
    return "inv";
  case ModifierKind::Ctrl:
    return "ctrl";
  case ModifierKind::Pow:
    return "pow";
  }
  llvm_unreachable("unknown modifier");
}

static OwningOpRef<ModuleOp>
buildInvalidNestedRegisterLoadProgram(MLIRContext* context,
                                      const ModifierKind modifier) {
  qc::QCProgramBuilder builder(context);
  builder.initialize();
  auto target = builder.allocQubit();
  auto reg = builder.allocQubitRegisterStorage(1);
  auto index = arith::ConstantIndexOp::create(builder, 0);
  const auto body = [&](Value) {
    builder.scfIf(true, [&] {
      auto loaded = builder.loadQubit(reg, index.getResult());
      builder.x(loaded);
    });
  };

  switch (modifier) {
  case ModifierKind::Inv:
    builder.inv(target, body);
    break;
  case ModifierKind::Ctrl:
    builder.ctrl(builder.allocQubit(), target, body);
    break;
  case ModifierKind::Pow:
    builder.pow(2.0, target, body);
    break;
  }
  return builder.finalize();
}

TEST_F(QCToQCORegressionTest,
       PreflightRejectsNestedRegisterLoadsInEveryModifier) {
  constexpr std::array modifiers{ModifierKind::Inv, ModifierKind::Ctrl,
                                 ModifierKind::Pow};

  for (const auto modifier : modifiers) {
    SCOPED_TRACE(testing::Message()
                 << "modifier=" << modifierName(modifier).str());
    auto moduleOp = buildInvalidNestedRegisterLoadProgram(&context, modifier);
    ASSERT_TRUE(moduleOp);

    bool sawExpectedDiagnostic = false;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      sawExpectedDiagnostic |=
          StringRef(diagnostic.str())
              .contains("body must not contain non-unitary operations or "
                        "access registers");
      return success();
    });

    PassManager pm(&context);
    pm.enableVerifier(false);
    pm.addPass(createQCToQCO());
    EXPECT_TRUE(failed(pm.run(*moduleOp)));
    EXPECT_TRUE(sawExpectedDiagnostic);
  }
}

namespace {
enum class CBitModifierBodyOp : std::uint8_t { Alloc, Load, Store };
} // namespace

static StringRef cbitOperationName(CBitModifierBodyOp operation) {
  switch (operation) {
  case CBitModifierBodyOp::Alloc:
    return "cbit.alloc";
  case CBitModifierBodyOp::Load:
    return "cbit.load";
  case CBitModifierBodyOp::Store:
    return "cbit.store";
  }
  llvm_unreachable("unknown CBit operation");
}

static OwningOpRef<ModuleOp>
buildInvalidCBitModifierProgram(MLIRContext* context,
                                const ModifierKind modifier,
                                CBitModifierBodyOp cbitOperation) {
  qc::QCProgramBuilder builder(context);
  builder.initialize();
  auto target = builder.allocQubit();
  auto reg = builder.allocClassicalBitRegister(1);
  auto index = arith::ConstantIndexOp::create(builder, 0);
  auto bit = builder.boolConstant(false);
  const auto body = [&](Value) {
    builder.scfIf(true, [&] {
      switch (cbitOperation) {
      case CBitModifierBodyOp::Alloc:
        cbit::AllocOp::create(builder,
                              cbit::RegisterType::get(builder.getContext(), 1),
                              cbit::Initialization::Zero);
        break;
      case CBitModifierBodyOp::Load:
        cbit::LoadOp::create(builder, builder.getI1Type(), reg,
                             index.getResult());
        break;
      case CBitModifierBodyOp::Store:
        cbit::StoreOp::create(builder, bit, reg, index.getResult());
        break;
      }
    });
  };

  switch (modifier) {
  case ModifierKind::Inv:
    builder.inv(target, body);
    break;
  case ModifierKind::Ctrl:
    builder.ctrl(builder.allocQubit(), target, body);
    break;
  case ModifierKind::Pow:
    builder.pow(2.0, target, body);
    break;
  }
  return builder.finalize();
}

TEST_F(QCToQCORegressionTest,
       PreflightRejectsEveryCBitOperationInEveryModifier) {
  constexpr std::array modifiers{ModifierKind::Inv, ModifierKind::Ctrl,
                                 ModifierKind::Pow};
  constexpr std::array operations{CBitModifierBodyOp::Alloc,
                                  CBitModifierBodyOp::Load,
                                  CBitModifierBodyOp::Store};

  for (const auto modifier : modifiers) {
    for (const auto operation : operations) {
      SCOPED_TRACE(testing::Message()
                   << "modifier=" << modifierName(modifier).str()
                   << ", operation=" << cbitOperationName(operation).str());
      auto moduleOp =
          buildInvalidCBitModifierProgram(&context, modifier, operation);
      ASSERT_TRUE(moduleOp);

      bool sawExpectedDiagnostic = false;
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
        sawExpectedDiagnostic |=
            StringRef(diagnostic.str())
                .contains("body must not contain non-unitary operations or "
                          "access registers");
        return success();
      });

      PassManager pm(&context);
      pm.enableVerifier(false);
      pm.addPass(createQCToQCO());
      EXPECT_TRUE(failed(pm.run(*moduleOp)));
      EXPECT_TRUE(sawExpectedDiagnostic);
    }
  }
}

static OwningOpRef<ModuleOp> buildInvalidModifierCaptureProgram(
    MLIRContext* context, const ModifierKind modifier,
    const bool registerBacked, const bool nested) {
  qc::QCProgramBuilder builder(context);
  builder.initialize();

  Value target;
  Value captured;
  if (registerBacked) {
    auto reg = builder.allocQubitRegisterStorage(2);
    auto targetIndex = arith::ConstantIndexOp::create(builder, 0);
    auto capturedIndex = arith::ConstantIndexOp::create(builder, 1);
    target = builder.loadQubit(reg, targetIndex.getResult());
    captured = builder.loadQubit(reg, capturedIndex.getResult());
  } else {
    target = builder.allocQubit();
    captured = builder.allocQubit();
  }

  const auto modifierBody = [&](Value) {
    if (nested) {
      builder.scfIf(true, [&] { builder.x(captured); });
      return;
    }
    builder.x(captured);
  };
  switch (modifier) {
  case ModifierKind::Inv:
    builder.inv(target, modifierBody);
    break;
  case ModifierKind::Ctrl:
    builder.ctrl(builder.allocQubit(), target, modifierBody);
    break;
  case ModifierKind::Pow:
    builder.pow(2.0, target, modifierBody);
    break;
  }
  return builder.finalize();
}

TEST_F(QCToQCORegressionTest,
       PreflightRejectsEveryUnsupportedModifierQubitCapture) {
  constexpr std::array modifiers{ModifierKind::Inv, ModifierKind::Ctrl,
                                 ModifierKind::Pow};

  for (const auto modifier : modifiers) {
    for (const bool registerBacked : {false, true}) {
      for (const bool nested : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "modifier=" << modifierName(modifier).str()
                     << ", register_backed=" << registerBacked
                     << ", nested=" << nested);
        auto moduleOp = buildInvalidModifierCaptureProgram(
            &context, modifier, registerBacked, nested);
        ASSERT_TRUE(moduleOp);

        bool sawExpectedDiagnostic = false;
        ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
          sawExpectedDiagnostic |=
              StringRef(diagnostic.str())
                  .contains("body must not capture qubits from above; use only "
                            "its aliased block arguments");
          return success();
        });

        PassManager pm(&context);
        pm.enableVerifier(false);
        pm.addPass(createQCToQCO());
        EXPECT_TRUE(failed(pm.run(*moduleOp)));
        EXPECT_TRUE(sawExpectedDiagnostic);
      }
    }
  }
}

static OwningOpRef<ModuleOp>
buildClassicalCaptureProgram(MLIRContext* context,
                             const ModifierKind modifier) {
  qc::QCProgramBuilder builder(context);
  builder.initialize();
  auto target = builder.allocQubit();
  auto theta = arith::ConstantOp::create(builder, builder.getF64FloatAttr(0.75))
                   .getResult();
  const auto modifierBody = [&](Value argument) {
    builder.rx(theta, argument);
  };

  switch (modifier) {
  case ModifierKind::Inv:
    builder.inv(target, modifierBody);
    break;
  case ModifierKind::Ctrl:
    builder.ctrl(builder.allocQubit(), target, modifierBody);
    break;
  case ModifierKind::Pow:
    builder.pow(2.0, target, modifierBody);
    break;
  }
  return builder.finalize();
}

TEST_F(QCToQCORegressionTest, ModifiersPermitClassicalCaptures) {
  constexpr std::array modifiers{ModifierKind::Inv, ModifierKind::Ctrl,
                                 ModifierKind::Pow};

  for (const auto modifier : modifiers) {
    SCOPED_TRACE(testing::Message()
                 << "modifier=" << modifierName(modifier).str());
    auto moduleOp = buildClassicalCaptureProgram(&context, modifier);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    expectNoQCOperations(*moduleOp);
  }
}

TEST_F(QCToQCORegressionTest,
       NestedModifiersCarryTheStructuredOperationResultByRegion) {
  qc::QCProgramBuilder builder(&context);
  builder.initialize();
  auto target = builder.allocQubit();
  builder.inv(target, [&](Value outerArgument) {
    builder.pow(2.0, outerArgument, [&](Value innerArgument) {
      builder.scfFor(0, 1, 1, [&](Value) { builder.x(innerArgument); });
    });
  });

  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  qco::InvOp inv;
  qco::PowOp pow;
  scf::ForOp loop;
  moduleOp->walk([&](qco::InvOp op) { inv = op; });
  moduleOp->walk([&](qco::PowOp op) { pow = op; });
  moduleOp->walk([&](scf::ForOp op) { loop = op; });
  ASSERT_TRUE(inv);
  ASSERT_TRUE(pow);
  ASSERT_TRUE(loop);

  auto invYield = cast<qco::YieldOp>(inv.getBody()->getTerminator());
  auto powYield = cast<qco::YieldOp>(pow.getBody()->getTerminator());
  ASSERT_EQ(invYield.getNumOperands(), 1);
  ASSERT_EQ(powYield.getNumOperands(), 1);
  EXPECT_EQ(invYield.getOperand(0), pow.getQubitsOut().front());
  EXPECT_EQ(powYield.getOperand(0), loop.getResults().back());
  expectNoQCOperations(*moduleOp);
}

INSTANTIATE_TEST_SUITE_P(
    ModifierStructuredMatrix, NestedModifierConversionTest,
    testing::Values(
        NestedModifierCase{"InvFor", ModifierKind::Inv, StructuredKind::For},
        NestedModifierCase{"InvWhile", ModifierKind::Inv,
                           StructuredKind::While},
        NestedModifierCase{"InvIf", ModifierKind::Inv, StructuredKind::If},
        NestedModifierCase{"InvIndexSwitch", ModifierKind::Inv,
                           StructuredKind::IndexSwitch},
        NestedModifierCase{"CtrlFor", ModifierKind::Ctrl, StructuredKind::For},
        NestedModifierCase{"CtrlWhile", ModifierKind::Ctrl,
                           StructuredKind::While},
        NestedModifierCase{"CtrlIf", ModifierKind::Ctrl, StructuredKind::If},
        NestedModifierCase{"CtrlIndexSwitch", ModifierKind::Ctrl,
                           StructuredKind::IndexSwitch},
        NestedModifierCase{"PowFor", ModifierKind::Pow, StructuredKind::For},
        NestedModifierCase{"PowWhile", ModifierKind::Pow,
                           StructuredKind::While},
        NestedModifierCase{"PowIf", ModifierKind::Pow, StructuredKind::If},
        NestedModifierCase{"PowIndexSwitch", ModifierKind::Pow,
                           StructuredKind::IndexSwitch}),
    [](const testing::TestParamInfo<NestedModifierCase>& info) {
      return info.param.name;
    });

TEST_F(QCToQCORegressionTest, DoesNotCaptureQubitsAllocatedInsideIf) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%condition: i1)
      attributes {mqt.entry_point} {
    scf.if %condition {
      %q = qc.alloc : !qc.qubit
      qc.h %q : !qc.qubit
      qc.dealloc %q : !qc.qubit
    }
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCToQCOConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::IfOp ifOp;
  moduleOp->walk([&](scf::IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  EXPECT_EQ(ifOp.getNumResults(), 0);
  std::size_t allocations = 0;
  ifOp.getThenRegion().walk([&](qco::AllocOp) { ++allocations; });
  EXPECT_EQ(allocations, 1);
  expectNoQCOperations(*moduleOp);
}

TEST_F(QCToQCORegressionTest,
       RejectsSameDynamicRegisterIndexWithinOneOperation) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%i: index) attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<2x!qc.qubit>
    %q0 = memref.load %reg[%i] : memref<2x!qc.qubit>
    %q1 = memref.load %reg[%i] : memref<2x!qc.qubit>
    qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
    memref.dealloc %reg : memref<2x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("use the same dynamic index");
    return success();
  });
  EXPECT_TRUE(failed(runQCToQCOConversion(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST_F(QCToQCORegressionTest,
       RejectsEqualConstantRegisterIndicesWithinOneOperation) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %reg = memref.alloc() : memref<2x!qc.qubit>
    %lhs = arith.constant 0 : index
    %rhs = arith.constant 0 : index
    %q0 = memref.load %reg[%lhs] : memref<2x!qc.qubit>
    %q1 = memref.load %reg[%rhs] : memref<2x!qc.qubit>
    qc.swap %q0, %q1 : !qc.qubit, !qc.qubit
    memref.dealloc %reg : memref<2x!qc.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("same constant index");
    return success();
  });
  EXPECT_TRUE(failed(runQCToQCOConversion(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST_P(QCToQCOTest, ProgramConversion) {
  const auto& [_, programBuilder, referenceBuilder, expectsCompleteTensorState,
               skipReferenceComparison] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCToQCOConversion(program.get())));
  printer.record(program.get(), "Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());
  if (expectsCompleteTensorState) {
    QCToQCORegressionTest::expectOperationLocalRegisterAccesses(program.get());
    QCToQCORegressionTest::expectStructuredStateUsesCompleteTensors(
        program.get());
  }

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  if (!skipReferenceComparison) {
    auto reference =
        ::mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
    ASSERT_TRUE(reference);
    printer.record(reference.get(), "Reference QCO IR" + name);
    EXPECT_TRUE(verify(*reference).succeeded());

    EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
    printer.record(reference.get(), "Canonicalized Reference QCO IR" + name);
    EXPECT_TRUE(verify(*reference).succeeded());

    EXPECT_TRUE(
        areModulesEquivalentWithPermutations(program.get(), reference.get()));
  }
}

/// \name QCToQCO/QubitManagement/StaticOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCStaticOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"StaticQubits", MQT_NAMED_BUILDER(qc::staticQubits),
                        MQT_NAMED_BUILDER(qco::staticQubits)},
        QCToQCOTestCase{"StaticQubitsWithOps",
                        MQT_NAMED_BUILDER(qc::staticQubitsWithOps),
                        MQT_NAMED_BUILDER(qco::staticQubitsWithOps)},
        QCToQCOTestCase{"StaticQubitsWithParametricOps",
                        MQT_NAMED_BUILDER(qc::staticQubitsWithParametricOps),
                        MQT_NAMED_BUILDER(qco::staticQubitsWithParametricOps)},
        QCToQCOTestCase{"StaticQubitsWithTwoTargetOps",
                        MQT_NAMED_BUILDER(qc::staticQubitsWithTwoTargetOps),
                        MQT_NAMED_BUILDER(qco::staticQubitsWithTwoTargetOps)},
        QCToQCOTestCase{"StaticQubitsWithCtrl",
                        MQT_NAMED_BUILDER(qc::staticQubitsWithCtrl),
                        MQT_NAMED_BUILDER(qco::staticQubitsWithCtrl)},
        QCToQCOTestCase{"StaticQubitsWithInv",
                        MQT_NAMED_BUILDER(qc::staticQubitsWithInv),
                        MQT_NAMED_BUILDER(qco::staticQubitsWithInv)},
        QCToQCOTestCase{"AllocDeallocPair",
                        MQT_NAMED_BUILDER(qc::allocDeallocPair),
                        MQT_NAMED_BUILDER(qco::emptyQCO)}));
/// @}

/// \name QCToQCO/Modifiers/PowOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCPowOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"CtrlPowSx",
                                    MQT_NAMED_BUILDER(qc::ctrlPowSx),
                                    MQT_NAMED_BUILDER(qco::ctrlPowSx)},
                    QCToQCOTestCase{"PowTwo", MQT_NAMED_BUILDER(qc::powTwo),
                                    MQT_NAMED_BUILDER(qco::powTwo)}));
/// @}

/// \name QCToQCO/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCCtrlOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"CtrlTwo", MQT_NAMED_BUILDER(qc::ctrlTwo),
                                    MQT_NAMED_BUILDER(qco::ctrlTwo)},
                    QCToQCOTestCase{"CtrlTwoMixed",
                                    MQT_NAMED_BUILDER(qc::ctrlTwoMixed),
                                    MQT_NAMED_BUILDER(qco::ctrlTwoMixed)},
                    QCToQCOTestCase{"CtrlInvTwo",
                                    MQT_NAMED_BUILDER(qc::ctrlInvTwo),
                                    MQT_NAMED_BUILDER(qco::ctrlInvTwo)}));
/// @}

/// \name QCToQCO/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCInvOpTest, QCToQCOTest,
    testing::Values(
        // iSWAP cannot be inverted with current canonicalization
        QCToQCOTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(qc::inverseIswap),
                        MQT_NAMED_BUILDER(qco::inverseIswap)},
        QCToQCOTestCase{"InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap)},
        QCToQCOTestCase{"InvTwo", MQT_NAMED_BUILDER(qc::invTwo),
                        MQT_NAMED_BUILDER(qco::invTwo)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCBarrierOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Barrier", MQT_NAMED_BUILDER(qc::barrier),
                                    MQT_NAMED_BUILDER(qco::barrier)},
                    QCToQCOTestCase{"BarrierTwoQubits",
                                    MQT_NAMED_BUILDER(qc::barrierTwoQubits),
                                    MQT_NAMED_BUILDER(qco::barrierTwoQubits)},
                    QCToQCOTestCase{
                        "BarrierMultipleQubits",
                        MQT_NAMED_BUILDER(qc::barrierMultipleQubits),
                        MQT_NAMED_BUILDER(qco::barrierMultipleQubits)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCDCXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"DCX", MQT_NAMED_BUILDER(qc::dcx),
                        MQT_NAMED_BUILDER(qco::dcx)},
        QCToQCOTestCase{"SingleControlledDCX",
                        MQT_NAMED_BUILDER(qc::singleControlledDcx),
                        MQT_NAMED_BUILDER(qco::singleControlledDcx)},
        QCToQCOTestCase{"MultipleControlledDCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx),
                        MQT_NAMED_BUILDER(qco::multipleControlledDcx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCECROpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"ECR", MQT_NAMED_BUILDER(qc::ecr),
                        MQT_NAMED_BUILDER(qco::ecr)},
        QCToQCOTestCase{"SingleControlledECR",
                        MQT_NAMED_BUILDER(qc::singleControlledEcr),
                        MQT_NAMED_BUILDER(qco::singleControlledEcr)},
        QCToQCOTestCase{"MultipleControlledECR",
                        MQT_NAMED_BUILDER(qc::multipleControlledEcr),
                        MQT_NAMED_BUILDER(qco::multipleControlledEcr)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCGPhaseOpTest, QCToQCOTest,
                         testing::Values(QCToQCOTestCase{
                             "GlobalPhase", MQT_NAMED_BUILDER(qc::globalPhase),
                             MQT_NAMED_BUILDER(qco::globalPhase)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCHOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"H", MQT_NAMED_BUILDER(qc::h),
                                    MQT_NAMED_BUILDER(qco::h)},
                    QCToQCOTestCase{"SingleControlledH",
                                    MQT_NAMED_BUILDER(qc::singleControlledH),
                                    MQT_NAMED_BUILDER(qco::singleControlledH)},
                    QCToQCOTestCase{
                        "MultipleControlledH",
                        MQT_NAMED_BUILDER(qc::multipleControlledH),
                        MQT_NAMED_BUILDER(qco::multipleControlledH)},
                    QCToQCOTestCase{"HWithoutRegister",
                                    MQT_NAMED_BUILDER(qc::hWithoutRegister),
                                    MQT_NAMED_BUILDER(qco::hWithoutRegister)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCIDOpTest, QCToQCOTest,
                         testing::Values(QCToQCOTestCase{
                             "Identity", MQT_NAMED_BUILDER(qc::identity),
                             MQT_NAMED_BUILDER(qco::alloc1QubitRegister)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCiSWAPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"iSWAP", MQT_NAMED_BUILDER(qc::iswap),
                        MQT_NAMED_BUILDER(qco::iswap)},
        QCToQCOTestCase{"SingleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledIswap),
                        MQT_NAMED_BUILDER(qco::singleControlledIswap)},
        QCToQCOTestCase{"MultipleControllediSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledIswap),
                        MQT_NAMED_BUILDER(qco::multipleControlledIswap)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCPOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"P", MQT_NAMED_BUILDER(qc::p),
                                    MQT_NAMED_BUILDER(qco::p)},
                    QCToQCOTestCase{"SingleControlledP",
                                    MQT_NAMED_BUILDER(qc::singleControlledP),
                                    MQT_NAMED_BUILDER(qco::singleControlledP)},
                    QCToQCOTestCase{
                        "MultipleControlledP",
                        MQT_NAMED_BUILDER(qc::multipleControlledP),
                        MQT_NAMED_BUILDER(qco::multipleControlledP)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRCCXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RCCX", MQT_NAMED_BUILDER(qc::rccx),
                        MQT_NAMED_BUILDER(qco::rccx)},
        QCToQCOTestCase{"SingleControlledRCCX",
                        MQT_NAMED_BUILDER(qc::singleControlledRccx),
                        MQT_NAMED_BUILDER(qco::singleControlledRccx)},
        QCToQCOTestCase{"MultipleControlledRCCX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRccx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRccx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCROpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"R", MQT_NAMED_BUILDER(qc::r),
                                    MQT_NAMED_BUILDER(qco::r)},
                    QCToQCOTestCase{"SingleControlledR",
                                    MQT_NAMED_BUILDER(qc::singleControlledR),
                                    MQT_NAMED_BUILDER(qco::singleControlledR)},
                    QCToQCOTestCase{
                        "MultipleControlledR",
                        MQT_NAMED_BUILDER(qc::multipleControlledR),
                        MQT_NAMED_BUILDER(qco::multipleControlledR)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RX", MQT_NAMED_BUILDER(qc::rx),
                                    MQT_NAMED_BUILDER(qco::rx)},
                    QCToQCOTestCase{"SingleControlledRX",
                                    MQT_NAMED_BUILDER(qc::singleControlledRx),
                                    MQT_NAMED_BUILDER(qco::singleControlledRx)},
                    QCToQCOTestCase{
                        "MultipleControlledRX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRXXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RXX", MQT_NAMED_BUILDER(qc::rxx),
                        MQT_NAMED_BUILDER(qco::rxx)},
        QCToQCOTestCase{"SingleControlledRXX",
                        MQT_NAMED_BUILDER(qc::singleControlledRxx),
                        MQT_NAMED_BUILDER(qco::singleControlledRxx)},
        QCToQCOTestCase{"MultipleControlledRXX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRxx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRxx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RY", MQT_NAMED_BUILDER(qc::ry),
                                    MQT_NAMED_BUILDER(qco::ry)},
                    QCToQCOTestCase{"SingleControlledRY",
                                    MQT_NAMED_BUILDER(qc::singleControlledRy),
                                    MQT_NAMED_BUILDER(qco::singleControlledRy)},
                    QCToQCOTestCase{
                        "MultipleControlledRY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRy),
                        MQT_NAMED_BUILDER(qco::multipleControlledRy)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RYY", MQT_NAMED_BUILDER(qc::ryy),
                        MQT_NAMED_BUILDER(qco::ryy)},
        QCToQCOTestCase{"SingleControlledRYY",
                        MQT_NAMED_BUILDER(qc::singleControlledRyy),
                        MQT_NAMED_BUILDER(qco::singleControlledRyy)},
        QCToQCOTestCase{"MultipleControlledRYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledRyy),
                        MQT_NAMED_BUILDER(qco::multipleControlledRyy)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"RZ", MQT_NAMED_BUILDER(qc::rz),
                                    MQT_NAMED_BUILDER(qco::rz)},
                    QCToQCOTestCase{"SingleControlledRZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledRz),
                                    MQT_NAMED_BUILDER(qco::singleControlledRz)},
                    QCToQCOTestCase{
                        "MultipleControlledRZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRz),
                        MQT_NAMED_BUILDER(qco::multipleControlledRz)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZX", MQT_NAMED_BUILDER(qc::rzx),
                        MQT_NAMED_BUILDER(qco::rzx)},
        QCToQCOTestCase{"SingleControlledRZX",
                        MQT_NAMED_BUILDER(qc::singleControlledRzx),
                        MQT_NAMED_BUILDER(qco::singleControlledRzx)},
        QCToQCOTestCase{"MultipleControlledRZX",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzx),
                        MQT_NAMED_BUILDER(qco::multipleControlledRzx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCRZZOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"RZZ", MQT_NAMED_BUILDER(qc::rzz),
                        MQT_NAMED_BUILDER(qco::rzz)},
        QCToQCOTestCase{"SingleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::singleControlledRzz),
                        MQT_NAMED_BUILDER(qco::singleControlledRzz)},
        QCToQCOTestCase{"MultipleControlledRZZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledRzz),
                        MQT_NAMED_BUILDER(qco::multipleControlledRzz)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"S", MQT_NAMED_BUILDER(qc::s),
                                    MQT_NAMED_BUILDER(qco::s)},
                    QCToQCOTestCase{"SingleControlledS",
                                    MQT_NAMED_BUILDER(qc::singleControlledS),
                                    MQT_NAMED_BUILDER(qco::singleControlledS)},
                    QCToQCOTestCase{
                        "MultipleControlledS",
                        MQT_NAMED_BUILDER(qc::multipleControlledS),
                        MQT_NAMED_BUILDER(qco::multipleControlledS)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Sdg", MQT_NAMED_BUILDER(qc::sdg),
                        MQT_NAMED_BUILDER(qco::sdg)},
        QCToQCOTestCase{"SingleControlledSdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSdg),
                        MQT_NAMED_BUILDER(qco::singleControlledSdg)},
        QCToQCOTestCase{"MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledSdg)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSWAPOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SWAP", MQT_NAMED_BUILDER(qc::swap),
                        MQT_NAMED_BUILDER(qco::swap)},
        QCToQCOTestCase{"SingleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::singleControlledSwap),
                        MQT_NAMED_BUILDER(qco::singleControlledSwap)},
        QCToQCOTestCase{"MultipleControlledSWAP",
                        MQT_NAMED_BUILDER(qc::multipleControlledSwap),
                        MQT_NAMED_BUILDER(qco::multipleControlledSwap)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSXOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"SX", MQT_NAMED_BUILDER(qc::sx),
                                    MQT_NAMED_BUILDER(qco::sx)},
                    QCToQCOTestCase{"SingleControlledSX",
                                    MQT_NAMED_BUILDER(qc::singleControlledSx),
                                    MQT_NAMED_BUILDER(qco::singleControlledSx)},
                    QCToQCOTestCase{
                        "MultipleControlledSX",
                        MQT_NAMED_BUILDER(qc::multipleControlledSx),
                        MQT_NAMED_BUILDER(qco::multipleControlledSx)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCSXdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                        MQT_NAMED_BUILDER(qco::sxdg)},
        QCToQCOTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qco::singleControlledSxdg)},
        QCToQCOTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledSxdg)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"T", MQT_NAMED_BUILDER(qc::t_),
                                    MQT_NAMED_BUILDER(qco::t_)},
                    QCToQCOTestCase{"SingleControlledT",
                                    MQT_NAMED_BUILDER(qc::singleControlledT),
                                    MQT_NAMED_BUILDER(qco::singleControlledT)},
                    QCToQCOTestCase{
                        "MultipleControlledT",
                        MQT_NAMED_BUILDER(qc::multipleControlledT),
                        MQT_NAMED_BUILDER(qco::multipleControlledT)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCTdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"Tdg", MQT_NAMED_BUILDER(qc::tdg),
                        MQT_NAMED_BUILDER(qco::tdg)},
        QCToQCOTestCase{"SingleControlledTdg",
                        MQT_NAMED_BUILDER(qc::singleControlledTdg),
                        MQT_NAMED_BUILDER(qco::singleControlledTdg)},
        QCToQCOTestCase{"MultipleControlledTdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledTdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledTdg)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCU2OpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U2", MQT_NAMED_BUILDER(qc::u2),
                                    MQT_NAMED_BUILDER(qco::u2)},
                    QCToQCOTestCase{"SingleControlledU2",
                                    MQT_NAMED_BUILDER(qc::singleControlledU2),
                                    MQT_NAMED_BUILDER(qco::singleControlledU2)},
                    QCToQCOTestCase{
                        "MultipleControlledU2",
                        MQT_NAMED_BUILDER(qc::multipleControlledU2),
                        MQT_NAMED_BUILDER(qco::multipleControlledU2)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCUOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"U", MQT_NAMED_BUILDER(qc::u),
                                    MQT_NAMED_BUILDER(qco::u)},
                    QCToQCOTestCase{"SingleControlledU",
                                    MQT_NAMED_BUILDER(qc::singleControlledU),
                                    MQT_NAMED_BUILDER(qco::singleControlledU)},
                    QCToQCOTestCase{
                        "MultipleControlledU",
                        MQT_NAMED_BUILDER(qc::multipleControlledU),
                        MQT_NAMED_BUILDER(qco::multipleControlledU)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"X", MQT_NAMED_BUILDER(qc::x),
                        MQT_NAMED_BUILDER(qco::x)},
        QCToQCOTestCase{"SingleControlledX",
                        MQT_NAMED_BUILDER(qc::singleControlledX),
                        MQT_NAMED_BUILDER(qco::singleControlledX)},
        QCToQCOTestCase{"MultipleControlledX",
                        MQT_NAMED_BUILDER(qc::multipleControlledX),
                        MQT_NAMED_BUILDER(qco::multipleControlledX)},
        QCToQCOTestCase{"RepeatedControlledX",
                        MQT_NAMED_BUILDER(qc::repeatedControlledX),
                        MQT_NAMED_BUILDER(qco::repeatedControlledX)}));

/// @}

/// \name QCToQCO/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXXMinusYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"XXMinusYY", MQT_NAMED_BUILDER(qc::xxMinusYY),
                        MQT_NAMED_BUILDER(qco::xxMinusYY)},
        QCToQCOTestCase{"SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY)},
        QCToQCOTestCase{"MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCXXPlusYYOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qc::xxPlusYY),
                        MQT_NAMED_BUILDER(qco::xxPlusYY)},
        QCToQCOTestCase{"SingleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::singleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY)},
        QCToQCOTestCase{"MultipleControlledXXPlusYY",
                        MQT_NAMED_BUILDER(qc::multipleControlledXxPlusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCYOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Y", MQT_NAMED_BUILDER(qc::y),
                                    MQT_NAMED_BUILDER(qco::y)},
                    QCToQCOTestCase{"SingleControlledY",
                                    MQT_NAMED_BUILDER(qc::singleControlledY),
                                    MQT_NAMED_BUILDER(qco::singleControlledY)},
                    QCToQCOTestCase{
                        "MultipleControlledY",
                        MQT_NAMED_BUILDER(qc::multipleControlledY),
                        MQT_NAMED_BUILDER(qco::multipleControlledY)}));
/// @}

/// \name QCToQCO/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCZOpTest, QCToQCOTest,
    testing::Values(QCToQCOTestCase{"Z", MQT_NAMED_BUILDER(qc::z),
                                    MQT_NAMED_BUILDER(qco::z)},
                    QCToQCOTestCase{"SingleControlledZ",
                                    MQT_NAMED_BUILDER(qc::singleControlledZ),
                                    MQT_NAMED_BUILDER(qco::singleControlledZ)},
                    QCToQCOTestCase{
                        "MultipleControlledZ",
                        MQT_NAMED_BUILDER(qc::multipleControlledZ),
                        MQT_NAMED_BUILDER(qco::multipleControlledZ)}));
/// @}

/// \name QCToQCO/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCMeasureOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SingleMeasurementToSingleBit",
                        MQT_NAMED_BUILDER(qc::singleMeasurementToSingleBit),
                        MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit)},
        QCToQCOTestCase{"RepeatedMeasurementToSameBit",
                        MQT_NAMED_BUILDER(qc::repeatedMeasurementToSameBit),
                        MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit)},
        QCToQCOTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qc::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits)},
        QCToQCOTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qc::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements)},
        QCToQCOTestCase{"PartialMeasurementToRegister",
                        MQT_NAMED_BUILDER(qc::partialMeasurementToRegister),
                        MQT_NAMED_BUILDER(qco::partialMeasurementToRegister)},
        QCToQCOTestCase{"DynamicallyIndexedMeasurement",
                        MQT_NAMED_BUILDER(qc::dynamicallyIndexedMeasurement),
                        MQT_NAMED_BUILDER(qco::dynamicallyIndexedMeasurement),
                        true},
        QCToQCOTestCase{"MeasurementWithoutRegisters",
                        MQT_NAMED_BUILDER(qc::measurementWithoutRegisters),
                        MQT_NAMED_BUILDER(qco::measurementWithoutRegisters)}));
/// @}

/// \name QCToQCO/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCResetOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"ResetQubitAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp),
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)},
        QCToQCOTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qc::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp)},
        QCToQCOTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qc::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)}));
/// @}

/// \name QCToQCO/Operations/IfOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFIfOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SimpleIfOp", MQT_NAMED_BUILDER(qc::simpleIf),
                        MQT_NAMED_BUILDER(qco::simpleIfCompleteTensorState),
                        true},
        QCToQCOTestCase{"IfElse", MQT_NAMED_BUILDER(qc::ifElse),
                        MQT_NAMED_BUILDER(qco::ifElseCompleteTensorState),
                        true},
        QCToQCOTestCase{"IfTwoQubits", MQT_NAMED_BUILDER(qc::ifTwoQubits),
                        MQT_NAMED_BUILDER(qco::ifTwoQubitsCompleteTensorState),
                        true},
        QCToQCOTestCase{
            "IfWithMeasurement", MQT_NAMED_BUILDER(qc::ifWithMeasurement),
            MQT_NAMED_BUILDER(qco::ifWithMeasurementCompleteTensorState), true},
        QCToQCOTestCase{"IfWithCreg", MQT_NAMED_BUILDER(qc::ifWithCreg),
                        MQT_NAMED_BUILDER(qco::ifWithCregCompleteTensorState),
                        true},
        QCToQCOTestCase{"NestedIfOpForLoop",
                        MQT_NAMED_BUILDER(qc::nestedIfOpForLoop),
                        MQT_NAMED_BUILDER(qco::nestedIfOpForLoop), true}));
/// @}

/// \name QCToQCO/Operations/IndexSwitchOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIndexSwitchOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{
            "SimpleIndexSwitchOp", MQT_NAMED_BUILDER(qc::simpleIndexSwitch),
            MQT_NAMED_BUILDER(qco::simpleIndexSwitchCompleteTensorState), true},
        QCToQCOTestCase{
            "IndexSwitchMultiCase", MQT_NAMED_BUILDER(qc::indexSwitchMultiCase),
            MQT_NAMED_BUILDER(qco::indexSwitchMultiCaseCompleteTensorState),
            true}));
/// @}

/// \name QCToQCO/Operations/WhileOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFWhileOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SimpleWhile", MQT_NAMED_BUILDER(qc::simpleWhileReset),
                        MQT_NAMED_BUILDER(qco::simpleWhileReset)},
        QCToQCOTestCase{"SimpleDoWhile",
                        MQT_NAMED_BUILDER(qc::simpleDoWhileReset),
                        MQT_NAMED_BUILDER(qco::simpleDoWhileReset)}));
/// @}

/// \name QCToQCO/Operations/ForOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFForOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SimpleForLoop", MQT_NAMED_BUILDER(qc::simpleForLoop),
                        MQT_NAMED_BUILDER(qco::simpleForLoop), true},
        QCToQCOTestCase{"NestedForLoopIfOp",
                        MQT_NAMED_BUILDER(qc::nestedForLoopIfOp),
                        MQT_NAMED_BUILDER(qco::nestedForLoopIfOp), true},
        QCToQCOTestCase{
            "NestedForLoopWhileOp", MQT_NAMED_BUILDER(qc::nestedForLoopWhileOp),
            MQT_NAMED_BUILDER(qco::nestedForLoopWhileOpCompleteTensorState),
            true},
        QCToQCOTestCase{
            "NestedForLoopSwitchOp",
            MQT_NAMED_BUILDER(qc::nestedForLoopSwitchOp),
            MQT_NAMED_BUILDER(qco::nestedForLoopSwitchOpCompleteTensorState),
            true},
        QCToQCOTestCase{
            "NestedForLoopCtrlOpWithSeparateQubit",
            MQT_NAMED_BUILDER(qc::nestedForLoopCtrlOpWithSeparateQubit),
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithSeparateQubit), true},
        QCToQCOTestCase{
            "NestedForLoopCtrlOpWithExtractedQubit",
            MQT_NAMED_BUILDER(qc::nestedForLoopCtrlOpWithExtractedQubit),
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithExtractedQubit),
            true}));
/// @}
