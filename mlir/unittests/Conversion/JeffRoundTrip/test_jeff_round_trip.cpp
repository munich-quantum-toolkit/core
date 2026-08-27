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
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/IR/JeffOps.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

using namespace mlir;

namespace {

struct JeffRoundTripTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<qco::QCOProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<qco::QCOProgramBuilder> referenceBuilder;
  bool expectPowAfterCleanup = false;

  friend std::ostream& operator<<(std::ostream& os,
                                  const JeffRoundTripTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const JeffRoundTripTestCase& info) {
  return os << "JeffRoundTrip{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class JeffRoundTripTest : public testing::TestWithParam<JeffRoundTripTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<mlir::mqt::MQTDialect, arith::ArithDialect,
                    cbit::CBitDialect, func::FuncDialect, jeff::JeffDialect,
                    memref::MemRefDialect, qco::QCODialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static Value measureToRegister(qco::QCOProgramBuilder& b, ValueRange qubits) {
  auto c = b.allocClassicalBitRegister(static_cast<int64_t>(qubits.size()));
  for (auto [i, q] : llvm::enumerate(qubits)) {
    b.measure(q, c, static_cast<int64_t>(i));
  }
  return c;
}

// DCX and U gates with dynamic parameters survive `FoldPowIntoGate`, so they
// exercise power modifiers that reach the jeff conversion.

static Value powDcx(qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto powOut = b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    auto [q0, q1] = b.dcx(qubits[0], qubits[1]);
    return SmallVector{q0, q1};
  });
  return measureToRegister(b, powOut);
}

static Value powInvDcx(qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto powOut = b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) {
    auto inner = b.inv({qubits[0], qubits[1]}, [&](ValueRange invArgs) {
      auto [q0, q1] = b.dcx(invArgs[0], invArgs[1]);
      return SmallVector{q0, q1};
    });
    return llvm::to_vector(inner);
  });
  return measureToRegister(b, powOut);
}

static Value ctrlPowDcx(qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [controlsOut, targetsOut] =
      b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
        auto inner =
            b.pow(2.0, {targets[0], targets[1]}, [&](ValueRange powArgs) {
              auto [q0, q1] = b.dcx(powArgs[0], powArgs[1]);
              return SmallVector{q0, q1};
            });
        return llvm::to_vector(inner);
      });
  return measureToRegister(
      b, {controlsOut[0], controlsOut[1], targetsOut[0], targetsOut[1]});
}

static Value ctrlPowInvDcx(qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(4);
  auto [controlsOut, targetsOut] =
      b.ctrl({q[0], q[1]}, {q[2], q[3]}, [&](ValueRange targets) {
        auto inner =
            b.pow(2.0, {targets[0], targets[1]}, [&](ValueRange powArgs) {
              auto invOut =
                  b.inv({powArgs[0], powArgs[1]}, [&](ValueRange invArgs) {
                    auto [q0, q1] = b.dcx(invArgs[0], invArgs[1]);
                    return SmallVector{q0, q1};
                  });
              return llvm::to_vector(invOut);
            });
        return llvm::to_vector(inner);
      });
  return measureToRegister(
      b, {controlsOut[0], controlsOut[1], targetsOut[0], targetsOut[1]});
}

static Value powU(qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(1);
  auto powOut =
      b.pow(3.0, q[0], [&](Value qubit) { return b.u(0.1, 0.2, 0.3, qubit); });
  return b.measure(powOut).second;
}

static void makePowUParameterDynamic(ModuleOp program) {
  auto funcOp = cast<func::FuncOp>(program.getBody()->front());
  ASSERT_TRUE(succeeded(funcOp.insertArgument(
      0, Float64Type::get(program.getContext()), {}, funcOp.getLoc())));
  auto powOp = *funcOp.getBody().getOps<qco::PowOp>().begin();
  auto uOp = *powOp.getBody()->getOps<qco::UOp>().begin();
  uOp.getThetaMutable().assign(funcOp.getArgument(0));
}

static Value ifWithAngle(qco::QCOProgramBuilder& b) {
  auto theta = b.floatConstant(0.123);
  auto reg = b.allocQubitRegister(1);
  auto q0 = b.h(reg[0]);
  auto [q1, measureResult] = b.measure(q0);
  auto q2 = b.qcoIf(measureResult, q1, [&](ValueRange args) {
    auto innerQubit = b.rx(theta, args[0]);
    return SmallVector{innerQubit};
  })[0];
  return b.measure(q2).second;
}

static Value forLoopWithAngle(qco::QCOProgramBuilder& b) {
  auto theta = b.floatConstant(0.123);
  auto reg = b.qtensorAlloc(2);
  auto res = b.scfFor(0, 2, 1, {reg}, [&](Value iv, ValueRange iterArgs) {
    auto [t0, q0] = b.qtensorExtract(iterArgs[0], iv);
    auto q1 = b.rx(theta, q0);
    auto insert = b.qtensorInsert(q1, t0, iv);
    return SmallVector{insert};
  });
  auto q = b.qtensorExtract(res[0], 0).second;
  return b.measure(q).second;
}

static Value nestedIfOpForLoopWithAngle(qco::QCOProgramBuilder& b) {
  auto theta1 = b.floatConstant(0.123);
  auto theta2 = b.floatConstant(0.456);
  auto reg = b.qtensorAlloc(3);
  auto q0 = b.allocQubit();
  auto q1 = b.h(q0);
  auto [q2, cond] = b.measure(q1);
  auto res = b.qcoIf(
      cond, {reg, q2},
      [&](ValueRange args) {
        auto q3 = b.rx(theta1, args[1]);
        return SmallVector{args[0], q3};
      },
      [&](ValueRange args) {
        auto scfFor =
            b.scfFor(0, 3, 1, args[0], [&](Value iv, ValueRange iterArgs) {
              auto [t0, q4] = b.qtensorExtract(iterArgs[0], iv);
              auto q5 = b.rx(theta2, q4);
              auto insert = b.qtensorInsert(q5, t0, iv);
              return SmallVector{insert};
            });
        return SmallVector{scfFor[0], args[1]};
      });
  return b.measure(res[1]).second;
}

static SmallVector<Value>
nestedIfWithCapturedMeasurement(qco::QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto c0 = b.allocClassicalBitRegister(1);
  auto c1 = b.allocClassicalBitRegister(1);
  auto measuredQubit = b.measure(q[0], c0, 0).first;
  auto results =
      b.qcoIf(c0, 0, {measuredQubit, q[1]}, [&](ValueRange outerArgs) {
        auto innerResult = b.qcoIf(true, outerArgs[1], [&](Value innerArg) {
          return b.measure(innerArg, c1, 0).first;
        });
        return SmallVector{outerArgs[0], innerResult};
      });
  b.sink(results[0]);
  b.sink(results[1]);
  return {c0, c1};
}

static Value whileWithAngle(qco::QCOProgramBuilder& b) {
  auto theta = b.floatConstant(0.123);
  auto q0 = b.allocQubit();
  auto q1 = b.h(q0);
  auto res = b.scfWhile(
      q1,
      [&](ValueRange iterArgs) {
        auto [q2, measureResult] = b.measure(iterArgs[0]);
        b.scfCondition(measureResult, q2);
        return SmallVector{q2};
      },
      [&](ValueRange iterArgs) {
        auto q3 = b.rx(theta, iterArgs[0]);
        return SmallVector{q3};
      });
  return b.measure(res[0]).second;
}

static Value forLoopWithTwoMeasurements(qco::QCOProgramBuilder& b) {
  auto reg = b.allocQubitRegister(2);
  auto c = b.allocClassicalBitRegister(2);
  b.scfFor(0, 1, 1, {reg.value}, [&](Value /*iv*/, ValueRange iterArgs) {
    auto [t0, q0] = b.qtensorExtract(iterArgs[0], 0);
    auto q0m = b.measure(q0, c, 0).first;
    auto t1 = b.qtensorInsert(q0m, t0, 0);
    auto [t2, q1] = b.qtensorExtract(t1, 1);
    auto q1m = b.measure(q1, c, 1).first;
    auto t3 = b.qtensorInsert(q1m, t2, 1);
    return SmallVector{t3};
  });
  return c;
}

static Value nestedForLoopForOp(qco::QCOProgramBuilder& b) {
  auto theta = b.floatConstant(0.123);
  auto reg = b.qtensorAlloc(2);
  auto res = b.scfFor(0, 2, 1, {reg}, [&](Value /*iv*/, ValueRange iterArgs) {
    auto inner =
        b.scfFor(0, 2, 1, iterArgs, [&](Value innerIv, ValueRange innerArgs) {
          auto [t0, q0] = b.qtensorExtract(innerArgs[0], innerIv);
          auto q1 = b.rx(theta, q0);
          auto insert = b.qtensorInsert(q1, t0, innerIv);
          return SmallVector{insert};
        });
    return SmallVector{inner[0]};
  });
  auto q = b.qtensorExtract(res[0], 0).second;
  return b.measure(q).second;
}

static Value whileWithMeasurement(qco::QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto c = b.allocClassicalBitRegister(1);
  auto q1 = b.h(q0);
  auto res = b.scfWhile(
      q1,
      [&](ValueRange iterArgs) {
        auto [q2, measureResult] = b.measure(iterArgs[0], c, 0);
        b.scfCondition(measureResult, q2);
        return SmallVector{q2};
      },
      [&](ValueRange iterArgs) {
        auto q3 = b.h(iterArgs[0]);
        return SmallVector{q3};
      });
  b.sink(res[0]);
  return c;
}

static Value whileWithRead(qco::QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto c = b.allocClassicalBitRegister(1);
  auto q1 = b.h(q0);
  auto res = b.scfWhile(
      q1,
      [&](ValueRange iterArgs) {
        auto q2 = b.measure(iterArgs[0], c, 0).first;
        b.scfCondition(c, 0, q2);
        return SmallVector{q2};
      },
      [&](ValueRange iterArgs) {
        auto q3 = b.h(iterArgs[0]);
        return SmallVector{q3};
      });
  b.sink(res[0]);
  return c;
}

static Value nestedWhileOpIfOp(qco::QCOProgramBuilder& b) {
  auto q0 = b.allocQubit();
  auto c = b.allocClassicalBitRegister(1);
  auto q1 = b.h(q0);
  auto res = b.scfWhile(
      q1,
      [&](ValueRange iterArgs) {
        auto q2 = b.measure(iterArgs[0], c, 0).first;
        b.scfCondition(c, 0, q2);
        return SmallVector{q2};
      },
      [&](ValueRange iterArgs) {
        auto inner = b.qcoIf(c, 0, iterArgs[0], [&](ValueRange innerArgs) {
          return SmallVector{b.x(innerArgs[0])};
        });
        return SmallVector{inner[0]};
      });
  return b.measure(res[0]).second;
}

static LogicalResult convertQCOToJeff(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(mlir::mqt::createUnrollModifiers());
  pm.addPass(createQCOToJeff());
  return pm.run(module);
}

static LogicalResult convertJeffToQCO(ModuleOp module) {
  PassManager pm(module.getContext());
  pm.addPass(createJeffToQCO());
  return pm.run(module);
}

TEST(JeffRoundTripRegressionTest, RestoresStatusResultAtEndOfEntryPoint) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, arith::ArithDialect, cbit::CBitDialect,
                  func::FuncDialect, jeff::JeffDialect, qco::QCODialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto program = ModuleOp::create(loc);
  program->setAttr(
      "jeff.entrypoint",
      builder.getIntegerAttr(builder.getIntegerType(16, false), 0));
  program->setAttr("jeff.strings",
                   builder.getArrayAttr({builder.getStringAttr("main")}));

  auto main = func::FuncOp::create(builder, loc, "main",
                                   builder.getFunctionType({}, {}));
  program.push_back(main);
  auto* block = main.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  arith::ConstantIntOp::create(builder, loc, 1, 1);
  func::ReturnOp::create(builder, loc);

  ASSERT_TRUE(succeeded(convertJeffToQCO(program)));
  EXPECT_TRUE(succeeded(verify(program)));
  EXPECT_EQ(main.getFunctionType(),
            builder.getFunctionType({}, {builder.getI64Type()}));
  auto returnOp = cast<func::ReturnOp>(block->getTerminator());
  ASSERT_EQ(returnOp.getNumOperands(), 1);
  EXPECT_TRUE(returnOp.getOperand(0).getType().isInteger(64));
}

TEST(JeffRoundTripRegressionTest, RestoresEntryPointWithObservableResults) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, arith::ArithDialect, cbit::CBitDialect,
                  func::FuncDialect, jeff::JeffDialect, memref::MemRefDialect,
                  qco::QCODialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto program = ::mqt::test::buildMLIRProgram(
      &context, MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit));
  ASSERT_TRUE(program);
  ASSERT_TRUE(succeeded(convertQCOToJeff(*program)));
  auto jeffMain = program->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(jeffMain);
  auto jeffReturn =
      cast<func::ReturnOp>(jeffMain.getBody().front().getTerminator());
  ASSERT_EQ(jeffReturn.getNumOperands(), 1);
  EXPECT_TRUE(
      jeffReturn.getOperand(0).getDefiningOp<jeff::IntArraySetIndexOp>());

  ASSERT_TRUE(succeeded(convertJeffToQCO(*program)));
  auto main = program->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  EXPECT_TRUE(mlir::mqt::isEntryPoint(main));
  auto cregType = cbit::RegisterType::get(&context, 1);
  ASSERT_EQ(main.getFunctionType().getNumResults(), 1);
  EXPECT_EQ(main.getFunctionType().getResult(0), cregType);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  ASSERT_EQ(returnOp.getNumOperands(), 1);
  EXPECT_EQ(returnOp.getOperand(0).getType(), cregType);
}

TEST(JeffRoundTripRegressionTest, ConvertsJeffBitArraysDirectlyToCBit) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, arith::ArithDialect, cbit::CBitDialect,
                  func::FuncDialect, jeff::JeffDialect, qco::QCODialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto program =
      ::mqt::test::buildMLIRProgram(&context, MQT_NAMED_BUILDER(whileWithRead));
  ASSERT_TRUE(program);
  ASSERT_TRUE(succeeded(convertQCOToJeff(*program)));

  auto data = serialize(*program);
  program = deserialize(&context, data);
  ASSERT_TRUE(program);
  ASSERT_TRUE(succeeded(convertJeffToQCO(*program)));
  ASSERT_TRUE(succeeded(verify(*program)));

  size_t allocations = 0;
  size_t loads = 0;
  size_t stores = 0;
  bool hasI1Tensor = false;
  program->walk([&](Operation* op) {
    allocations += isa<cbit::AllocOp>(op);
    loads += isa<cbit::LoadOp>(op);
    stores += isa<cbit::StoreOp>(op);
    hasI1Tensor |= llvm::any_of(op->getResultTypes(), [](Type type) {
      const auto tensorType = dyn_cast<RankedTensorType>(type);
      return tensorType && tensorType.getElementType().isInteger(1);
    });
  });

  EXPECT_EQ(allocations, 1);
  EXPECT_GE(loads, 1);
  EXPECT_GE(stores, 1);
  EXPECT_FALSE(hasI1Tensor);
}

TEST(JeffRoundTripRegressionTest, RejectsClassicalIfResultsPrecisely) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, arith::ArithDialect, cbit::CBitDialect,
                  func::FuncDialect, jeff::JeffDialect, qco::QCODialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%condition: i1) -> i64
      attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %then = arith.constant 1 : i64
    %else = arith.constant 2 : i64
    %result, %q1 = qco.if %condition args(%arg = %q0)
        -> (i64, !qco.qubit) {
      qco.yield %then, %arg : i64, !qco.qubit
    } else args(%arg = %q0) {
      qco.yield %else, %arg : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(stream);
    sawExpectedDiagnostic |= StringRef(message).contains(
        "classical qco.if results are not supported by the QCO-to-Jeff "
        "conversion");
    return success();
  });
  EXPECT_TRUE(failed(convertQCOToJeff(*module)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST(JeffRoundTripRegressionTest, RejectsLegacyClassicalMemref) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, arith::ArithDialect, cbit::CBitDialect,
                  func::FuncDialect, jeff::JeffDialect, memref::MemRefDialect,
                  qco::QCODialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%size: index) -> memref<?xi1>
      attributes {mqt.entry_point} {
    %c = memref.alloc(%size) : memref<?xi1>
    return %c : memref<?xi1>
  }
}
)mlir";
  auto module = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(failed(convertQCOToJeff(*module)));
}

TEST_P(JeffRoundTripTest, ProgramEquivalence) {
  const auto& [nameStr, programBuilder, referenceBuilder,
               expectPowAfterCleanup] = GetParam();
  const auto name = " (" + nameStr + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  if (expectPowAfterCleanup) {
    makePowUParameterDynamic(*program);
  }
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());
  if (expectPowAfterCleanup) {
    size_t powCount = 0;
    program->walk([&](qco::PowOp) { ++powCount; });
    EXPECT_EQ(powCount, 1U);
  }

  EXPECT_TRUE(succeeded(convertQCOToJeff(program.get())));
  printer.record(program.get(), "Converted jeff IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  PassManager pm(context.get());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  EXPECT_TRUE(pm.run(program.get()).succeeded());

  printer.record(program.get(), "Canonicalized Converted jeff IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  // Serialize and deserialize to ensure the jeff program is valid
  auto data = serialize(*program);
  program = deserialize(context.get(), data);

  EXPECT_TRUE(succeeded(convertJeffToQCO(program.get())));
  printer.record(program.get(), "Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized Converted QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      ::mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  if (expectPowAfterCleanup) {
    makePowUParameterDynamic(*reference);
  }
  printer.record(reference.get(), "Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QCO IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

/// \name JeffRoundTrip/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(QCOCtrlOpTest, JeffRoundTripTest,
                         testing::Values(JeffRoundTripTestCase{
                             "CtrlTwo", MQT_NAMED_BUILDER(qco::ctrlTwo),
                             MQT_NAMED_BUILDER(qco::ctrlTwoUnrolled)}));
/// @}

/// \name JeffRoundTrip/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"InvTwo", MQT_NAMED_BUILDER(qco::invTwo),
                              MQT_NAMED_BUILDER(qco::invTwoUnrolled)},
        JeffRoundTripTestCase{"InverseiSWAP",
                              MQT_NAMED_BUILDER(qco::inverseIswap),
                              MQT_NAMED_BUILDER(qco::inverseIswap)},
        JeffRoundTripTestCase{
            "InverseMultipleControllediSWAP",
            MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap),
            MQT_NAMED_BUILDER(qco::inverseMultipleControlledIswap)},
        JeffRoundTripTestCase{"InverseDCX", MQT_NAMED_BUILDER(qco::inverseDcx),
                              MQT_NAMED_BUILDER(qco::inverseDcx)},
        JeffRoundTripTestCase{
            "InverseMultipleControlledDCX",
            MQT_NAMED_BUILDER(qco::inverseMultipleControlledDcx),
            MQT_NAMED_BUILDER(qco::inverseMultipleControlledDcx)}));
/// @}

/// \name JeffRoundTrip/Modifiers/PowOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPowOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"PowEvenH", MQT_NAMED_BUILDER(qco::powEvenH),
                              MQT_NAMED_BUILDER(qco::alloc1QubitRegister)},
        JeffRoundTripTestCase{"PowOddH", MQT_NAMED_BUILDER(qco::powOddH),
                              MQT_NAMED_BUILDER(qco::h)},
        JeffRoundTripTestCase{"PowRxScaled",
                              MQT_NAMED_BUILDER(qco::powRxScaled),
                              MQT_NAMED_BUILDER(qco::rxScaled)},
        JeffRoundTripTestCase{"PowDCX", MQT_NAMED_BUILDER(powDcx),
                              MQT_NAMED_BUILDER(powDcx)},
        JeffRoundTripTestCase{"PowInvDCX", MQT_NAMED_BUILDER(powInvDcx),
                              MQT_NAMED_BUILDER(powInvDcx)},
        JeffRoundTripTestCase{"CtrlPowDcx", MQT_NAMED_BUILDER(ctrlPowDcx),
                              MQT_NAMED_BUILDER(ctrlPowDcx)},
        JeffRoundTripTestCase{"CtrlPowInvDcx", MQT_NAMED_BUILDER(ctrlPowInvDcx),
                              MQT_NAMED_BUILDER(ctrlPowInvDcx)},
        JeffRoundTripTestCase{"PowU", MQT_NAMED_BUILDER(powU),
                              MQT_NAMED_BUILDER(powU), true}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Barrier", MQT_NAMED_BUILDER(qco::barrier),
                              MQT_NAMED_BUILDER(qco::barrier)},
        JeffRoundTripTestCase{"BarrierTwoQubits",
                              MQT_NAMED_BUILDER(qco::barrierTwoQubits),
                              MQT_NAMED_BUILDER(qco::barrierTwoQubits)},
        JeffRoundTripTestCase{"BarrierMultipleQubits",
                              MQT_NAMED_BUILDER(qco::barrierMultipleQubits),
                              MQT_NAMED_BUILDER(qco::barrierMultipleQubits)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCODCXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"DCX", MQT_NAMED_BUILDER(qco::dcx),
                              MQT_NAMED_BUILDER(qco::dcx)},
        JeffRoundTripTestCase{"SingleControlledDCX",
                              MQT_NAMED_BUILDER(qco::singleControlledDcx),
                              MQT_NAMED_BUILDER(qco::singleControlledDcx)},
        JeffRoundTripTestCase{"MultipleControlledDCX",
                              MQT_NAMED_BUILDER(qco::multipleControlledDcx),
                              MQT_NAMED_BUILDER(qco::multipleControlledDcx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"ECR", MQT_NAMED_BUILDER(qco::ecr),
                              MQT_NAMED_BUILDER(qco::ecr)},
        JeffRoundTripTestCase{"SingleControlledECR",
                              MQT_NAMED_BUILDER(qco::singleControlledEcr),
                              MQT_NAMED_BUILDER(qco::singleControlledEcr)},
        JeffRoundTripTestCase{"MultipleControlledECR",
                              MQT_NAMED_BUILDER(qco::multipleControlledEcr),
                              MQT_NAMED_BUILDER(qco::multipleControlledEcr)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOGPhaseOpTest, JeffRoundTripTest,
    testing::Values(JeffRoundTripTestCase{"GlobalPhase",
                                          MQT_NAMED_BUILDER(qco::globalPhase),
                                          MQT_NAMED_BUILDER(qco::globalPhase)},
                    JeffRoundTripTestCase{
                        "SingleControlledGlobalPhase",
                        MQT_NAMED_BUILDER(qco::singleControlledGlobalPhase),
                        MQT_NAMED_BUILDER(qco::p)},
                    JeffRoundTripTestCase{
                        "MultipleControlledGlobalPhase",
                        MQT_NAMED_BUILDER(qco::multipleControlledGlobalPhase),
                        MQT_NAMED_BUILDER(qco::multipleControlledP)},
                    JeffRoundTripTestCase{
                        "InverseGlobalPhase",
                        MQT_NAMED_BUILDER(qco::inverseGlobalPhase),
                        MQT_NAMED_BUILDER(qco::globalPhase)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"H", MQT_NAMED_BUILDER(qco::h),
                              MQT_NAMED_BUILDER(qco::h)},
        JeffRoundTripTestCase{"SingleControlledH",
                              MQT_NAMED_BUILDER(qco::singleControlledH),
                              MQT_NAMED_BUILDER(qco::singleControlledH)},
        JeffRoundTripTestCase{"MultipleControlledH",
                              MQT_NAMED_BUILDER(qco::multipleControlledH),
                              MQT_NAMED_BUILDER(qco::multipleControlledH)},
        JeffRoundTripTestCase{"HWithoutRegister",
                              MQT_NAMED_BUILDER(qco::hWithoutRegister),
                              MQT_NAMED_BUILDER(qco::hWithoutRegister)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"iSWAP", MQT_NAMED_BUILDER(qco::iswap),
                              MQT_NAMED_BUILDER(qco::iswap)},
        JeffRoundTripTestCase{"SingleControllediSWAP",
                              MQT_NAMED_BUILDER(qco::singleControlledIswap),
                              MQT_NAMED_BUILDER(qco::singleControlledIswap)},
        JeffRoundTripTestCase{
            "MultipleControllediSWAP",
            MQT_NAMED_BUILDER(qco::multipleControlledIswap),
            MQT_NAMED_BUILDER(qco::multipleControlledIswap)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"P", MQT_NAMED_BUILDER(qco::p),
                              MQT_NAMED_BUILDER(qco::p)},
        JeffRoundTripTestCase{"SingleControlledP",
                              MQT_NAMED_BUILDER(qco::singleControlledP),
                              MQT_NAMED_BUILDER(qco::singleControlledP)},
        JeffRoundTripTestCase{"MultipleControlledP",
                              MQT_NAMED_BUILDER(qco::multipleControlledP),
                              MQT_NAMED_BUILDER(qco::multipleControlledP)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORCCXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RCCX", MQT_NAMED_BUILDER(qco::rccx),
                              MQT_NAMED_BUILDER(qco::rccx)},
        JeffRoundTripTestCase{"SingleControlledRCCX",
                              MQT_NAMED_BUILDER(qco::singleControlledRccx),
                              MQT_NAMED_BUILDER(qco::singleControlledRccx)},
        JeffRoundTripTestCase{"MultipleControlledRCCX",
                              MQT_NAMED_BUILDER(qco::multipleControlledRccx),
                              MQT_NAMED_BUILDER(qco::multipleControlledRccx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOROpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"R", MQT_NAMED_BUILDER(qco::r),
                              MQT_NAMED_BUILDER(qco::r)},
        JeffRoundTripTestCase{"SingleControlledR",
                              MQT_NAMED_BUILDER(qco::singleControlledR),
                              MQT_NAMED_BUILDER(qco::singleControlledR)},
        JeffRoundTripTestCase{"MultipleControlledR",
                              MQT_NAMED_BUILDER(qco::multipleControlledR),
                              MQT_NAMED_BUILDER(qco::multipleControlledR)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RX", MQT_NAMED_BUILDER(qco::rx),
                              MQT_NAMED_BUILDER(qco::rx)},
        JeffRoundTripTestCase{"SingleControlledRX",
                              MQT_NAMED_BUILDER(qco::singleControlledRx),
                              MQT_NAMED_BUILDER(qco::singleControlledRx)},
        JeffRoundTripTestCase{"MultipleControlledRX",
                              MQT_NAMED_BUILDER(qco::multipleControlledRx),
                              MQT_NAMED_BUILDER(qco::multipleControlledRx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RXX", MQT_NAMED_BUILDER(qco::rxx),
                              MQT_NAMED_BUILDER(qco::rxx)},
        JeffRoundTripTestCase{"SingleControlledRXX",
                              MQT_NAMED_BUILDER(qco::singleControlledRxx),
                              MQT_NAMED_BUILDER(qco::singleControlledRxx)},
        JeffRoundTripTestCase{"MultipleControlledRXX",
                              MQT_NAMED_BUILDER(qco::multipleControlledRxx),
                              MQT_NAMED_BUILDER(qco::multipleControlledRxx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RY", MQT_NAMED_BUILDER(qco::ry),
                              MQT_NAMED_BUILDER(qco::ry)},
        JeffRoundTripTestCase{"SingleControlledRY",
                              MQT_NAMED_BUILDER(qco::singleControlledRy),
                              MQT_NAMED_BUILDER(qco::singleControlledRy)},
        JeffRoundTripTestCase{"MultipleControlledRY",
                              MQT_NAMED_BUILDER(qco::multipleControlledRy),
                              MQT_NAMED_BUILDER(qco::multipleControlledRy)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RYY", MQT_NAMED_BUILDER(qco::ryy),
                              MQT_NAMED_BUILDER(qco::ryy)},
        JeffRoundTripTestCase{"SingleControlledRYY",
                              MQT_NAMED_BUILDER(qco::singleControlledRyy),
                              MQT_NAMED_BUILDER(qco::singleControlledRyy)},
        JeffRoundTripTestCase{"MultipleControlledRYY",
                              MQT_NAMED_BUILDER(qco::multipleControlledRyy),
                              MQT_NAMED_BUILDER(qco::multipleControlledRyy)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RZ", MQT_NAMED_BUILDER(qco::rz),
                              MQT_NAMED_BUILDER(qco::rz)},
        JeffRoundTripTestCase{"SingleControlledRZ",
                              MQT_NAMED_BUILDER(qco::singleControlledRz),
                              MQT_NAMED_BUILDER(qco::singleControlledRz)},
        JeffRoundTripTestCase{"MultipleControlledRZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledRz),
                              MQT_NAMED_BUILDER(qco::multipleControlledRz)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RZX", MQT_NAMED_BUILDER(qco::rzx),
                              MQT_NAMED_BUILDER(qco::rzx)},
        JeffRoundTripTestCase{"SingleControlledRZX",
                              MQT_NAMED_BUILDER(qco::singleControlledRzx),
                              MQT_NAMED_BUILDER(qco::singleControlledRzx)},
        JeffRoundTripTestCase{"MultipleControlledRZX",
                              MQT_NAMED_BUILDER(qco::multipleControlledRzx),
                              MQT_NAMED_BUILDER(qco::multipleControlledRzx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"RZZ", MQT_NAMED_BUILDER(qco::rzz),
                              MQT_NAMED_BUILDER(qco::rzz)},
        JeffRoundTripTestCase{"SingleControlledRZZ",
                              MQT_NAMED_BUILDER(qco::singleControlledRzz),
                              MQT_NAMED_BUILDER(qco::singleControlledRzz)},
        JeffRoundTripTestCase{"MultipleControlledRZZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledRzz),
                              MQT_NAMED_BUILDER(qco::multipleControlledRzz)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"S", MQT_NAMED_BUILDER(qco::s),
                              MQT_NAMED_BUILDER(qco::s)},
        JeffRoundTripTestCase{"SingleControlledS",
                              MQT_NAMED_BUILDER(qco::singleControlledS),
                              MQT_NAMED_BUILDER(qco::singleControlledS)},
        JeffRoundTripTestCase{"MultipleControlledS",
                              MQT_NAMED_BUILDER(qco::multipleControlledS),
                              MQT_NAMED_BUILDER(qco::multipleControlledS)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Sdg", MQT_NAMED_BUILDER(qco::sdg),
                              MQT_NAMED_BUILDER(qco::sdg)},
        JeffRoundTripTestCase{"SingleControlledSdg",
                              MQT_NAMED_BUILDER(qco::singleControlledSdg),
                              MQT_NAMED_BUILDER(qco::singleControlledSdg)},
        JeffRoundTripTestCase{"MultipleControlledSdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledSdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledSdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SWAP", MQT_NAMED_BUILDER(qco::swap),
                              MQT_NAMED_BUILDER(qco::swap)},
        JeffRoundTripTestCase{"SingleControlledSWAP",
                              MQT_NAMED_BUILDER(qco::singleControlledSwap),
                              MQT_NAMED_BUILDER(qco::singleControlledSwap)},
        JeffRoundTripTestCase{"MultipleControlledSWAP",
                              MQT_NAMED_BUILDER(qco::multipleControlledSwap),
                              MQT_NAMED_BUILDER(qco::multipleControlledSwap)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SX", MQT_NAMED_BUILDER(qco::sx),
                              MQT_NAMED_BUILDER(qco::sx)},
        JeffRoundTripTestCase{"SingleControlledSX",
                              MQT_NAMED_BUILDER(qco::singleControlledSx),
                              MQT_NAMED_BUILDER(qco::singleControlledSx)},
        JeffRoundTripTestCase{"MultipleControlledSX",
                              MQT_NAMED_BUILDER(qco::multipleControlledSx),
                              MQT_NAMED_BUILDER(qco::multipleControlledSx)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SXdg", MQT_NAMED_BUILDER(qco::sxdg),
                              MQT_NAMED_BUILDER(qco::sxdg)},
        JeffRoundTripTestCase{"SingleControlledSXdg",
                              MQT_NAMED_BUILDER(qco::singleControlledSxdg),
                              MQT_NAMED_BUILDER(qco::singleControlledSxdg)},
        JeffRoundTripTestCase{"MultipleControlledSXdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledSxdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledSxdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"T", MQT_NAMED_BUILDER(qco::t_),
                              MQT_NAMED_BUILDER(qco::t_)},
        JeffRoundTripTestCase{"SingleControlledT",
                              MQT_NAMED_BUILDER(qco::singleControlledT),
                              MQT_NAMED_BUILDER(qco::singleControlledT)},
        JeffRoundTripTestCase{"MultipleControlledT",
                              MQT_NAMED_BUILDER(qco::multipleControlledT),
                              MQT_NAMED_BUILDER(qco::multipleControlledT)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Tdg", MQT_NAMED_BUILDER(qco::tdg),
                              MQT_NAMED_BUILDER(qco::tdg)},
        JeffRoundTripTestCase{"SingleControlledTdg",
                              MQT_NAMED_BUILDER(qco::singleControlledTdg),
                              MQT_NAMED_BUILDER(qco::singleControlledTdg)},
        JeffRoundTripTestCase{"MultipleControlledTdg",
                              MQT_NAMED_BUILDER(qco::multipleControlledTdg),
                              MQT_NAMED_BUILDER(qco::multipleControlledTdg)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOU2OpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"U2", MQT_NAMED_BUILDER(qco::u2),
                              MQT_NAMED_BUILDER(qco::u2)},
        JeffRoundTripTestCase{"SingleControlledU2",
                              MQT_NAMED_BUILDER(qco::singleControlledU2),
                              MQT_NAMED_BUILDER(qco::singleControlledU2)},
        JeffRoundTripTestCase{"MultipleControlledU2",
                              MQT_NAMED_BUILDER(qco::multipleControlledU2),
                              MQT_NAMED_BUILDER(qco::multipleControlledU2)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"U", MQT_NAMED_BUILDER(qco::u),
                              MQT_NAMED_BUILDER(qco::u)},
        JeffRoundTripTestCase{"SingleControlledU",
                              MQT_NAMED_BUILDER(qco::singleControlledU),
                              MQT_NAMED_BUILDER(qco::singleControlledU)},
        JeffRoundTripTestCase{"MultipleControlledU",
                              MQT_NAMED_BUILDER(qco::multipleControlledU),
                              MQT_NAMED_BUILDER(qco::multipleControlledU)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"X", MQT_NAMED_BUILDER(qco::x),
                              MQT_NAMED_BUILDER(qco::x)},
        JeffRoundTripTestCase{"SingleControlledX",
                              MQT_NAMED_BUILDER(qco::singleControlledX),
                              MQT_NAMED_BUILDER(qco::singleControlledX)},
        JeffRoundTripTestCase{"MultipleControlledX",
                              MQT_NAMED_BUILDER(qco::multipleControlledX),
                              MQT_NAMED_BUILDER(qco::multipleControlledX)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXMinusYYOpTest, JeffRoundTripTest,
    testing::Values(JeffRoundTripTestCase{"XXMinusYY",
                                          MQT_NAMED_BUILDER(qco::xxMinusYY),
                                          MQT_NAMED_BUILDER(qco::xxMinusYY)},
                    JeffRoundTripTestCase{
                        "SingleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::singleControlledXxMinusYY)},
                    JeffRoundTripTestCase{
                        "MultipleControlledXXMinusYY",
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY),
                        MQT_NAMED_BUILDER(qco::multipleControlledXxMinusYY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXPlusYYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"XXPlusYY", MQT_NAMED_BUILDER(qco::xxPlusYY),
                              MQT_NAMED_BUILDER(qco::xxPlusYY)},
        JeffRoundTripTestCase{"SingleControlledXXPlusYY",
                              MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY),
                              MQT_NAMED_BUILDER(qco::singleControlledXxPlusYY)},
        JeffRoundTripTestCase{
            "MultipleControlledXXPlusYY",
            MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY),
            MQT_NAMED_BUILDER(qco::multipleControlledXxPlusYY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Y", MQT_NAMED_BUILDER(qco::y),
                              MQT_NAMED_BUILDER(qco::y)},
        JeffRoundTripTestCase{"SingleControlledY",
                              MQT_NAMED_BUILDER(qco::singleControlledY),
                              MQT_NAMED_BUILDER(qco::singleControlledY)},
        JeffRoundTripTestCase{"MultipleControlledY",
                              MQT_NAMED_BUILDER(qco::multipleControlledY),
                              MQT_NAMED_BUILDER(qco::multipleControlledY)}));
/// @}

/// \name JeffRoundTrip/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"Z", MQT_NAMED_BUILDER(qco::z),
                              MQT_NAMED_BUILDER(qco::z)},
        JeffRoundTripTestCase{"SingleControlledZ",
                              MQT_NAMED_BUILDER(qco::singleControlledZ),
                              MQT_NAMED_BUILDER(qco::singleControlledZ)},
        JeffRoundTripTestCase{"MultipleControlledZ",
                              MQT_NAMED_BUILDER(qco::multipleControlledZ),
                              MQT_NAMED_BUILDER(qco::multipleControlledZ)}));
/// @}

/// \name JeffRoundTrip/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{
            "SingleMeasurementToSingleBit",
            MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit),
            MQT_NAMED_BUILDER(qco::singleMeasurementToSingleBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToSameBit",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToSameBit)},
        JeffRoundTripTestCase{
            "RepeatedMeasurementToDifferentBits",
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits),
            MQT_NAMED_BUILDER(qco::repeatedMeasurementToDifferentBits)},
        JeffRoundTripTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(qco::multipleClassicalRegistersAndMeasurements)},
        JeffRoundTripTestCase{
            "PartialMeasurementToRegister",
            MQT_NAMED_BUILDER(qco::partialMeasurementToRegister),
            MQT_NAMED_BUILDER(qco::partialMeasurementToRegister)},
        JeffRoundTripTestCase{
            "DynamicallyIndexedMeasurement",
            MQT_NAMED_BUILDER(qco::dynamicallyIndexedMeasurement),
            MQT_NAMED_BUILDER(qco::dynamicallyIndexedMeasurement)},
        JeffRoundTripTestCase{
            "MeasurementWithoutRegisters",
            MQT_NAMED_BUILDER(qco::measurementWithoutRegisters),
            MQT_NAMED_BUILDER(qco::measurementWithoutRegisters)}));
/// @}

/// \name JeffRoundTrip/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"ResetQubitAfterSingleOp",
                              MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp),
                              MQT_NAMED_BUILDER(qco::resetQubitAfterSingleOp)},
        JeffRoundTripTestCase{
            "ResetMultipleQubitsAfterSingleOp",
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp),
            MQT_NAMED_BUILDER(qco::resetMultipleQubitsAfterSingleOp)}));
/// @}

/// \name JeffRoundTrip/Operations/IfOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIfOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SimpleIf", MQT_NAMED_BUILDER(qco::simpleIf),
                              MQT_NAMED_BUILDER(qco::simpleIf)},
        JeffRoundTripTestCase{"IfElse", MQT_NAMED_BUILDER(qco::ifElse),
                              MQT_NAMED_BUILDER(qco::ifElse)},
        JeffRoundTripTestCase{"IfTwoQubits",
                              MQT_NAMED_BUILDER(qco::ifTwoQubits),
                              MQT_NAMED_BUILDER(qco::ifTwoQubits)},
        JeffRoundTripTestCase{"IfWithMeasurement",
                              MQT_NAMED_BUILDER(qco::ifWithMeasurement),
                              MQT_NAMED_BUILDER(qco::ifWithMeasurement)},
        JeffRoundTripTestCase{"IfWithCreg", MQT_NAMED_BUILDER(qco::ifWithCreg),
                              MQT_NAMED_BUILDER(qco::ifWithCreg)},
        JeffRoundTripTestCase{"IfWithAngle", MQT_NAMED_BUILDER(ifWithAngle),
                              MQT_NAMED_BUILDER(ifWithAngle)},
        JeffRoundTripTestCase{"NestedIfOpForLoop",
                              MQT_NAMED_BUILDER(qco::nestedIfOpForLoop),
                              MQT_NAMED_BUILDER(qco::nestedIfOpForLoop)},
        JeffRoundTripTestCase{"NestedIfOpForLoopWithAngle",
                              MQT_NAMED_BUILDER(nestedIfOpForLoopWithAngle),
                              MQT_NAMED_BUILDER(nestedIfOpForLoopWithAngle)},
        JeffRoundTripTestCase{
            "NestedIfWithCapturedMeasurement",
            MQT_NAMED_BUILDER(nestedIfWithCapturedMeasurement),
            MQT_NAMED_BUILDER(nestedIfWithCapturedMeasurement)}));
/// @}

/// \name JeffRoundTrip/Operations/ForOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFForOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SimpleForLoop",
                              MQT_NAMED_BUILDER(qco::simpleForLoop),
                              MQT_NAMED_BUILDER(qco::simpleForLoop)},
        JeffRoundTripTestCase{"ForLoopWithAngle",
                              MQT_NAMED_BUILDER(forLoopWithAngle),
                              MQT_NAMED_BUILDER(forLoopWithAngle)},
        JeffRoundTripTestCase{"ForLoopWithTwoMeasurements",
                              MQT_NAMED_BUILDER(forLoopWithTwoMeasurements),
                              MQT_NAMED_BUILDER(forLoopWithTwoMeasurements)},
        JeffRoundTripTestCase{"NestedForLoopIfOp",
                              MQT_NAMED_BUILDER(qco::nestedForLoopIfOp),
                              MQT_NAMED_BUILDER(qco::nestedForLoopIfOp)},
        JeffRoundTripTestCase{"NestedForLoopWhileOp",
                              MQT_NAMED_BUILDER(qco::nestedForLoopWhileOp),
                              MQT_NAMED_BUILDER(qco::nestedForLoopWhileOp)},
        JeffRoundTripTestCase{"NestedForLoopForOp",
                              MQT_NAMED_BUILDER(nestedForLoopForOp),
                              MQT_NAMED_BUILDER(nestedForLoopForOp)},
        JeffRoundTripTestCase{
            "NestedForLoopCtrlOpWithSeparateQubit",
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithSeparateQubit),
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithSeparateQubit)},
        JeffRoundTripTestCase{
            "NestedForLoopCtrlOpWithExtractedQubit",
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithExtractedQubit),
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithExtractedQubit)}));
/// @}

/// \name JeffRoundTrip/Operations/WhileOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    SCFWhileOpTest, JeffRoundTripTest,
    testing::Values(
        JeffRoundTripTestCase{"SimpleWhile",
                              MQT_NAMED_BUILDER(qco::simpleWhileReset),
                              MQT_NAMED_BUILDER(qco::simpleWhileReset)},
        JeffRoundTripTestCase{"SimpleDoWhile",
                              MQT_NAMED_BUILDER(qco::simpleDoWhileReset),
                              MQT_NAMED_BUILDER(qco::simpleDoWhileReset)},
        JeffRoundTripTestCase{"NestedWhileOpIfOp",
                              MQT_NAMED_BUILDER(nestedWhileOpIfOp),
                              MQT_NAMED_BUILDER(nestedWhileOpIfOp)},
        JeffRoundTripTestCase{"WhileWithAngle",
                              MQT_NAMED_BUILDER(whileWithAngle),
                              MQT_NAMED_BUILDER(whileWithAngle)},
        JeffRoundTripTestCase{"WhileWithMeasurement",
                              MQT_NAMED_BUILDER(whileWithMeasurement),
                              MQT_NAMED_BUILDER(whileWithMeasurement)},
        JeffRoundTripTestCase{"WhileWithRead", MQT_NAMED_BUILDER(whileWithRead),
                              MQT_NAMED_BUILDER(whileWithRead)}));
/// @}
