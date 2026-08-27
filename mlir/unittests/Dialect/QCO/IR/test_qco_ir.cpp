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
#include "mlir/Dialect/CBit/IR/CBitAttributes.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/CBit/IR/CBitOps.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"
#include "qco_programs.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {

struct QCOTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<QCOProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<QCOProgramBuilder> referenceBuilder;

  friend std::ostream& operator<<(std::ostream& os, const QCOTestCase& info);
};

// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
std::ostream& operator<<(std::ostream& os, const QCOTestCase& info) {
  return os << "QCO{" << info.name << ", original="
            << ::mqt::test::displayName(info.programBuilder.name)
            << ", reference="
            << ::mqt::test::displayName(info.referenceBuilder.name) << "}";
}

class QCOTest : public testing::TestWithParam<QCOTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override;
};
} // namespace

void QCOTest::SetUp() {
  // Register all necessary dialects
  DialectRegistry registry;
  registry.insert<cbit::CBitDialect, QCODialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
                  qtensor::QTensorDialect>();
  context = std::make_unique<MLIRContext>();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

static Value measureRegister(QCOProgramBuilder& b, ValueRange qubits) {
  auto c = b.allocClassicalBitRegister(static_cast<int64_t>(qubits.size()));
  for (auto [i, qubit] : llvm::enumerate(qubits)) {
    b.measure(qubit, c, static_cast<int64_t>(i));
  }
  return c;
}

TEST_P(QCOTest, ProgramEquivalence) {
  const auto& [_, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + GetParam().name + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

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

TEST_F(QCOTest, QubitIsVectorElement) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @f(%arg: vector<2x!qco.qubit>) {
        return
      }
    }
  )mlir",
                                            context.get());
  ASSERT_TRUE(module);

  auto function = *module->getOps<func::FuncOp>().begin();
  const auto vectorType =
      dyn_cast<VectorType>(function.getArgument(0).getType());
  ASSERT_TRUE(vectorType);
  EXPECT_TRUE(isa<QubitType>(vectorType.getElementType()));
}

TEST_F(QCOTest, BuilderRejectsMixedStaticAndDynamicQubitAllocationModes) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        mixedStaticThenDynamicQubit(builder);
      },
      "Cannot mix static and dynamic qubit allocation modes");

  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        mixedDynamicRegisterThenStaticQubit(builder);
      },
      "Cannot mix dynamic and static qubit allocation modes");
}

TEST_F(QCOTest, BuilderReturnsTrackedQubit) {
  static_assert(std::is_convertible_v<Value, QCOProgramBuilder::Qubit>);
  static_assert(std::is_constructible_v<QCOProgramBuilder::Qubit, Value>);
  static_assert(std::is_assignable_v<QCOProgramBuilder::Qubit&, Value>);
  static_assert(std::is_convertible_v<QCOProgramBuilder::Qubit, Value>);
  static_assert(std::is_convertible_v<Value, QCOProgramBuilder::Tensor>);
  static_assert(std::is_convertible_v<QCOProgramBuilder::Tensor, Value>);

  QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto qubit = builder.allocQubit();

  EXPECT_TRUE(qubit.getDefiningOp<AllocOp>());
  EXPECT_EQ(qubit.regId, -1);
  EXPECT_FALSE(qubit.regIndex);

  auto output = builder.x(qubit);
  auto reassigned = QCOProgramBuilder::Qubit{qubit.value, 0, qubit.value};
  reassigned = output;
  EXPECT_EQ(reassigned.value, output);
  EXPECT_EQ(reassigned.regId, -1);
  EXPECT_FALSE(reassigned.regIndex);

  EXPECT_DEATH(builder.x(qubit), "Invalid qubit value used");
  EXPECT_NO_FATAL_FAILURE(builder.x(output));
}

TEST_F(QCOTest, CleanupPreservesReturnedStaticQubit) {
  auto module = QCOProgramBuilder::build(
      context.get(), [&](auto& builder) { return builder.staticQubit(0); });
  ASSERT_TRUE(module);

  auto mainFunc = *module->getOps<func::FuncOp>().begin();
  auto returnOp = cast<func::ReturnOp>(mainFunc.getBody().front().back());
  ASSERT_EQ(returnOp.getNumOperands(), 1U);
  auto returnedQubit = returnOp.getOperand(0);
  EXPECT_TRUE(returnedQubit.getDefiningOp<StaticOp>());
  EXPECT_TRUE(returnedQubit.hasOneUse());
  EXPECT_EQ(*returnedQubit.user_begin(), returnOp.getOperation());
  EXPECT_TRUE(mainFunc.getBody().getOps<SinkOp>().empty());

  ASSERT_TRUE(runQCOCleanupPipeline(*module).succeeded());
  EXPECT_TRUE(verify(*module).succeeded());

  returnOp = cast<func::ReturnOp>(mainFunc.getBody().front().back());
  EXPECT_TRUE(returnOp.getOperand(0).getDefiningOp<StaticOp>());
}

TEST_F(QCOTest, CleanupPreservesReturnedQubitTensor) {
  auto module = QCOProgramBuilder::build(
      context.get(), [&](auto& builder) { return builder.qtensorAlloc(2); });
  ASSERT_TRUE(module);

  auto mainFunc = *module->getOps<func::FuncOp>().begin();
  auto returnOp = cast<func::ReturnOp>(mainFunc.getBody().front().back());
  ASSERT_EQ(returnOp.getNumOperands(), 1U);
  auto returnedTensor = returnOp.getOperand(0);
  EXPECT_TRUE(returnedTensor.getDefiningOp<qtensor::AllocOp>());
  EXPECT_TRUE(returnedTensor.hasOneUse());
  EXPECT_EQ(*returnedTensor.user_begin(), returnOp.getOperation());
  EXPECT_TRUE(mainFunc.getBody().getOps<qtensor::DeallocOp>().empty());

  ASSERT_TRUE(runQCOCleanupPipeline(*module).succeeded());
  EXPECT_TRUE(verify(*module).succeeded());

  returnOp = cast<func::ReturnOp>(mainFunc.getBody().front().back());
  EXPECT_TRUE(returnOp.getOperand(0).getDefiningOp<qtensor::AllocOp>());
}

TEST_F(QCOTest, BuilderRejectsUntrackedTensorInitArg) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        auto size = arith::ConstantIndexOp::create(builder, 1);
        auto tensor =
            qtensor::AllocOp::create(builder, size.getResult()).getResult();
        const auto identity = [](Value value) { return value; };
        builder.qcoIf(true, tensor, identity, identity);
      },
      "Invalid tensor value used");
}

TEST_F(QCOTest, BuilderRejectsOutOfBoundsClassicalRegisterIndices) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        const auto q = builder.allocQubit();
        auto c = builder.allocClassicalBitRegister(1);
        builder.measure(q, c, -1);
      },
      "Register index must be non-negative");

  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        const auto q = builder.allocQubit();
        auto c = builder.allocClassicalBitRegister(1);
        builder.measure(q, c, 1);
      },
      "Register index is out of bounds");

  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        auto c = builder.allocClassicalBitRegister(1);
        builder.qcoIf(c, -1, ValueRange{},
                      [](ValueRange) { return SmallVector<Value>{}; });
      },
      "Register index must be non-negative");

  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        auto c = builder.allocClassicalBitRegister(1);
        builder.scfCondition(c, 1, ValueRange{});
      },
      "Register index is out of bounds");
}

TEST_F(QCOTest, BuilderSupportsIndependentClassicalRegisterInitialization) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto zero = builder.allocClassicalBitRegister(3);
  auto undefined = builder.allocClassicalBitRegister(
      2, "undefined", cbit::Initialization::Undefined);
  builder.retype({zero.getType(), undefined.getType()});
  auto moduleOp = builder.finalize({zero, undefined});
  ASSERT_TRUE(moduleOp);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));

  SmallVector<cbit::AllocOp> allocations;
  moduleOp->walk([&](cbit::AllocOp op) { allocations.push_back(op); });
  ASSERT_EQ(allocations.size(), 2);
  EXPECT_EQ(allocations[0].getInitialization(), cbit::Initialization::Zero);
  EXPECT_FALSE(allocations[0]->getAttr(
      ::mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr()));
  EXPECT_EQ(allocations[1].getInitialization(),
            cbit::Initialization::Undefined);
  EXPECT_EQ(
      allocations[1]
          ->getAttrOfType<StringAttr>(
              ::mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr())
          .getValue(),
      "undefined");
}

TEST_F(QCOTest, DirectSingleQubitPowBuilder) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();
  const auto qubit = builder.allocQubit();

  Value bodyQubit;
  Value bodyResult;
  auto pow = PowOp::create(builder, qubit, 2.0, [&](Value argument) -> Value {
    bodyQubit = argument;
    bodyResult = XOp::create(builder, argument);
    return bodyResult;
  });

  ASSERT_EQ(pow.getQubitsIn().size(), 1);
  ASSERT_EQ(pow.getQubitsOut().size(), 1);
  ASSERT_EQ(pow.getBody()->getNumArguments(), 1);
  ASSERT_EQ(pow.getBody()->getTerminator()->getNumOperands(), 1);
  EXPECT_EQ(pow.getQubitsIn().front(), qubit.value);
  EXPECT_EQ(pow.getBody()->getArgument(0), bodyQubit);
  EXPECT_EQ(pow.getBody()->getTerminator()->getOperand(0), bodyResult);
  EXPECT_TRUE(pow.verify().succeeded());
}

TEST_F(QCOTest, UnitaryVerifierRejectsNonFiniteConstantParameters) {
  constexpr std::array<StringLiteral, 2> invalidPrograms{
      R"mlir(
        module {
          func.func @main(%input: f64) {
            %q = qco.alloc : !qco.qubit
            %infinity = arith.constant 0x7FF0000000000000 : f64
            %theta = arith.addf %input, %infinity : f64
            %out = qco.rx(%theta) %q : !qco.qubit -> !qco.qubit
            qco.sink %out : !qco.qubit
            return
          }
        }
      )mlir",
      R"mlir(
        module {
          func.func @main() {
            %q = qco.alloc : !qco.qubit
            %nan = arith.constant 0x7FF8000000000000 : f64
            %out = qco.pow(%nan) (%arg = %q) {
              %body = qco.x %arg : !qco.qubit -> !qco.qubit
              qco.yield %body : !qco.qubit
            } : {!qco.qubit} -> {!qco.qubit}
            qco.sink %out : !qco.qubit
            return
          }
        }
      )mlir"};

  for (const auto source : invalidPrograms) {
    bool sawExpectedDiagnostic = false;
    ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
      sawExpectedDiagnostic |= StringRef(diagnostic.str())
                                   .contains("constant parameter expression at "
                                             "index 0 must be finite");
      return success();
    });
    EXPECT_FALSE(parseSourceString<ModuleOp>(source, context.get()));
    EXPECT_TRUE(sawExpectedDiagnostic);
  }
}

namespace {

enum class VerifierModifierKind : uint8_t { Inv, Ctrl, Pow };
enum class ForbiddenModifierBodyOp : uint8_t {
  Measure,
  CBitAlloc,
  CBitLoad,
  CBitStore
};

} // namespace

static StringRef modifierName(const VerifierModifierKind kind) {
  switch (kind) {
  case VerifierModifierKind::Inv:
    return "inv";
  case VerifierModifierKind::Ctrl:
    return "ctrl";
  case VerifierModifierKind::Pow:
    return "pow";
  }
  llvm_unreachable("unknown modifier");
}

static StringRef forbiddenOperationName(ForbiddenModifierBodyOp kind) {
  switch (kind) {
  case ForbiddenModifierBodyOp::Measure:
    return "measure";
  case ForbiddenModifierBodyOp::CBitAlloc:
    return "cbit.alloc";
  case ForbiddenModifierBodyOp::CBitLoad:
    return "cbit.load";
  case ForbiddenModifierBodyOp::CBitStore:
    return "cbit.store";
  }
  llvm_unreachable("unknown forbidden modifier operation");
}

static Operation*
buildInvalidModifierCapture(QCOProgramBuilder& builder,
                            const VerifierModifierKind modifier,
                            const bool nested) {
  builder.initialize();
  const auto target = builder.allocQubit();
  const auto captured = builder.allocQubit();
  const auto control = builder.allocQubit();
  const auto modifierBody = [&](Value argument) -> Value {
    if (nested) {
      auto condition = builder.boolConstant(true);
      auto ifOp = scf::IfOp::create(builder, TypeRange{}, condition, false);
      const OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      XOp::create(builder, captured);
    } else {
      XOp::create(builder, captured);
    }
    return argument;
  };

  switch (modifier) {
  case VerifierModifierKind::Inv:
    return InvOp::create(builder, target, modifierBody).getOperation();
  case VerifierModifierKind::Ctrl:
    return CtrlOp::create(builder, control, target, modifierBody)
        .getOperation();
  case VerifierModifierKind::Pow:
    return PowOp::create(builder, target, 2.0, modifierBody).getOperation();
  }
  llvm_unreachable("unknown modifier");
}

static Operation*
buildInvalidNestedModifierBody(QCOProgramBuilder& builder,
                               const VerifierModifierKind modifier,
                               ForbiddenModifierBodyOp forbiddenOperation) {
  builder.initialize();
  const auto target = builder.allocQubit();
  const auto control = builder.allocQubit();
  auto condition = builder.boolConstant(true);
  auto cbitReg = builder.allocClassicalBitRegister(1);
  auto index = arith::ConstantIndexOp::create(builder, 0);
  const auto modifierBody = [&](Value argument) -> Value {
    auto ifOp = IfOp::create(
        builder, condition, argument, [&](Value nestedArgument) -> Value {
          switch (forbiddenOperation) {
          case ForbiddenModifierBodyOp::Measure:
            return MeasureOp::create(builder, nestedArgument).getQubitOut();
          case ForbiddenModifierBodyOp::CBitAlloc:
            cbit::AllocOp::create(
                builder, cbit::RegisterType::get(builder.getContext(), 1),
                cbit::Initialization::Zero);
            break;
          case ForbiddenModifierBodyOp::CBitLoad:
            cbit::LoadOp::create(builder, builder.getI1Type(), cbitReg,
                                 index.getResult());
            break;
          case ForbiddenModifierBodyOp::CBitStore:
            cbit::StoreOp::create(builder, condition, cbitReg,
                                  index.getResult());
            break;
          }
          return nestedArgument;
        });
    return ifOp.getResult(0);
  };

  switch (modifier) {
  case VerifierModifierKind::Inv:
    return InvOp::create(builder, target, modifierBody).getOperation();
  case VerifierModifierKind::Ctrl:
    return CtrlOp::create(builder, control, target, modifierBody)
        .getOperation();
  case VerifierModifierKind::Pow:
    return PowOp::create(builder, target, 2.0, modifierBody).getOperation();
  }
  llvm_unreachable("unknown modifier");
}

TEST_F(QCOTest, ModifiersRecursivelyRejectNonUnitaryOperations) {
  constexpr std::array modifiers{VerifierModifierKind::Inv,
                                 VerifierModifierKind::Ctrl,
                                 VerifierModifierKind::Pow};
  constexpr std::array forbiddenOperations{
      ForbiddenModifierBodyOp::Measure, ForbiddenModifierBodyOp::CBitAlloc,
      ForbiddenModifierBodyOp::CBitLoad, ForbiddenModifierBodyOp::CBitStore};

  for (const auto modifier : modifiers) {
    for (const auto forbiddenOperation : forbiddenOperations) {
      SCOPED_TRACE(testing::Message()
                   << "modifier=" << modifierName(modifier).str()
                   << ", operation="
                   << forbiddenOperationName(forbiddenOperation).str());
      QCOProgramBuilder builder(context.get());
      auto* modifierOp =
          buildInvalidNestedModifierBody(builder, modifier, forbiddenOperation);

      bool sawExpectedDiagnostic = false;
      ScopedDiagnosticHandler handler(
          context.get(), [&](Diagnostic& diagnostic) {
            sawExpectedDiagnostic |=
                StringRef(diagnostic.str())
                    .contains("body must not contain non-unitary operations or "
                              "access registers");
            return success();
          });
      EXPECT_TRUE(failed(verify(modifierOp)));
      EXPECT_TRUE(sawExpectedDiagnostic);
    }
  }
}

TEST_F(QCOTest, ModifiersRejectDirectAndNestedQubitCaptures) {
  constexpr std::array modifiers{VerifierModifierKind::Inv,
                                 VerifierModifierKind::Ctrl,
                                 VerifierModifierKind::Pow};

  for (const auto modifier : modifiers) {
    for (const bool nested : {false, true}) {
      SCOPED_TRACE(testing::Message()
                   << "modifier=" << modifierName(modifier).str()
                   << ", nested=" << nested);
      QCOProgramBuilder builder(context.get());
      auto* modifierOp = buildInvalidModifierCapture(builder, modifier, nested);

      bool sawExpectedDiagnostic = false;
      ScopedDiagnosticHandler handler(
          context.get(), [&](Diagnostic& diagnostic) {
            sawExpectedDiagnostic |=
                StringRef(diagnostic.str())
                    .contains("body must not capture qubits from above; use "
                              "only its aliased block arguments");
            return success();
          });
      EXPECT_TRUE(failed(verify(modifierOp)));
      EXPECT_TRUE(sawExpectedDiagnostic);
    }
  }
}

TEST_F(QCOTest, DirectIfBuilder) {
  QCOProgramBuilder builder(context.get());
  auto cbitType = cbit::RegisterType::get(context.get(), 1);
  builder.initialize({cbitType, cbitType});
  auto zero = arith::ConstantIndexOp::create(builder, 0);
  auto one = arith::ConstantIndexOp::create(builder, 1);
  auto r0 = qtensor::AllocOp::create(builder, one);
  auto extractOp = qtensor::ExtractOp::create(builder, r0, zero);
  auto q1 = HOp::create(builder, extractOp.getResult());
  auto c0 = builder.allocClassicalBitRegister(1);
  auto c1 = builder.allocClassicalBitRegister(1);
  auto measureOp = MeasureOp::create(builder, q1);
  builder.storeClassicalBit(measureOp.getResult(), c0, 0);
  auto condition = builder.loadClassicalBit(c0, 0);
  auto ifOp = IfOp::create(builder, condition, measureOp.getQubitOut(),
                           [&](ValueRange qubits) -> SmallVector<Value> {
                             auto innerQubit = XOp::create(builder, qubits[0]);
                             return SmallVector<Value>{innerQubit};
                           });
  auto finalMeasureOp = MeasureOp::create(builder, ifOp.getResult(0));
  builder.storeClassicalBit(finalMeasureOp.getResult(), c1, 0);
  auto r2 = qtensor::InsertOp::create(builder, finalMeasureOp.getQubitOut(),
                                      extractOp.getOutTensor(), zero);
  qtensor::DeallocOp::create(builder, r2);

  auto direct = builder.finalize({c0, c1});
  ASSERT_TRUE(direct);
  EXPECT_TRUE(verify(*direct).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(direct.get()).succeeded());
  EXPECT_TRUE(verify(*direct).succeeded());

  auto ref =
      ::mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(simpleIf));
  ASSERT_TRUE(ref);
  EXPECT_TRUE(verify(*ref).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(ref.get()).succeeded());
  EXPECT_TRUE(verify(*ref).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(direct.get(), ref.get()));
}

TEST_F(QCOTest, DirectSingleTargetIndexSwitchBuilder) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  auto index = arith::ConstantIndexOp::create(builder, 0);
  const auto target = builder.allocQubit();
  const auto identity = [](Value value) { return value; };
  const SmallVector<function_ref<Value(Value)>> caseBuilders{identity};

  auto switchOp =
      IndexSwitchOp::create(builder, index.getResult(), target,
                            ArrayRef<int64_t>{0}, caseBuilders, identity);

  ASSERT_EQ(switchOp.getTargets().size(), 1);
  ASSERT_EQ(switchOp.getResults().size(), 1);
  ASSERT_EQ(switchOp.getNumCases(), 1);
  EXPECT_EQ(switchOp.getCaseBlock(0)->getArgument(0).getType(),
            target.getType());
  EXPECT_EQ(switchOp.getDefaultBlock()->getArgument(0).getType(),
            target.getType());
  EXPECT_TRUE(switchOp.verify().succeeded());
}

TEST_F(QCOTest, IndexSwitchBuildersRejectMismatchedCasesAndBodies) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        auto index = arith::ConstantIndexOp::create(builder, 0);
        const auto target = builder.allocQubit();
        const auto identity = [](Value value) { return value; };
        const SmallVector<function_ref<Value(Value)>> noCaseBuilders;
        IndexSwitchOp::create(builder, index.getResult(), target,
                              ArrayRef<int64_t>{0}, noCaseBuilders, identity);
      },
      "Each case must have a corresponding case body function");

  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(context.get());
        builder.initialize();
        const auto target = builder.allocQubit();
        const auto identity = [](Value value) { return value; };
        const SmallVector<function_ref<Value(Value)>> noCaseBodies;
        builder.qcoIndexSwitch(0, target, ArrayRef<int64_t>{0}, noCaseBodies,
                               identity);
      },
      "Each case must have a corresponding case body function");
}

TEST_F(QCOTest, IndexSwitchTiedValuesAndTargetExtension) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  auto index = arith::ConstantIndexOp::create(builder, 0);
  const auto target = builder.allocQubit();
  const auto addon = builder.allocQubit();
  const auto identity = [](Value value) { return value; };
  const SmallVector<function_ref<Value(Value)>> caseBuilders{identity};
  auto switchOp =
      IndexSwitchOp::create(builder, index.getResult(), target,
                            ArrayRef<int64_t>{0}, caseBuilders, identity);

  auto* targetOperand = &switchOp->getOpOperand(1);
  auto caseArgument = switchOp.getCaseBlock(0)->getArgument(0);
  auto defaultArgument = switchOp.getDefaultBlock()->getArgument(0);
  auto* caseYieldOperand = &switchOp.getCaseYield(0)->getOpOperand(0);
  auto* defaultYieldOperand = &switchOp.getDefaultYield()->getOpOperand(0);

  EXPECT_EQ(switchOp.getTiedResult(targetOperand), switchOp.getResult(0));
  EXPECT_EQ(switchOp.getTiedTarget(cast<OpResult>(switchOp.getResult(0))),
            targetOperand);
  EXPECT_EQ(switchOp.getTiedCaseBlockArgument(targetOperand, 0), caseArgument);
  EXPECT_EQ(switchOp.getTiedCaseYieldedValue(caseArgument, 0),
            caseYieldOperand);
  EXPECT_EQ(switchOp.getTiedDefaultBlockArgument(targetOperand),
            defaultArgument);
  EXPECT_EQ(switchOp.getTiedDefaultYieldedValue(defaultArgument),
            defaultYieldOperand);

  EXPECT_FALSE(switchOp.getTiedResult(caseYieldOperand));
  EXPECT_EQ(switchOp.getTiedTarget(cast<OpResult>(addon.value)), nullptr);
  EXPECT_FALSE(switchOp.getTiedCaseBlockArgument(caseYieldOperand, 0));
  EXPECT_FALSE(switchOp.getTiedCaseBlockArgument(targetOperand, 1));
  EXPECT_EQ(switchOp.getTiedCaseYieldedValue(defaultArgument, 0), nullptr);
  EXPECT_EQ(switchOp.getTiedCaseYieldedValue(caseArgument, 1), nullptr);
  EXPECT_FALSE(switchOp.getTiedDefaultBlockArgument(defaultYieldOperand));
  EXPECT_EQ(switchOp.getTiedDefaultYieldedValue(caseArgument), nullptr);

  IRRewriter rewriter(context.get());
  rewriter.setInsertionPoint(switchOp);
  EXPECT_EQ(switchOp.replaceWithAdditionalTargets(rewriter, ValueRange{}),
            switchOp);

  auto expanded =
      switchOp.replaceWithAdditionalTargets(rewriter, {addon.value});
  ASSERT_EQ(expanded.getTargets().size(), 2);
  ASSERT_EQ(expanded.getResults().size(), 2);
  EXPECT_EQ(expanded.getTargets()[0], target.value);
  EXPECT_EQ(expanded.getTargets()[1], addon.value);
  for (Region* region : expanded.getRegions()) {
    ASSERT_EQ(region->getNumArguments(), 2);
    auto yield = cast<YieldOp>(region->front().getTerminator());
    ASSERT_EQ(yield.getTargets().size(), 2);
    EXPECT_EQ(yield.getTargets()[0], region->getArgument(0));
    EXPECT_EQ(yield.getTargets()[1], region->getArgument(1));
  }
  EXPECT_TRUE(expanded.verify().succeeded());
}

TEST_F(QCOTest, IfOpParser) {
  // Test IfOp parser
  const char* mlirCode = R"(
      module {
        func.func @main() -> i1 attributes {mqt.entry_point} {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %q0_0 = qco.alloc : !qco.qubit
            %t0 = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
            %q0_1 = qco.h %q0_0 : !qco.qubit -> !qco.qubit
            %q0_2, %cond = qco.measure %q0_1 : !qco.qubit
            %q0_4, %t3 = qco.if %cond args(%arg0 = %q0_2, %arg1 = %t0) -> (!qco.qubit, tensor<1x!qco.qubit>) {
                %q0_3 = qco.x %arg0 : !qco.qubit -> !qco.qubit
                %t1, %q1_0 = qtensor.extract %arg1[%c0] : tensor<1x!qco.qubit>
                %q1_1 = qco.x %q1_0 : !qco.qubit -> !qco.qubit
                %t2 = qtensor.insert %q1_1 into %t1[%c0] : tensor<1x!qco.qubit>
                qco.yield %q0_3, %t2 : !qco.qubit, tensor<1x!qco.qubit>
            } else args(%arg0 = %q0_2, %arg1 = %t0) {
                qco.yield %arg0, %arg1 : !qco.qubit, tensor<1x!qco.qubit>
            }
            %q0_5, %c = qco.measure %q0_4 : !qco.qubit
            qco.sink %q0_5 : !qco.qubit
            qtensor.dealloc %t3 : tensor<1x!qco.qubit>
            return %c : i1
        }
    })";

  auto parsed = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(parsed);
  EXPECT_TRUE(verify(*parsed).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(parsed.get()).succeeded());
  EXPECT_TRUE(verify(*parsed).succeeded());

  auto ref = ::mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(ifOneQubitOneTensor));
  ASSERT_TRUE(ref);
  EXPECT_TRUE(verify(*ref).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(ref.get()).succeeded());
  EXPECT_TRUE(verify(*ref).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(parsed.get(), ref.get()));
}

TEST_F(QCOTest, IfOpWithClassicalResultRoundTripsAndPreservesTies) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1) -> i1 {
        %q0 = qco.alloc : !qco.qubit
        %true = arith.constant true
        %false = arith.constant false
        %flag, %q1 = qco.if %condition args(%arg0 = %q0)
            -> (i1, !qco.qubit) {
          %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %true, %q2 : i1, !qco.qubit
        } else args(%arg0 = %q0) {
          qco.yield %false, %arg0 : i1, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %flag : i1
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  IfOp ifOp;
  module->walk([&](IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getClassicalResults().size(), 1);
  ASSERT_EQ(ifOp.getLinearResults().size(), 1);
  EXPECT_TRUE(ifOp.getClassicalResults().front().getType().isInteger(1));
  EXPECT_TRUE(isa<QubitType>(ifOp.getLinearResults().front().getType()));

  auto* conditionOperand = &ifOp->getOpOperand(0);
  auto* qubitOperand = &ifOp->getOpOperand(1);
  EXPECT_FALSE(ifOp.getTiedResult(conditionOperand));
  EXPECT_FALSE(ifOp.getTiedThenBlockArgument(conditionOperand));
  EXPECT_FALSE(ifOp.getTiedElseBlockArgument(conditionOperand));
  EXPECT_EQ(ifOp.getTiedResult(qubitOperand), ifOp.getLinearResults().front());
  EXPECT_EQ(
      ifOp.getTiedQubit(cast<OpResult>(ifOp.getClassicalResults().front())),
      nullptr);
  EXPECT_EQ(ifOp.getTiedQubit(cast<OpResult>(ifOp.getLinearResults().front())),
            qubitOperand);
  EXPECT_EQ(
      ifOp.getTiedThenYieldedValue(ifOp.thenBlock()->getArgument(0))->get(),
      ifOp.thenYield().getTargets().back());
  EXPECT_EQ(
      ifOp.getTiedElseYieldedValue(ifOp.elseBlock()->getArgument(0))->get(),
      ifOp.elseYield().getTargets().back());
  EXPECT_EQ(ifOp.getTiedThenYieldedValue(ifOp.elseBlock()->getArgument(0)),
            nullptr);
  EXPECT_EQ(ifOp.getTiedElseYieldedValue(ifOp.thenBlock()->getArgument(0)),
            nullptr);

  SmallVector<RegionSuccessor> successors;
  ifOp.getSuccessorRegions(RegionBranchPoint(ifOp.thenYield()), successors);
  ASSERT_EQ(successors.size(), 1);
  EXPECT_EQ(ifOp.getSuccessorInputs(successors.front()), ifOp.getResults());

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  auto reparsedModule = parseSourceString<ModuleOp>(printed, context.get());
  ASSERT_TRUE(reparsedModule);
  EXPECT_TRUE(succeeded(verify(*reparsedModule)));
  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reparsedModule.get()));
}

TEST_F(QCOTest, IfOpRejectsMismatchedClassicalYield) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1) -> i1 {
        %q0 = qco.alloc : !qco.qubit
        %true = arith.constant true
        %flag, %q1 = qco.if %condition args(%arg0 = %q0)
            -> (i1, !qco.qubit) {
          qco.yield %arg0 : !qco.qubit
        } else args(%arg0 = %q0) {
          qco.yield %true, %arg0 : i1, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %flag : i1
      }
    }
  )mlir";

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |= StringRef(diagnostic.str())
                                 .contains("must yield 2 values for parent "
                                           "operation but yields 1");
    return success();
  });
  EXPECT_FALSE(parseSourceString<ModuleOp>(mlirCode, context.get()));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST_F(QCOTest, ModifierYieldStillRejectsClassicalValues) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main() {
        %q0 = qco.alloc : !qco.qubit
        %true = arith.constant true
        %q1 = qco.inv (%arg = %q0) {
          qco.yield %true : i1
        } : {!qco.qubit} -> {!qco.qubit}
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  std::string diagnosticMessage;
  ScopedDiagnosticHandler handler(context.get(), [&](Diagnostic& diagnostic) {
    diagnosticMessage += diagnostic.str();
    return success();
  });
  EXPECT_FALSE(parseSourceString<ModuleOp>(mlirCode, context.get()));
  EXPECT_TRUE(StringRef(diagnosticMessage)
                  .contains("'qco.yield' op operand 0 has type 'i1' but parent "
                            "operation expects '!qco.qubit'"))
      << diagnosticMessage;
}

TEST_F(QCOTest, CanonicalizesConstantIfWithClassicalResult) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main() -> i1 {
        %condition = arith.constant true
        %q0 = qco.alloc : !qco.qubit
        %true = arith.constant true
        %false = arith.constant false
        %flag, %q1 = qco.if %condition args(%arg0 = %q0)
            -> (i1, !qco.qubit) {
          %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %true, %q2 : i1, !qco.qubit
        } else args(%arg0 = %q0) {
          qco.yield %false, %arg0 : i1, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %flag : i1
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(module.get())));
  ASSERT_TRUE(succeeded(verify(*module)));

  bool containsIf = false;
  module->walk([&](IfOp) { containsIf = true; });
  EXPECT_FALSE(containsIf);
  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  APInt result;
  ASSERT_TRUE(matchPattern(returnOp.getOperand(0), m_ConstantInt(&result)));
  EXPECT_TRUE(result.isOne());
}

TEST_F(QCOTest, CanonicalizesRedundantClassicalIfResults) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1, %shared: i64) -> (i64, i64, i64) {
        %q0 = qco.alloc : !qco.qubit
        %same, %first, %duplicate, %unused, %q1 =
            qco.if %condition args(%arg0 = %q0)
                -> (i64, i64, i64, i64, !qco.qubit) {
          %value = arith.constant 1 : i64
          %dead = arith.constant 3 : i64
          %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %shared, %value, %value, %dead, %q2
              : i64, i64, i64, i64, !qco.qubit
        } else args(%arg0 = %q0) {
          %value = arith.constant 2 : i64
          %dead = arith.constant 4 : i64
          %q2 = qco.x %arg0 : !qco.qubit -> !qco.qubit
          qco.yield %shared, %value, %value, %dead, %q2
              : i64, i64, i64, i64, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %same, %first, %duplicate : i64, i64, i64
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(module.get())));
  ASSERT_TRUE(succeeded(verify(*module)));

  IfOp ifOp;
  module->walk([&](IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getClassicalResults().size(), 1);
  ASSERT_EQ(ifOp.getLinearResults().size(), 1);
  EXPECT_EQ(ifOp.getProperties().getResultSegmentSizes(),
            ArrayRef<int32_t>({1, 1}));
  for (YieldOp yield : {ifOp.thenYield(), ifOp.elseYield()}) {
    ASSERT_EQ(yield.getTargets().size(), 2);
    EXPECT_EQ(yield.getTargets().front().getType(),
              ifOp.getClassicalResults().front().getType());
    EXPECT_EQ(yield.getTargets().back().getType(),
              ifOp.getLinearResults().front().getType());
  }

  auto main = module->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  ASSERT_EQ(returnOp.getNumOperands(), 3);
  EXPECT_EQ(returnOp.getOperand(0), main.getArgument(1));
  EXPECT_EQ(returnOp.getOperand(1), ifOp.getClassicalResults().front());
  EXPECT_EQ(returnOp.getOperand(2), returnOp.getOperand(1));
}

TEST_F(QCOTest, IndexSwitchParser) {
  // Test IndexSwitch parser
  const char* mlirCode = R"(
      module {
        func.func @main() -> !cbit.reg<3> attributes {mqt.entry_point} {
            %c2 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %c0 = arith.constant 0 : index
            %c3 = arith.constant 3 : index
            %c = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<3>
            %0 = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
            %1 = scf.for %arg0 = %c0 to %c3 step %c1 iter_args(%arg1 = %0) -> (tensor<3x!qco.qubit>) {
            %5 = arith.remui %arg0, %c3 : index
            %out_tensor_9, %result_10 = qtensor.extract %arg1[%arg0] : tensor<3x!qco.qubit>
            %6 = qco.index_switch %5 -> !qco.qubit
            case 0 args(%arg2 = %result_10) {
                %8 = qco.x %arg2 : !qco.qubit -> !qco.qubit
                qco.yield %8 : !qco.qubit
            }
            case 1 args(%arg2 = %result_10) {
                %8 = qco.y %arg2 : !qco.qubit -> !qco.qubit
                qco.yield %8 : !qco.qubit
            }
            case 2 args(%arg2 = %result_10) {
                %8 = qco.x %arg2 : !qco.qubit -> !qco.qubit
                %9 = qco.y %8 : !qco.qubit -> !qco.qubit
                qco.yield %9 : !qco.qubit
            }
            default args(%arg2 = %result_10) {
                qco.yield %arg2 : !qco.qubit
            }
            %7 = qtensor.insert %6 into %out_tensor_9[%arg0] : tensor<3x!qco.qubit>
            scf.yield %7 : tensor<3x!qco.qubit>
            }
            %out_tensor, %result = qtensor.extract %1[%c0] : tensor<3x!qco.qubit>
            %qubit_out, %result_0 = qco.measure %result : !qco.qubit
            cbit.store %result_0, %c[%c0] : !cbit.reg<3>
            %2 = qtensor.insert %qubit_out into %out_tensor[%c0] : tensor<3x!qco.qubit>
            %out_tensor_1, %result_2 = qtensor.extract %2[%c1] : tensor<3x!qco.qubit>
            %qubit_out_3, %result_4 = qco.measure %result_2 : !qco.qubit
            cbit.store %result_4, %c[%c1] : !cbit.reg<3>
            %3 = qtensor.insert %qubit_out_3 into %out_tensor_1[%c1] : tensor<3x!qco.qubit>
            %out_tensor_5, %result_6 = qtensor.extract %3[%c2] : tensor<3x!qco.qubit>
            %qubit_out_7, %result_8 = qco.measure %result_6 : !qco.qubit
            cbit.store %result_8, %c[%c2] : !cbit.reg<3>
            %4 = qtensor.insert %qubit_out_7 into %out_tensor_5[%c2] : tensor<3x!qco.qubit>
            qtensor.dealloc %4 : tensor<3x!qco.qubit>
            return %c : !cbit.reg<3>
        }
    })";

  auto parsed = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(parsed);
  EXPECT_TRUE(verify(*parsed).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(parsed.get()).succeeded());
  EXPECT_TRUE(verify(*parsed).succeeded());

  auto ref = ::mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(nestedForLoopSwitchOp));
  ASSERT_TRUE(ref);
  EXPECT_TRUE(verify(*ref).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(ref.get()).succeeded());
  EXPECT_TRUE(verify(*ref).succeeded());

  EXPECT_TRUE(areModulesEquivalentWithPermutations(parsed.get(), ref.get()));
}

TEST_F(QCOTest, DefaultOnlyIndexSwitchParser) {
  const char* mlirCode = R"(
      module {
        func.func @main(%index: index) -> i1 {
          %q0 = qco.alloc : !qco.qubit
          %q1 = qco.index_switch %index -> !qco.qubit
          default args(%arg0 = %q0) {
            qco.yield %arg0 : !qco.qubit
          }
          %q2, %result = qco.measure %q1 : !qco.qubit
          qco.sink %q2 : !qco.qubit
          return %result : i1
        }
      })";

  auto parsedModule = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(parsedModule);
  EXPECT_TRUE(verify(*parsedModule).succeeded());

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  parsedModule->print(stream);
  stream.flush();

  auto reparsedModule = parseSourceString<ModuleOp>(printed, context.get());
  ASSERT_TRUE(reparsedModule);
  EXPECT_TRUE(verify(*reparsedModule).succeeded());
}

TEST_F(QCOTest, IndexSwitchWithClassicalResultRoundTripsAndPreservesTies) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%index: index) -> i64 {
        %q0 = qco.alloc : !qco.qubit
        %state, %q1 = qco.index_switch %index -> (i64, !qco.qubit)
        case 0 args(%arg0 = %q0) {
          %q2 = qco.h %arg0 : !qco.qubit -> !qco.qubit
          %case = arith.constant 1 : i64
          qco.yield %case, %q2 : i64, !qco.qubit
        }
        default args(%arg0 = %q0) {
          %default = arith.constant 2 : i64
          qco.yield %default, %arg0 : i64, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %state : i64
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));

  IndexSwitchOp switchOp;
  module->walk([&](IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getClassicalResults().size(), 1);
  ASSERT_EQ(switchOp.getLinearResults().size(), 1);
  EXPECT_TRUE(switchOp.getClassicalResults().front().getType().isInteger(64));
  EXPECT_TRUE(isa<QubitType>(switchOp.getLinearResults().front().getType()));

  auto* targetOperand = &switchOp->getOpOperand(1);
  auto* indexOperand = &switchOp->getOpOperand(0);
  EXPECT_FALSE(switchOp.getTiedResult(indexOperand));
  EXPECT_FALSE(switchOp.getTiedCaseBlockArgument(indexOperand, 0));
  EXPECT_FALSE(switchOp.getTiedDefaultBlockArgument(indexOperand));
  EXPECT_EQ(switchOp.getTiedResult(targetOperand),
            switchOp.getLinearResults().front());
  EXPECT_EQ(switchOp.getTiedTarget(
                cast<OpResult>(switchOp.getClassicalResults().front())),
            nullptr);
  EXPECT_EQ(switchOp.getTiedTarget(
                cast<OpResult>(switchOp.getLinearResults().front())),
            targetOperand);
  EXPECT_EQ(
      switchOp
          .getTiedCaseYieldedValue(switchOp.getCaseBlock(0)->getArgument(0), 0)
          ->get(),
      switchOp.getCaseYield(0).getTargets().back());
  EXPECT_EQ(switchOp
                .getTiedDefaultYieldedValue(
                    switchOp.getDefaultBlock()->getArgument(0))
                ->get(),
            switchOp.getDefaultYield().getTargets().back());

  SmallVector<Attribute> operands(switchOp->getNumOperands());
  SmallVector<RegionSuccessor> successors;
  switchOp.getEntrySuccessorRegions(operands, successors);
  ASSERT_EQ(successors.size(), 2);
  for (const RegionSuccessor successor : successors) {
    ASSERT_EQ(switchOp.getSuccessorInputs(successor).size(), 1);
    EXPECT_EQ(switchOp.getEntrySuccessorOperands(successor).front(),
              switchOp.getTargets().front());
  }

  successors.clear();
  switchOp.getSuccessorRegions(RegionBranchPoint(switchOp.getCaseYield(0)),
                               successors);
  ASSERT_EQ(successors.size(), 1);
  EXPECT_EQ(switchOp.getSuccessorInputs(successors.front()),
            switchOp.getResults());

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  auto reparsedModule = parseSourceString<ModuleOp>(printed, context.get());
  ASSERT_TRUE(reparsedModule);
  EXPECT_TRUE(succeeded(verify(*reparsedModule)));
  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(module.get(), reparsedModule.get()));
}

TEST_F(QCOTest, ClassicalYieldOrderAffectsConditionalEquivalence) {
  constexpr StringLiteral ifSource = R"mlir(
    module {
      func.func @main(%condition: i1) -> (i64, i64) {
        %q0 = qco.alloc : !qco.qubit
        %first, %second, %q1 = qco.if %condition args(%arg0 = %q0)
            -> (i64, i64, !qco.qubit) {
          %one = arith.constant 1 : i64
          %two = arith.constant 2 : i64
          qco.yield %one, %two, %arg0 : i64, i64, !qco.qubit
        } else args(%arg0 = %q0) {
          %one = arith.constant 1 : i64
          %two = arith.constant 2 : i64
          qco.yield %one, %two, %arg0 : i64, i64, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %first, %second : i64, i64
      }
    }
  )mlir";
  constexpr StringLiteral switchSource = R"mlir(
    module {
      func.func @main(%index: index) -> (i64, i64) {
        %q0 = qco.alloc : !qco.qubit
        %first, %second, %q1 = qco.index_switch %index
            -> (i64, i64, !qco.qubit)
        case 0 args(%arg0 = %q0) {
          %one = arith.constant 1 : i64
          %two = arith.constant 2 : i64
          qco.yield %one, %two, %arg0 : i64, i64, !qco.qubit
        }
        default args(%arg0 = %q0) {
          %one = arith.constant 1 : i64
          %two = arith.constant 2 : i64
          qco.yield %one, %two, %arg0 : i64, i64, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %first, %second : i64, i64
      }
    }
  )mlir";

  for (const StringRef source : {ifSource, switchSource}) {
    auto lhs = parseSourceString<ModuleOp>(source, context.get());
    auto rhs = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(lhs);
    ASSERT_TRUE(rhs);

    const auto findFirstYield = [](ModuleOp module) {
      YieldOp result;
      module.walk([&](YieldOp candidate) {
        if (!result) {
          result = candidate;
        }
      });
      return result;
    };

    auto rhsYield = findFirstYield(*rhs);
    ASSERT_TRUE(rhsYield);
    SmallVector<Value> operands(rhsYield.getTargets());
    ASSERT_GE(operands.size(), 2);
    std::swap(operands[0], operands[1]);
    rhsYield->setOperands(operands);
    ASSERT_TRUE(succeeded(verify(*rhs)));

    EXPECT_FALSE(areModulesEquivalentWithPermutations(lhs.get(), rhs.get()));

    auto duplicateLhs = parseSourceString<ModuleOp>(source, context.get());
    auto duplicateRhs = parseSourceString<ModuleOp>(source, context.get());
    ASSERT_TRUE(duplicateLhs);
    ASSERT_TRUE(duplicateRhs);
    for (ModuleOp module : {*duplicateLhs, *duplicateRhs}) {
      auto yield = findFirstYield(module);
      ASSERT_TRUE(yield);
      SmallVector<Value> duplicateOperands(yield.getTargets());
      ASSERT_GE(duplicateOperands.size(), 2);
      duplicateOperands[1] = duplicateOperands[0];
      yield->setOperands(duplicateOperands);
      ASSERT_TRUE(succeeded(verify(module)));
    }
    EXPECT_TRUE(areModulesEquivalentWithPermutations(duplicateLhs.get(),
                                                     duplicateRhs.get()));
  }
}

TEST_F(QCOTest, ExtendsMixedResultIndexSwitchTargets) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%index: index) -> i64 {
        %q0 = qco.alloc : !qco.qubit
        %state, %q1 = qco.index_switch %index -> (i64, !qco.qubit)
        case 0 args(%arg0 = %q0) {
          %case = arith.constant 1 : i64
          qco.yield %case, %arg0 : i64, !qco.qubit
        }
        default args(%arg0 = %q0) {
          %default = arith.constant 2 : i64
          qco.yield %default, %arg0 : i64, !qco.qubit
        }
        qco.sink %q1 : !qco.qubit
        return %state : i64
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);
  IndexSwitchOp switchOp;
  module->walk([&](IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);

  IRRewriter rewriter(context.get());
  rewriter.setInsertionPoint(switchOp);
  auto addon = AllocOp::create(rewriter, switchOp.getLoc());
  auto extended =
      switchOp.replaceWithAdditionalTargets(rewriter, addon.getResult());
  rewriter.setInsertionPointAfter(extended);
  SinkOp::create(rewriter, extended.getLoc(),
                 extended.getLinearResults().back());

  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_EQ(extended.getClassicalResults().size(), 1);
  ASSERT_EQ(extended.getLinearResults().size(), 2);
  for (Region* region : extended.getRegions()) {
    ASSERT_EQ(region->getNumArguments(), 2);
    auto yield = cast<YieldOp>(region->front().getTerminator());
    ASSERT_EQ(yield.getNumOperands(), 3);
    EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
  }
}

TEST_F(QCOTest, EquivalentTensorIndexSwitches) {
  const auto build = [&]() {
    QCOProgramBuilder builder(context.get());
    builder.initialize();

    const auto identity = [](ValueRange args) { return llvm::to_vector(args); };
    const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies{
        identity};

    auto tensor = builder.qtensorAlloc(1);
    auto result = builder.qcoIndexSwitch(0, tensor, SmallVector<int64_t>{0},
                                         caseBodies, identity);
    builder.qtensorDealloc(result.front());
    return builder.finalize();
  };

  const auto lhs = build();
  const auto rhs = build();
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  EXPECT_TRUE(verify(*lhs).succeeded());
  EXPECT_TRUE(verify(*rhs).succeeded());
  EXPECT_TRUE(areModulesEquivalentWithPermutations(lhs.get(), rhs.get()));
}

TEST_F(QCOTest, IndexSwitchCaseValuesAffectEquivalence) {
  const auto build = [&](const int64_t caseValue) {
    QCOProgramBuilder builder(context.get());
    builder.initialize();

    const auto identity = [](Value value) { return value; };
    const SmallVector<function_ref<Value(Value)>> caseBodies{identity};

    const auto q0 = builder.allocQubit();
    auto result = builder.qcoIndexSwitch(0, q0, SmallVector<int64_t>{caseValue},
                                         caseBodies, identity);
    builder.sink(result);
    return builder.finalize();
  };

  const auto lhs = build(0);
  const auto rhs = build(1);
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  EXPECT_TRUE(verify(*lhs).succeeded());
  EXPECT_TRUE(verify(*rhs).succeeded());
  EXPECT_FALSE(areModulesEquivalentWithPermutations(lhs.get(), rhs.get()));
}

TEST_F(QCOTest, NonEquivalentTensorIndexSwitches) {
  const auto build = [&](const bool switchFirstTensor) {
    QCOProgramBuilder builder(context.get());
    builder.initialize();

    const auto identity = [](ValueRange args) { return llvm::to_vector(args); };
    const SmallVector<function_ref<SmallVector<Value>(ValueRange)>> caseBodies{
        identity};

    auto first = builder.qtensorAlloc(1);
    auto second = builder.qtensorAlloc(1);
    auto target = switchFirstTensor ? first : second;
    auto untouched = switchFirstTensor ? second : first;
    auto result = builder.qcoIndexSwitch(0, target, SmallVector<int64_t>{0},
                                         caseBodies, identity);
    builder.qtensorDealloc(result.front());
    builder.qtensorDealloc(untouched);
    return builder.finalize();
  };

  const auto lhs = build(true);
  const auto rhs = build(false);
  ASSERT_TRUE(lhs);
  ASSERT_TRUE(rhs);
  EXPECT_TRUE(verify(*lhs).succeeded());
  EXPECT_TRUE(verify(*rhs).succeeded());
  EXPECT_FALSE(areModulesEquivalentWithPermutations(lhs.get(), rhs.get()));
}

TEST_F(QCOTest, IndexSwitchConstantSuccessor) {
  QCOProgramBuilder builder(context.get());
  builder.initialize();

  const auto identity = [](Value value) { return value; };
  const SmallVector<function_ref<Value(Value)>> caseBodies{identity, identity};

  const auto q0 = builder.allocQubit();
  auto result = builder.qcoIndexSwitch(1, q0, SmallVector<int64_t>{0, 1},
                                       caseBodies, identity);
  builder.sink(result);
  [[maybe_unused]] auto module = builder.finalize();

  auto switchOp = result.getDefiningOp<IndexSwitchOp>();
  ASSERT_TRUE(switchOp);

  SmallVector<Attribute> unknownOperands(switchOp->getNumOperands());
  SmallVector<InvocationBounds> unknownBounds;
  switchOp.getRegionInvocationBounds(unknownOperands, unknownBounds);
  ASSERT_EQ(unknownBounds.size(), switchOp->getNumRegions());
  for (const auto& bound : unknownBounds) {
    EXPECT_EQ(bound.getLowerBound(), 0);
    ASSERT_TRUE(bound.getUpperBound().has_value());
    EXPECT_EQ(*bound.getUpperBound(), 1);
  }

  const auto checkConstant = [&](const int64_t value, Region* expectedRegion,
                                 const size_t expectedRegionIndex) {
    SmallVector<Attribute> operands(switchOp->getNumOperands());
    operands.front() = builder.getIndexAttr(value);

    SmallVector<RegionSuccessor> successors;
    switchOp.getEntrySuccessorRegions(operands, successors);
    ASSERT_EQ(successors.size(), 1);
    EXPECT_EQ(successors.front().getSuccessor(), expectedRegion);

    SmallVector<InvocationBounds> bounds;
    switchOp.getRegionInvocationBounds(operands, bounds);
    ASSERT_EQ(bounds.size(), switchOp->getNumRegions());
    for (const auto [index, bound] : llvm::enumerate(bounds)) {
      EXPECT_EQ(bound.getLowerBound(), 0);
      ASSERT_TRUE(bound.getUpperBound().has_value());
      EXPECT_EQ(*bound.getUpperBound(), index == expectedRegionIndex ? 1 : 0);
    }
  };

  checkConstant(0, switchOp.getCaseRegions().data(), 1);
  checkConstant(1, &switchOp.getCaseRegions()[1], 2);
  checkConstant(2, &switchOp.getDefaultRegion(), 0);

  switchOp->setAttr(
      "cases", DenseI64ArrayAttr::get(context.get(), ArrayRef<int64_t>{0}));
  ScopedDiagnosticHandler handler(context.get(),
                                  [](Diagnostic&) { return success(); });
  EXPECT_TRUE(switchOp.verify().failed());

  switchOp->setAttr(
      "cases", DenseI64ArrayAttr::get(context.get(), ArrayRef<int64_t>{0, 0}));
  EXPECT_TRUE(switchOp.verify().failed());
}

TEST_F(QCOTest, CanonicalizesConstantIndexSwitchToSelectedCaseOrDefault) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @selected_case() -> i64 {
        %c1 = arith.constant 1 : index
        %q0 = qco.alloc : !qco.qubit
        %number, %q1 = qco.index_switch %c1 -> (i64, !qco.qubit)
        case 1 args(%arg0 = %q0) {
          %caseQubit = qco.x %arg0 : !qco.qubit -> !qco.qubit
          %caseNumber = arith.constant 11 : i64
          qco.yield %caseNumber, %caseQubit : i64, !qco.qubit
        }
        default args(%arg0 = %q0) {
          %defaultQubit = qco.z %arg0 : !qco.qubit -> !qco.qubit
          %defaultNumber = arith.constant 12 : i64
          qco.yield %defaultNumber, %defaultQubit : i64, !qco.qubit
        }
        %q2 = qco.h %q1 : !qco.qubit -> !qco.qubit
        qco.sink %q2 : !qco.qubit
        return %number : i64
      }

      func.func @selected_default() -> i64 {
        %c7 = arith.constant 7 : index
        %q0 = qco.alloc : !qco.qubit
        %number, %q1 = qco.index_switch %c7 -> (i64, !qco.qubit)
        case 1 args(%arg0 = %q0) {
          %caseQubit = qco.x %arg0 : !qco.qubit -> !qco.qubit
          %caseNumber = arith.constant 21 : i64
          qco.yield %caseNumber, %caseQubit : i64, !qco.qubit
        }
        default args(%arg0 = %q0) {
          %defaultQubit = qco.z %arg0 : !qco.qubit -> !qco.qubit
          %defaultNumber = arith.constant 22 : i64
          qco.yield %defaultNumber, %defaultQubit : i64, !qco.qubit
        }
        %q2 = qco.h %q1 : !qco.qubit -> !qco.qubit
        qco.sink %q2 : !qco.qubit
        return %number : i64
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(mlirCode, context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(module.get())));
  ASSERT_TRUE(succeeded(verify(*module)));

  bool containsSwitch = false;
  module->walk([&](IndexSwitchOp) { containsSwitch = true; });
  EXPECT_FALSE(containsSwitch);

  const auto checkSelectedRegion = [&](const StringRef functionName,
                                       const int64_t expectedNumber,
                                       const StringRef expectedGate) {
    auto func = module->lookupSymbol<func::FuncOp>(functionName);
    ASSERT_TRUE(func);

    HOp consumer;
    func->walk([&](HOp candidate) { consumer = candidate; });
    ASSERT_TRUE(consumer);
    Value input = cast<UnitaryOpInterface>(consumer.getOperation())
                      .getInputQubits()
                      .front();
    ASSERT_TRUE(input.getDefiningOp());
    EXPECT_EQ(input.getDefiningOp()->getName().getStringRef(), expectedGate);

    auto returnOp = cast<func::ReturnOp>(func.getBody().back().back());
    APInt number;
    ASSERT_TRUE(matchPattern(returnOp.getOperand(0), m_ConstantInt(&number)));
    EXPECT_EQ(number.getSExtValue(), expectedNumber);
  };

  checkSelectedRegion("selected_case", 11, "qco.x");
  checkSelectedRegion("selected_default", 22, "qco.z");
}

/// \name QCO/SCF/IfOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIfOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SimpleIf", MQT_NAMED_BUILDER(simpleIf),
                    MQT_NAMED_BUILDER(simpleIf)},
        QCOTestCase{"TwoQubitIf", MQT_NAMED_BUILDER(ifTwoQubits),
                    MQT_NAMED_BUILDER(ifTwoQubits)},
        QCOTestCase{"IfElse", MQT_NAMED_BUILDER(ifElse),
                    MQT_NAMED_BUILDER(ifElse)},
        QCOTestCase{"OneTensorIf", MQT_NAMED_BUILDER(ifOneTensor),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"ConstantTrueIf", MQT_NAMED_BUILDER(constantTrueIf),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"ConstantFalseIf", MQT_NAMED_BUILDER(constantFalseIf),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"NestedTrueIf", MQT_NAMED_BUILDER(nestedTrueIf),
                    MQT_NAMED_BUILDER(simpleIf)},
        QCOTestCase{"NestedFalseIf", MQT_NAMED_BUILDER(nestedFalseIf),
                    MQT_NAMED_BUILDER(ifElse)}));
/// @}

/// \name QCO/Modifiers/CtrlOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOCtrlOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"TrivialCtrl", MQT_NAMED_BUILDER(trivialCtrl),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"EmptyCtrl", MQT_NAMED_BUILDER(emptyCtrl),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"NestedCtrl", MQT_NAMED_BUILDER(nestedCtrl),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"TripleNestedCtrl", MQT_NAMED_BUILDER(tripleNestedCtrl),
                    MQT_NAMED_BUILDER(tripleControlledRxx)},
        QCOTestCase{"CtrlInvSandwich", MQT_NAMED_BUILDER(ctrlInvSandwich),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"DoubleNestedCtrlTwoQubits",
                    MQT_NAMED_BUILDER(doubleNestedCtrlTwoQubits),
                    MQT_NAMED_BUILDER(fourControlledRxx)},
        QCOTestCase{"NestedCtrlTwo", MQT_NAMED_BUILDER(nestedCtrlTwo),
                    MQT_NAMED_BUILDER(ctrlTwo)},
        QCOTestCase{"ModifierBodyReuseReordered",
                    MQT_NAMED_BUILDER(modifierBodyReuseReordered),
                    MQT_NAMED_BUILDER(modifierBodyReuseReorderedRef)}));
/// @}

/// \name QCO/Modifiers/InvOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOInvOpTest, QCOTest,
    testing::Values(QCOTestCase{"EmptyInv", MQT_NAMED_BUILDER(emptyInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"NestedInv", MQT_NAMED_BUILDER(nestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"TripleNestedInv",
                                MQT_NAMED_BUILDER(tripleNestedInv),
                                MQT_NAMED_BUILDER(rxx)},
                    QCOTestCase{"InvControlSandwich",
                                MQT_NAMED_BUILDER(invCtrlSandwich),
                                MQT_NAMED_BUILDER(singleControlledRxx)},
                    QCOTestCase{"InvCtrlTwo", MQT_NAMED_BUILDER(invCtrlTwo),
                                MQT_NAMED_BUILDER(ctrlInvTwo)},
                    QCOTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                                MQT_NAMED_BUILDER(tdg)}));
/// @}

/// A power modifier with a qubit that its body does not use.
static Value powWithUnusedQubit(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto powOut =
      b.pow(2.0, {q[0], q[1]}, [&](ValueRange qubits) -> SmallVector<Value> {
        return {b.id(qubits[0]), qubits[1]};
      });
  return measureRegister(b, powOut);
}

/// \name QCO/Modifiers/PowOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPowOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Pow1Inline", MQT_NAMED_BUILDER(pow1Inline),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"Pow0Erase", MQT_NAMED_BUILDER(pow0Erase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"Pow0Two", MQT_NAMED_BUILDER(pow0Two),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"EmptyPow", MQT_NAMED_BUILDER(emptyPow),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"NestedPow", MQT_NAMED_BUILDER(nestedPow),
                    MQT_NAMED_BUILDER(powSingleExponent)},
        QCOTestCase{"NegPowRx", MQT_NAMED_BUILDER(negPowRx),
                    MQT_NAMED_BUILDER(powRxNeg)},
        QCOTestCase{"InvPowRx", MQT_NAMED_BUILDER(invPowRx),
                    MQT_NAMED_BUILDER(powRxNeg)},
        QCOTestCase{"InvPowReordered", MQT_NAMED_BUILDER(invPowReordered),
                    MQT_NAMED_BUILDER(invPowReorderedRef)},
        QCOTestCase{"MergeNestedPowReordered",
                    MQT_NAMED_BUILDER(mergeNestedPowReordered),
                    MQT_NAMED_BUILDER(mergeNestedPowReorderedRef)},
        QCOTestCase{"PowCtrlRx", MQT_NAMED_BUILDER(powCtrlRx),
                    MQT_NAMED_BUILDER(ctrlPowRx)},
        QCOTestCase{"NegPowInvIswap", MQT_NAMED_BUILDER(negPowInvIswap),
                    MQT_NAMED_BUILDER(negPowInvIswapRef)},
        QCOTestCase{"InvPowHFrac", MQT_NAMED_BUILDER(invPowHFrac),
                    MQT_NAMED_BUILDER(powHFracNeg)},
        QCOTestCase{"InvPowEvenH", MQT_NAMED_BUILDER(invPowEvenH),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"InvPowEvenSwap", MQT_NAMED_BUILDER(invPowEvenSwap),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"InvPowSquaredZ", MQT_NAMED_BUILDER(invPowSquaredZ),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowWithUnusedQubit", MQT_NAMED_BUILDER(powWithUnusedQubit),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

TEST_F(QCOTest, PowExponentIsUnitaryParameter) {
  auto program =
      ::mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(powRxx));
  ASSERT_TRUE(program);

  auto funcOp = cast<func::FuncOp>(program->getBody()->front());
  auto powOp = *funcOp.getBody().getOps<PowOp>().begin();
  auto unitary = cast<UnitaryOpInterface>(powOp.getOperation());
  EXPECT_EQ(unitary.getNumParams(), 1);
  EXPECT_EQ(unitary.getParameter(0), powOp.getExponent());
  ASSERT_EQ(unitary.getParameters().size(), 1);
  EXPECT_EQ(unitary.getParameters().front(), powOp.getExponent());
}

TEST_F(QCOTest, NestedPowAcrossBranchCutDoesNotMerge) {
  auto program = ::mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(nestedPowBranchCut));
  ASSERT_TRUE(program);
  ASSERT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());

  std::size_t powCount = 0;
  std::size_t xCount = 0;
  PowOp remainingPow;
  program->walk([&](PowOp op) {
    ++powCount;
    remainingPow = op;
  });
  program->walk([&](XOp) { ++xCount; });
  EXPECT_EQ(powCount, 1);
  EXPECT_EQ(xCount, 0);
  ASSERT_TRUE(remainingPow);
  const auto matrix = remainingPow.getUnitaryMatrix();
  ASSERT_TRUE(matrix);
  EXPECT_TRUE(matrix->isApprox(DynamicMatrix::identity(2), 1e-10));
}

/// pow(rxx) folds the exponent into the rotation angle: pow(2){rxx(θ)} =>
/// rxx(2θ). Verify cleanup and the hoisted parameter's SSA dominance.
TEST_F(QCOTest, PowRxxFold) {
  auto program =
      ::mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(powRxx));
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  std::size_t powCount = 0;
  SmallVector<RXXOp> rxxOps;
  program->walk([&](PowOp) { ++powCount; });
  program->walk([&](RXXOp op) { rxxOps.push_back(op); });
  EXPECT_EQ(powCount, 0) << "PowOp around rxx should be folded away";
  ASSERT_EQ(rxxOps.size(), 1);

  auto rxx = rxxOps.front();
  FloatAttr angle;
  ASSERT_TRUE(matchPattern(rxx.getTheta(), m_Constant(&angle)));
  EXPECT_NEAR(angle.getValueAsDouble(), 0.246, 1e-12);

  auto* parameterDef = rxx.getTheta().getDefiningOp();
  ASSERT_NE(parameterDef, nullptr);
  const DominanceInfo dominance(program.get());
  EXPECT_TRUE(dominance.properlyDominates(parameterDef, rxx.getOperation()));
}

static Value powRzxWithReorderedBody(QCOProgramBuilder& builder) {
  auto qubits = builder.allocQubitRegister(2);
  auto powOut = builder.pow(
      2.0, qubits.qubits, [&](ValueRange args) -> SmallVector<Value> {
        auto [out1, out0] = builder.rzx(0.123, args[1], args[0]);
        return {out0, out1};
      });
  return measureRegister(builder, powOut);
}

static Value powBarrierWithReorderedBody(QCOProgramBuilder& builder) {
  auto q0 = builder.allocQubit();
  auto q1 = builder.allocQubit();
  auto powOut =
      builder.pow(2.0, {q0, q1}, [&](ValueRange args) -> SmallVector<Value> {
        auto barrierOut = builder.barrier({args[1], args[0]});
        return {barrierOut[1], barrierOut[0]};
      });
  return measureRegister(builder, powOut);
}

static Value powEvenSwapWithReorderedBody(QCOProgramBuilder& builder) {
  auto q0 = builder.allocQubit();
  auto q1 = builder.allocQubit();
  auto powOut =
      builder.pow(2.0, {q0, q1}, [&](ValueRange args) -> SmallVector<Value> {
        auto [out1, out0] = builder.swap(args[1], args[0]);
        return {out1, out0};
      });
  return measureRegister(builder, powOut);
}

TEST_F(QCOTest, PowGateFoldPreservesReorderedBodyResults) {
  auto program = ::mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(powRzxWithReorderedBody));
  ASSERT_TRUE(program);
  ASSERT_TRUE(succeeded(verify(*program)));

  PassManager pm(context.get());
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*program)));
  ASSERT_TRUE(succeeded(verify(*program)));

  SmallVector<RZXOp> gates;
  SmallVector<MeasureOp> measurements;
  program->walk([&](RZXOp gate) { gates.push_back(gate); });
  program->walk([&](MeasureOp measure) { measurements.push_back(measure); });
  ASSERT_EQ(gates.size(), 1);
  ASSERT_EQ(measurements.size(), 2);
  EXPECT_EQ(measurements[0].getQubitIn(), gates[0].getOutputTarget(1));
  EXPECT_EQ(measurements[1].getQubitIn(), gates[0].getOutputTarget(0));
}

TEST_F(QCOTest, PowBarrierFoldPreservesReorderedBodyResults) {
  auto program = ::mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(powBarrierWithReorderedBody));
  ASSERT_TRUE(program);
  ASSERT_TRUE(succeeded(verify(*program)));

  PassManager pm(context.get());
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*program)));
  ASSERT_TRUE(succeeded(verify(*program)));

  SmallVector<AllocOp> allocations;
  SmallVector<BarrierOp> barriers;
  SmallVector<MeasureOp> measurements;
  program->walk([&](AllocOp alloc) { allocations.push_back(alloc); });
  program->walk([&](BarrierOp barrier) { barriers.push_back(barrier); });
  program->walk([&](MeasureOp measure) { measurements.push_back(measure); });
  ASSERT_EQ(allocations.size(), 2);
  ASSERT_EQ(barriers.size(), 1);
  ASSERT_EQ(measurements.size(), 2);
  EXPECT_EQ(barriers[0].getQubitsIn()[0], allocations[1].getResult());
  EXPECT_EQ(barriers[0].getQubitsIn()[1], allocations[0].getResult());
  EXPECT_EQ(measurements[0].getQubitIn(), barriers[0].getOutputQubits()[1]);
  EXPECT_EQ(measurements[1].getQubitIn(), barriers[0].getOutputQubits()[0]);
}

TEST_F(QCOTest, EvenPowFoldPreservesReorderedBodyResults) {
  auto program = ::mqt::test::buildMLIRProgram(
      context.get(), MQT_NAMED_BUILDER(powEvenSwapWithReorderedBody));
  ASSERT_TRUE(program);
  ASSERT_TRUE(succeeded(verify(*program)));

  PassManager pm(context.get());
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*program)));
  ASSERT_TRUE(succeeded(verify(*program)));

  SmallVector<AllocOp> allocations;
  SmallVector<MeasureOp> measurements;
  program->walk([&](AllocOp alloc) { allocations.push_back(alloc); });
  program->walk([&](MeasureOp measure) { measurements.push_back(measure); });
  ASSERT_EQ(allocations.size(), 2);
  ASSERT_EQ(measurements.size(), 2);
  EXPECT_EQ(measurements[0].getQubitIn(), allocations[1].getResult());
  EXPECT_EQ(measurements[1].getQubitIn(), allocations[0].getResult());
}

/// pow(-0.5) { h } cannot fold a negative fractional exponent
/// into H (no angle to scale). Verify that PowOp survives.
TEST_F(QCOTest, NegPowHNoFold) {
  auto program =
      ::mqt::test::buildMLIRProgram(context.get(), MQT_NAMED_BUILDER(negPowH));
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  int powCount = 0;
  program->walk([&](PowOp) { ++powCount; });
  EXPECT_EQ(powCount, 1) << "PowOp around h must survive the pipeline";
}

/// pow(sx) inside a ctrl modifier expands into GPhase + RX. Global-phase
/// normalization then turns the controlled GPhase into P on the control.
/// Verify the CtrlOp survives and the relative phase remains observable.
TEST_F(QCOTest, CtrlPowSxExpands) {
  auto program = ::mqt::test::buildMLIRProgram(context.get(),
                                               MQT_NAMED_BUILDER(ctrlPowSx));
  ASSERT_TRUE(program);
  EXPECT_TRUE(verify(*program).succeeded());
  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  EXPECT_TRUE(verify(*program).succeeded());

  int ctrlCount = 0;
  int powCount = 0;
  int gphaseCount = 0;
  int pCount = 0;
  int rxCount = 0;
  program->walk([&](CtrlOp) { ++ctrlCount; });
  program->walk([&](PowOp) { ++powCount; });
  program->walk([&](GPhaseOp) { ++gphaseCount; });
  program->walk([&](POp) { ++pCount; });
  program->walk([&](RXOp) { ++rxCount; });
  EXPECT_EQ(ctrlCount, 1) << "CtrlOp must survive the pipeline";
  EXPECT_EQ(powCount, 0) << "PowOp inside ctrl must be expanded";
  EXPECT_EQ(gphaseCount, 0) << "controlled GPhase must be extracted";
  EXPECT_EQ(pCount, 1) << "controlled GPhase must become P on the control";
  EXPECT_EQ(rxCount, 1) << "SX fold must emit an RX";
}

TEST_F(QCOTest, CtrlGPhasePassesTargetsThrough) {
  auto program = QCOProgramBuilder::build(context.get(), [&](auto& builder) {
    auto controlIn = builder.staticQubit(0);
    auto targetIn = builder.staticQubit(1);
    auto [control, target] =
        builder.ctrl(controlIn, targetIn, [&](Value targetArg) {
          builder.gphase(0.123);
          return targetArg;
        });
    return SmallVector<Value>{control, target};
  });
  ASSERT_TRUE(program);

  ASSERT_TRUE(runQCOCleanupPipeline(*program).succeeded());
  ASSERT_TRUE(verify(*program).succeeded());

  auto mainFunc = *program->getOps<func::FuncOp>().begin();
  auto returnOp = cast<func::ReturnOp>(mainFunc.getBody().front().back());
  ASSERT_EQ(returnOp.getNumOperands(), 2U);
  EXPECT_TRUE(returnOp.getOperand(0).getDefiningOp<POp>());
  EXPECT_TRUE(returnOp.getOperand(1).getDefiningOp<StaticOp>());
  EXPECT_TRUE(mainFunc.getBody().getOps<CtrlOp>().empty());
}

/// \name QCO/Operations/StandardGates/BarrierOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOBarrierOpTest, QCOTest,
    testing::Values(QCOTestCase{"Barrier", MQT_NAMED_BUILDER(barrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"BarrierTwoQubits",
                                MQT_NAMED_BUILDER(barrierTwoQubits),
                                MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCOTestCase{"BarrierMultipleQubits",
                                MQT_NAMED_BUILDER(barrierMultipleQubits),
                                MQT_NAMED_BUILDER(barrierMultipleQubits)},
                    QCOTestCase{"SingleControlledBarrier",
                                MQT_NAMED_BUILDER(singleControlledBarrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"InverseBarrier",
                                MQT_NAMED_BUILDER(inverseBarrier),
                                MQT_NAMED_BUILDER(barrier)},
                    QCOTestCase{"TwoBarrier", MQT_NAMED_BUILDER(twoBarrier),
                                MQT_NAMED_BUILDER(barrierTwoQubits)},
                    QCOTestCase{"PowBarrier", MQT_NAMED_BUILDER(powBarrier),
                                MQT_NAMED_BUILDER(barrier)}));
/// @}

/// \name QCO/Operations/StandardGates/DcxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCODCXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"DCX", MQT_NAMED_BUILDER(dcx), MQT_NAMED_BUILDER(dcx)},
        QCOTestCase{"SingleControlledDCX",
                    MQT_NAMED_BUILDER(singleControlledDcx),
                    MQT_NAMED_BUILDER(singleControlledDcx)},
        QCOTestCase{"MultipleControlledDCX",
                    MQT_NAMED_BUILDER(multipleControlledDcx),
                    MQT_NAMED_BUILDER(multipleControlledDcx)},
        QCOTestCase{"NestedControlledDCX",
                    MQT_NAMED_BUILDER(nestedControlledDcx),
                    MQT_NAMED_BUILDER(multipleControlledDcx)},
        QCOTestCase{"TrivialControlledDCX",
                    MQT_NAMED_BUILDER(trivialControlledDcx),
                    MQT_NAMED_BUILDER(dcx)},
        QCOTestCase{"InverseDCX", MQT_NAMED_BUILDER(inverseDcx),
                    MQT_NAMED_BUILDER(inverseDcx)},
        QCOTestCase{"InverseMultipleControlledDCX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledDcx),
                    MQT_NAMED_BUILDER(inverseMultipleControlledDcx)},
        QCOTestCase{"TwoDCX", MQT_NAMED_BUILDER(twoDcx),
                    MQT_NAMED_BUILDER(twoDcx)},
        QCOTestCase{"TwoDCXSwappedTargets",
                    MQT_NAMED_BUILDER(twoDcxSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/EcrOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOECROpTest, QCOTest,
    testing::Values(QCOTestCase{"ECR", MQT_NAMED_BUILDER(ecr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"SingleControlledECR",
                                MQT_NAMED_BUILDER(singleControlledEcr),
                                MQT_NAMED_BUILDER(singleControlledEcr)},
                    QCOTestCase{"MultipleControlledECR",
                                MQT_NAMED_BUILDER(multipleControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"NestedControlledECR",
                                MQT_NAMED_BUILDER(nestedControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"TrivialControlledECR",
                                MQT_NAMED_BUILDER(trivialControlledEcr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"InverseECR", MQT_NAMED_BUILDER(inverseEcr),
                                MQT_NAMED_BUILDER(ecr)},
                    QCOTestCase{"InverseMultipleControlledECR",
                                MQT_NAMED_BUILDER(inverseMultipleControlledEcr),
                                MQT_NAMED_BUILDER(multipleControlledEcr)},
                    QCOTestCase{"TwoECR", MQT_NAMED_BUILDER(twoEcr),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)},
                    QCOTestCase{"PowEvenECR", MQT_NAMED_BUILDER(powEvenEcr),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)},
                    QCOTestCase{"PowOddECR", MQT_NAMED_BUILDER(powOddEcr),
                                MQT_NAMED_BUILDER(ecr)}));
/// @}

/// \name QCO/Operations/StandardGates/GphaseOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOGPhaseOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"GlobalPhase", MQT_NAMED_BUILDER(globalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"SingleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(singleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"MultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"InverseGlobalPhase", MQT_NAMED_BUILDER(inverseGlobalPhase),
                    MQT_NAMED_BUILDER(globalPhase)},
        QCOTestCase{"InverseMultipleControlledGlobalPhase",
                    MQT_NAMED_BUILDER(inverseMultipleControlledGlobalPhase),
                    MQT_NAMED_BUILDER(multipleControlledGlobalPhase)},
        QCOTestCase{"PowGphaseScaled", MQT_NAMED_BUILDER(powGphaseScaled),
                    MQT_NAMED_BUILDER(powGphaseScaledRef)},
        QCOTestCase{"NegPowGphase", MQT_NAMED_BUILDER(negPowGphase),
                    MQT_NAMED_BUILDER(negPowGphaseRef)}));
/// @}

/// \name QCO/Operations/StandardGates/HOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOHOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"H", MQT_NAMED_BUILDER(h), MQT_NAMED_BUILDER(h)},
        QCOTestCase{"SingleControlledH", MQT_NAMED_BUILDER(singleControlledH),
                    MQT_NAMED_BUILDER(singleControlledH)},
        QCOTestCase{"MultipleControlledH",
                    MQT_NAMED_BUILDER(multipleControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"NestedControlledH", MQT_NAMED_BUILDER(nestedControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"TrivialControlledH", MQT_NAMED_BUILDER(trivialControlledH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"InverseH", MQT_NAMED_BUILDER(inverseH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"InverseMultipleControlledH",
                    MQT_NAMED_BUILDER(inverseMultipleControlledH),
                    MQT_NAMED_BUILDER(multipleControlledH)},
        QCOTestCase{"TwoH", MQT_NAMED_BUILDER(twoH),
                    MQT_NAMED_BUILDER(allocQubit)},
        QCOTestCase{"PowEvenH", MQT_NAMED_BUILDER(powEvenH),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowOddH", MQT_NAMED_BUILDER(powOddH),
                    MQT_NAMED_BUILDER(h)}));
/// @}

/// \name QCO/Operations/StandardGates/IdOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOIDOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                    MQT_NAMED_BUILDER(allocQubit)},
        QCOTestCase{"SingleControlledIdentity",
                    MQT_NAMED_BUILDER(singleControlledIdentity),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"MultipleControlledIdentity",
                    MQT_NAMED_BUILDER(multipleControlledIdentity),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"NestedControlledIdentity",
                    MQT_NAMED_BUILDER(nestedControlledIdentity),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"TrivialControlledIdentity",
                    MQT_NAMED_BUILDER(trivialControlledIdentity),
                    MQT_NAMED_BUILDER(allocQubit)},
        QCOTestCase{"InverseIdentity", MQT_NAMED_BUILDER(inverseIdentity),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"InverseMultipleControlledIdentity",
                    MQT_NAMED_BUILDER(inverseMultipleControlledIdentity),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"PowId", MQT_NAMED_BUILDER(powId),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/IswapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOiSWAPOpTest, QCOTest,
    testing::Values(QCOTestCase{"iSWAP", MQT_NAMED_BUILDER(iswap),
                                MQT_NAMED_BUILDER(iswap)},
                    QCOTestCase{"SingleControllediSWAP",
                                MQT_NAMED_BUILDER(singleControlledIswap),
                                MQT_NAMED_BUILDER(singleControlledIswap)},
                    QCOTestCase{"MultipleControllediSWAP",
                                MQT_NAMED_BUILDER(multipleControlledIswap),
                                MQT_NAMED_BUILDER(multipleControlledIswap)},
                    QCOTestCase{"NestedControllediSWAP",
                                MQT_NAMED_BUILDER(nestedControlledIswap),
                                MQT_NAMED_BUILDER(multipleControlledIswap)},
                    QCOTestCase{"TrivialControllediSWAP",
                                MQT_NAMED_BUILDER(trivialControlledIswap),
                                MQT_NAMED_BUILDER(iswap)},
                    QCOTestCase{"InverseiSWAP", MQT_NAMED_BUILDER(inverseIswap),
                                MQT_NAMED_BUILDER(inverseIswap)},
                    QCOTestCase{
                        "InverseMultipleControllediSWAP",
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap),
                        MQT_NAMED_BUILDER(inverseMultipleControlledIswap)},
                    QCOTestCase{"PowHalfiSWAP", MQT_NAMED_BUILDER(powHalfIswap),
                                MQT_NAMED_BUILDER(powHalfIswapRef)}));
/// @}

/// \name QCO/Operations/StandardGates/POp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOPOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"P", MQT_NAMED_BUILDER(p), MQT_NAMED_BUILDER(p)},
        QCOTestCase{"SingleControlledP", MQT_NAMED_BUILDER(singleControlledP),
                    MQT_NAMED_BUILDER(singleControlledP)},
        QCOTestCase{"MultipleControlledP",
                    MQT_NAMED_BUILDER(multipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"NestedControlledP", MQT_NAMED_BUILDER(nestedControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"TrivialControlledP", MQT_NAMED_BUILDER(trivialControlledP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"InverseP", MQT_NAMED_BUILDER(inverseP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"InverseMultipleControlledP",
                    MQT_NAMED_BUILDER(inverseMultipleControlledP),
                    MQT_NAMED_BUILDER(multipleControlledP)},
        QCOTestCase{"TwoPOppositePhase", MQT_NAMED_BUILDER(twoPOppositePhase),
                    MQT_NAMED_BUILDER(allocQubit)}));
/// @}

/// \name QCO/Operations/StandardGates/RCCXOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORCCXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RCCX", MQT_NAMED_BUILDER(rccx), MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"SingleControlledRCCX",
                    MQT_NAMED_BUILDER(singleControlledRccx),
                    MQT_NAMED_BUILDER(singleControlledRccx)},
        QCOTestCase{"MultipleControlledRCCX",
                    MQT_NAMED_BUILDER(multipleControlledRccx),
                    MQT_NAMED_BUILDER(multipleControlledRccx)},
        QCOTestCase{"NestedControlledRCCX",
                    MQT_NAMED_BUILDER(nestedControlledRccx),
                    MQT_NAMED_BUILDER(multipleControlledRccx)},
        QCOTestCase{"TrivialControlledRCCX",
                    MQT_NAMED_BUILDER(trivialControlledRccx),
                    MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"InverseRCCX", MQT_NAMED_BUILDER(inverseRccx),
                    MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"PowEvenRCCX", MQT_NAMED_BUILDER(powEvenRccx),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)},
        QCOTestCase{"PowOddRCCX", MQT_NAMED_BUILDER(powOddRccx),
                    MQT_NAMED_BUILDER(rccx)},
        QCOTestCase{"InverseMultipleControlledRCCX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRccx),
                    MQT_NAMED_BUILDER(multipleControlledRccx)},
        QCOTestCase{"TwoRCCX", MQT_NAMED_BUILDER(twoRccx),
                    MQT_NAMED_BUILDER(alloc3QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/ROp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOROpTest, QCOTest,
    testing::Values(
        QCOTestCase{"R", MQT_NAMED_BUILDER(r), MQT_NAMED_BUILDER(r)},
        QCOTestCase{"SingleControlledR", MQT_NAMED_BUILDER(singleControlledR),
                    MQT_NAMED_BUILDER(singleControlledR)},
        QCOTestCase{"MultipleControlledR",
                    MQT_NAMED_BUILDER(multipleControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"NestedControlledR", MQT_NAMED_BUILDER(nestedControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"TrivialControlledR", MQT_NAMED_BUILDER(trivialControlledR),
                    MQT_NAMED_BUILDER(r)},
        QCOTestCase{"InverseR", MQT_NAMED_BUILDER(inverseR),
                    MQT_NAMED_BUILDER(r)},
        QCOTestCase{"InverseMultipleControlledR",
                    MQT_NAMED_BUILDER(inverseMultipleControlledR),
                    MQT_NAMED_BUILDER(multipleControlledR)},
        QCOTestCase{"CanonicalizeRToRx", MQT_NAMED_BUILDER(canonicalizeRToRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"CanonicalizeRToRy", MQT_NAMED_BUILDER(canonicalizeRToRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"TwoR", MQT_NAMED_BUILDER(twoR), MQT_NAMED_BUILDER(r)},
        QCOTestCase{"PowRScaled", MQT_NAMED_BUILDER(powRScaled),
                    MQT_NAMED_BUILDER(powRScaledRef)}));
/// @}

/// \name QCO/Operations/StandardGates/RxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RX", MQT_NAMED_BUILDER(rx), MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"SingleControlledRX", MQT_NAMED_BUILDER(singleControlledRx),
                    MQT_NAMED_BUILDER(singleControlledRx)},
        QCOTestCase{"MultipleControlledRX",
                    MQT_NAMED_BUILDER(multipleControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"NestedControlledRX", MQT_NAMED_BUILDER(nestedControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"TrivialControlledRX",
                    MQT_NAMED_BUILDER(trivialControlledRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"InverseRX", MQT_NAMED_BUILDER(inverseRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"InverseMultipleControlledRX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRx),
                    MQT_NAMED_BUILDER(multipleControlledRx)},
        QCOTestCase{"TwoRXOppositePhase", MQT_NAMED_BUILDER(twoRxOppositePhase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowRxScaled", MQT_NAMED_BUILDER(powRxScaled),
                    MQT_NAMED_BUILDER(rxScaled)}));
/// @}

/// \name QCO/Operations/StandardGates/RxxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORXXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RXX", MQT_NAMED_BUILDER(rxx), MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"SingleControlledRXX",
                    MQT_NAMED_BUILDER(singleControlledRxx),
                    MQT_NAMED_BUILDER(singleControlledRxx)},
        QCOTestCase{"MultipleControlledRXX",
                    MQT_NAMED_BUILDER(multipleControlledRxx),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"NestedControlledRXX",
                    MQT_NAMED_BUILDER(nestedControlledRxx),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"TrivialControlledRXX",
                    MQT_NAMED_BUILDER(trivialControlledRxx),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"InverseRXX", MQT_NAMED_BUILDER(inverseRxx),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"InverseMultipleControlledRXX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRxx),
                    MQT_NAMED_BUILDER(multipleControlledRxx)},
        QCOTestCase{"TwoRXX", MQT_NAMED_BUILDER(twoRxx),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"TwoRXXSwappedTargets",
                    MQT_NAMED_BUILDER(twoRxxSwappedTargets),
                    MQT_NAMED_BUILDER(rxx)},
        QCOTestCase{"TwoRXXOppositePhase",
                    MQT_NAMED_BUILDER(twoRxxOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoRXXOppositePhaseSwappedTargets",
                    MQT_NAMED_BUILDER(twoRxxOppositePhaseSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RY", MQT_NAMED_BUILDER(ry), MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"SingleControlledRY", MQT_NAMED_BUILDER(singleControlledRy),
                    MQT_NAMED_BUILDER(singleControlledRy)},
        QCOTestCase{"MultipleControlledRY",
                    MQT_NAMED_BUILDER(multipleControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"NestedControlledRY", MQT_NAMED_BUILDER(nestedControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"TrivialControlledRY",
                    MQT_NAMED_BUILDER(trivialControlledRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"InverseRY", MQT_NAMED_BUILDER(inverseRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"InverseMultipleControlledRY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRy),
                    MQT_NAMED_BUILDER(multipleControlledRy)},
        QCOTestCase{"TwoRYOppositePhase", MQT_NAMED_BUILDER(twoRyOppositePhase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RyyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RYY", MQT_NAMED_BUILDER(ryy), MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"SingleControlledRYY",
                    MQT_NAMED_BUILDER(singleControlledRyy),
                    MQT_NAMED_BUILDER(singleControlledRyy)},
        QCOTestCase{"MultipleControlledRYY",
                    MQT_NAMED_BUILDER(multipleControlledRyy),
                    MQT_NAMED_BUILDER(multipleControlledRyy)},
        QCOTestCase{"NestedControlledRYY",
                    MQT_NAMED_BUILDER(nestedControlledRyy),
                    MQT_NAMED_BUILDER(multipleControlledRyy)},
        QCOTestCase{"TrivialControlledRYY",
                    MQT_NAMED_BUILDER(trivialControlledRyy),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"InverseRYY", MQT_NAMED_BUILDER(inverseRyy),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"InverseMultipleControlledRYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRyy),
                    MQT_NAMED_BUILDER(multipleControlledRyy)},
        QCOTestCase{"TwoRYY", MQT_NAMED_BUILDER(twoRyy),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"TwoRYYSwappedTargets",
                    MQT_NAMED_BUILDER(twoRyySwappedTargets),
                    MQT_NAMED_BUILDER(ryy)},
        QCOTestCase{"TwoRYYOppositePhaseSwappedTargets",
                    MQT_NAMED_BUILDER(twoRyyOppositePhaseSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoRYYOppositePhase",
                    MQT_NAMED_BUILDER(twoRyyOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RZ", MQT_NAMED_BUILDER(rz), MQT_NAMED_BUILDER(rz)},
        QCOTestCase{"SingleControlledRZ", MQT_NAMED_BUILDER(singleControlledRz),
                    MQT_NAMED_BUILDER(singleControlledRz)},
        QCOTestCase{"MultipleControlledRZ",
                    MQT_NAMED_BUILDER(multipleControlledRz),
                    MQT_NAMED_BUILDER(multipleControlledRz)},
        QCOTestCase{"NestedControlledRZ", MQT_NAMED_BUILDER(nestedControlledRz),
                    MQT_NAMED_BUILDER(multipleControlledRz)},
        QCOTestCase{"TrivialControlledRZ",
                    MQT_NAMED_BUILDER(trivialControlledRz),
                    MQT_NAMED_BUILDER(rz)},
        QCOTestCase{"InverseRZ", MQT_NAMED_BUILDER(inverseRz),
                    MQT_NAMED_BUILDER(rz)},
        QCOTestCase{"InverseMultipleControlledRZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRz),
                    MQT_NAMED_BUILDER(multipleControlledRz)},
        QCOTestCase{"TwoRZOppositePhase", MQT_NAMED_BUILDER(twoRzOppositePhase),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RzxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZXOpTest, QCOTest,
    testing::Values(QCOTestCase{"RZX", MQT_NAMED_BUILDER(rzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"SingleControlledRZX",
                                MQT_NAMED_BUILDER(singleControlledRzx),
                                MQT_NAMED_BUILDER(singleControlledRzx)},
                    QCOTestCase{"MultipleControlledRZX",
                                MQT_NAMED_BUILDER(multipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"NestedControlledRZX",
                                MQT_NAMED_BUILDER(nestedControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"TrivialControlledRZX",
                                MQT_NAMED_BUILDER(trivialControlledRzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"InverseRZX", MQT_NAMED_BUILDER(inverseRzx),
                                MQT_NAMED_BUILDER(rzx)},
                    QCOTestCase{"InverseMultipleControlledRZX",
                                MQT_NAMED_BUILDER(inverseMultipleControlledRzx),
                                MQT_NAMED_BUILDER(multipleControlledRzx)},
                    QCOTestCase{"TwoRZXOppositePhase",
                                MQT_NAMED_BUILDER(twoRzxOppositePhase),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/RzzOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCORZZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"RZZ", MQT_NAMED_BUILDER(rzz), MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"SingleControlledRZZ",
                    MQT_NAMED_BUILDER(singleControlledRzz),
                    MQT_NAMED_BUILDER(singleControlledRzz)},
        QCOTestCase{"MultipleControlledRZZ",
                    MQT_NAMED_BUILDER(multipleControlledRzz),
                    MQT_NAMED_BUILDER(multipleControlledRzz)},
        QCOTestCase{"NestedControlledRZZ",
                    MQT_NAMED_BUILDER(nestedControlledRzz),
                    MQT_NAMED_BUILDER(multipleControlledRzz)},
        QCOTestCase{"TrivialControlledRZZ",
                    MQT_NAMED_BUILDER(trivialControlledRzz),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"InverseRZZ", MQT_NAMED_BUILDER(inverseRzz),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"InverseMultipleControlledRZZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledRzz),
                    MQT_NAMED_BUILDER(multipleControlledRzz)},
        QCOTestCase{"TwoRZZ", MQT_NAMED_BUILDER(twoRzz),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"TwoRZZSwappedTargets",
                    MQT_NAMED_BUILDER(twoRzzSwappedTargets),
                    MQT_NAMED_BUILDER(rzz)},
        QCOTestCase{"TwoRZZOppositePhaseSwappedTargets",
                    MQT_NAMED_BUILDER(twoRzzOppositePhaseSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoRZZOppositePhase",
                    MQT_NAMED_BUILDER(twoRzzOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)}));
/// @}

/// \name QCO/Operations/StandardGates/SOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"S", MQT_NAMED_BUILDER(s), MQT_NAMED_BUILDER(s)},
        QCOTestCase{"SingleControlledS", MQT_NAMED_BUILDER(singleControlledS),
                    MQT_NAMED_BUILDER(singleControlledS)},
        QCOTestCase{"MultipleControlledS",
                    MQT_NAMED_BUILDER(multipleControlledS),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"NestedControlledS", MQT_NAMED_BUILDER(nestedControlledS),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"TrivialControlledS", MQT_NAMED_BUILDER(trivialControlledS),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"InverseS", MQT_NAMED_BUILDER(inverseS),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"InverseMultipleControlledS",
                    MQT_NAMED_BUILDER(inverseMultipleControlledS),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"SThenSdg", MQT_NAMED_BUILDER(sThenSdg),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoS", MQT_NAMED_BUILDER(twoS), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowTwoS", MQT_NAMED_BUILDER(powTwoS),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowFourSErase", MQT_NAMED_BUILDER(powFourS),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfSToT", MQT_NAMED_BUILDER(powHalfS),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"PowThirdSToP", MQT_NAMED_BUILDER(powThirdS),
                    MQT_NAMED_BUILDER(powThirdSRef)}));
/// @}

/// \name QCO/Operations/StandardGates/SdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Sdg", MQT_NAMED_BUILDER(sdg), MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"SingleControlledSdg",
                    MQT_NAMED_BUILDER(singleControlledSdg),
                    MQT_NAMED_BUILDER(singleControlledSdg)},
        QCOTestCase{"MultipleControlledSdg",
                    MQT_NAMED_BUILDER(multipleControlledSdg),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"NestedControlledSdg",
                    MQT_NAMED_BUILDER(nestedControlledSdg),
                    MQT_NAMED_BUILDER(multipleControlledSdg)},
        QCOTestCase{"TrivialControlledSdg",
                    MQT_NAMED_BUILDER(trivialControlledSdg),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"InverseSdg", MQT_NAMED_BUILDER(inverseSdg),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"InverseMultipleControlledSdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSdg),
                    MQT_NAMED_BUILDER(multipleControlledS)},
        QCOTestCase{"SdgThenS", MQT_NAMED_BUILDER(sdgThenS),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoSdg", MQT_NAMED_BUILDER(twoSdg), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowTwoSdg", MQT_NAMED_BUILDER(powTwoSdg),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"PowHalfSdgToTdg", MQT_NAMED_BUILDER(powHalfSdg),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"PowThirdSdgToP", MQT_NAMED_BUILDER(powThirdSdg),
                    MQT_NAMED_BUILDER(powThirdSdgRef)}));
/// @}

/// \name QCO/Operations/StandardGates/SwapOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSWAPOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SWAP", MQT_NAMED_BUILDER(swap), MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"SingleControlledSWAP",
                    MQT_NAMED_BUILDER(singleControlledSwap),
                    MQT_NAMED_BUILDER(singleControlledSwap)},
        QCOTestCase{"MultipleControlledSWAP",
                    MQT_NAMED_BUILDER(multipleControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"NestedControlledSWAP",
                    MQT_NAMED_BUILDER(nestedControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"TrivialControlledSWAP",
                    MQT_NAMED_BUILDER(trivialControlledSwap),
                    MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"InverseSWAP", MQT_NAMED_BUILDER(inverseSwap),
                    MQT_NAMED_BUILDER(swap)},
        QCOTestCase{"InverseMultipleControlledSWAP",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSwap),
                    MQT_NAMED_BUILDER(multipleControlledSwap)},
        QCOTestCase{"TwoSWAP", MQT_NAMED_BUILDER(twoSwap),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoSWAPSwappedTargets",
                    MQT_NAMED_BUILDER(twoSwapSwappedTargets),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"PowEvenSWAP", MQT_NAMED_BUILDER(powEvenSwap),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"PowOddSWAP", MQT_NAMED_BUILDER(powOddSwap),
                    MQT_NAMED_BUILDER(swap)}));
/// @}

/// \name QCO/Operations/StandardGates/SxOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SX", MQT_NAMED_BUILDER(sx), MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"SingleControlledSX", MQT_NAMED_BUILDER(singleControlledSx),
                    MQT_NAMED_BUILDER(singleControlledSx)},
        QCOTestCase{"MultipleControlledSX",
                    MQT_NAMED_BUILDER(multipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"NestedControlledSX", MQT_NAMED_BUILDER(nestedControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"TrivialControlledSX",
                    MQT_NAMED_BUILDER(trivialControlledSx),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseSX", MQT_NAMED_BUILDER(inverseSx),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseMultipleControlledSX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSx),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"SXThenSXdg", MQT_NAMED_BUILDER(sxThenSxdg),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoSX", MQT_NAMED_BUILDER(twoSx), MQT_NAMED_BUILDER(x)},
        QCOTestCase{"PowTwoSX", MQT_NAMED_BUILDER(powTwoSx),
                    MQT_NAMED_BUILDER(powTwoSxRef)},
        QCOTestCase{"PowThirdSxGeneral", MQT_NAMED_BUILDER(powThirdSx),
                    MQT_NAMED_BUILDER(powThirdSxRef)}));
/// @}

/// \name QCO/Operations/StandardGates/SxdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg), MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"SingleControlledSXdg",
                    MQT_NAMED_BUILDER(singleControlledSxdg),
                    MQT_NAMED_BUILDER(singleControlledSxdg)},
        QCOTestCase{"MultipleControlledSXdg",
                    MQT_NAMED_BUILDER(multipleControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"NestedControlledSXdg",
                    MQT_NAMED_BUILDER(nestedControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSxdg)},
        QCOTestCase{"TrivialControlledSXdg",
                    MQT_NAMED_BUILDER(trivialControlledSxdg),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"InverseSXdg", MQT_NAMED_BUILDER(inverseSxdg),
                    MQT_NAMED_BUILDER(sx)},
        QCOTestCase{"InverseMultipleControlledSXdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledSxdg),
                    MQT_NAMED_BUILDER(multipleControlledSx)},
        QCOTestCase{"SXdgThenSX", MQT_NAMED_BUILDER(sxdgThenSx),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoSXdg", MQT_NAMED_BUILDER(twoSxdg),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"PowTwoSXdg", MQT_NAMED_BUILDER(powTwoSxdg),
                    MQT_NAMED_BUILDER(powTwoSxdgRef)},
        QCOTestCase{"PowThirdSxdgGeneral", MQT_NAMED_BUILDER(powThirdSxdg),
                    MQT_NAMED_BUILDER(powThirdSxdgRef)}));
/// @}

/// \name QCO/Operations/StandardGates/TOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"T", MQT_NAMED_BUILDER(t_), MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"SingleControlledT", MQT_NAMED_BUILDER(singleControlledT),
                    MQT_NAMED_BUILDER(singleControlledT)},
        QCOTestCase{"MultipleControlledT",
                    MQT_NAMED_BUILDER(multipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"NestedControlledT", MQT_NAMED_BUILDER(nestedControlledT),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"TrivialControlledT", MQT_NAMED_BUILDER(trivialControlledT),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"InverseT", MQT_NAMED_BUILDER(inverseT),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"InverseMultipleControlledT",
                    MQT_NAMED_BUILDER(inverseMultipleControlledT),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"TThenTdg", MQT_NAMED_BUILDER(tThenTdg),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoT", MQT_NAMED_BUILDER(twoT), MQT_NAMED_BUILDER(s)},
        QCOTestCase{"PowTwoT", MQT_NAMED_BUILDER(powTwoT),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"PowThirdTToP", MQT_NAMED_BUILDER(powThirdT),
                    MQT_NAMED_BUILDER(powThirdTRef)}));
/// @}

/// \name QCO/Operations/StandardGates/TdgOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOTdgOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Tdg", MQT_NAMED_BUILDER(tdg), MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"SingleControlledTdg",
                    MQT_NAMED_BUILDER(singleControlledTdg),
                    MQT_NAMED_BUILDER(singleControlledTdg)},
        QCOTestCase{"MultipleControlledTdg",
                    MQT_NAMED_BUILDER(multipleControlledTdg),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"NestedControlledTdg",
                    MQT_NAMED_BUILDER(nestedControlledTdg),
                    MQT_NAMED_BUILDER(multipleControlledTdg)},
        QCOTestCase{"TrivialControlledTdg",
                    MQT_NAMED_BUILDER(trivialControlledTdg),
                    MQT_NAMED_BUILDER(tdg)},
        QCOTestCase{"InverseTdg", MQT_NAMED_BUILDER(inverseTdg),
                    MQT_NAMED_BUILDER(t_)},
        QCOTestCase{"InverseMultipleControlledTdg",
                    MQT_NAMED_BUILDER(inverseMultipleControlledTdg),
                    MQT_NAMED_BUILDER(multipleControlledT)},
        QCOTestCase{"TdgThenT", MQT_NAMED_BUILDER(tdgThenT),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"TwoTdg", MQT_NAMED_BUILDER(twoTdg),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"PowTwoTdg", MQT_NAMED_BUILDER(powTwoTdg),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"PowThirdTdgToP", MQT_NAMED_BUILDER(powThirdTdg),
                    MQT_NAMED_BUILDER(powThirdTdgRef)}));
/// @}

/// \name QCO/Operations/StandardGates/U2Op.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOU2OpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U2", MQT_NAMED_BUILDER(u2), MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"SingleControlledU2", MQT_NAMED_BUILDER(singleControlledU2),
                    MQT_NAMED_BUILDER(singleControlledU2)},
        QCOTestCase{"MultipleControlledU2",
                    MQT_NAMED_BUILDER(multipleControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"NestedControlledU2", MQT_NAMED_BUILDER(nestedControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"TrivialControlledU2",
                    MQT_NAMED_BUILDER(trivialControlledU2),
                    MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"InverseU2", MQT_NAMED_BUILDER(inverseU2),
                    MQT_NAMED_BUILDER(u2)},
        QCOTestCase{"InverseMultipleControlledU2",
                    MQT_NAMED_BUILDER(inverseMultipleControlledU2),
                    MQT_NAMED_BUILDER(multipleControlledU2)},
        QCOTestCase{"CanonicalizeU2ToH", MQT_NAMED_BUILDER(canonicalizeU2ToH),
                    MQT_NAMED_BUILDER(h)},
        QCOTestCase{"CanonicalizeU2ToRx", MQT_NAMED_BUILDER(canonicalizeU2ToRx),
                    MQT_NAMED_BUILDER(rxPiOver2)},
        QCOTestCase{"CanonicalizeU2ToRy", MQT_NAMED_BUILDER(canonicalizeU2ToRy),
                    MQT_NAMED_BUILDER(ryPiOver2)}));
/// @}

/// \name QCO/Operations/StandardGates/UOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOUOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"U", MQT_NAMED_BUILDER(u), MQT_NAMED_BUILDER(u)},
        QCOTestCase{"SingleControlledU", MQT_NAMED_BUILDER(singleControlledU),
                    MQT_NAMED_BUILDER(singleControlledU)},
        QCOTestCase{"MultipleControlledU",
                    MQT_NAMED_BUILDER(multipleControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"NestedControlledU", MQT_NAMED_BUILDER(nestedControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"TrivialControlledU", MQT_NAMED_BUILDER(trivialControlledU),
                    MQT_NAMED_BUILDER(u)},
        QCOTestCase{"InverseU", MQT_NAMED_BUILDER(inverseU),
                    MQT_NAMED_BUILDER(u)},
        QCOTestCase{"InverseMultipleControlledU",
                    MQT_NAMED_BUILDER(inverseMultipleControlledU),
                    MQT_NAMED_BUILDER(multipleControlledU)},
        QCOTestCase{"CanonicalizeUToP", MQT_NAMED_BUILDER(canonicalizeUToP),
                    MQT_NAMED_BUILDER(p)},
        QCOTestCase{"CanonicalizeUToRx", MQT_NAMED_BUILDER(canonicalizeUToRx),
                    MQT_NAMED_BUILDER(rx)},
        QCOTestCase{"CanonicalizeUToRy", MQT_NAMED_BUILDER(canonicalizeUToRy),
                    MQT_NAMED_BUILDER(ry)},
        QCOTestCase{"CanonicalizeUToU2", MQT_NAMED_BUILDER(canonicalizeUToU2),
                    MQT_NAMED_BUILDER(u2)}));
/// @}

/// \name QCO/Operations/StandardGates/XOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"X", MQT_NAMED_BUILDER(x), MQT_NAMED_BUILDER(x)},
        QCOTestCase{"SingleControlledX", MQT_NAMED_BUILDER(singleControlledX),
                    MQT_NAMED_BUILDER(singleControlledX)},
        QCOTestCase{"MultipleControlledX",
                    MQT_NAMED_BUILDER(multipleControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"NestedControlledX", MQT_NAMED_BUILDER(nestedControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"TrivialControlledX", MQT_NAMED_BUILDER(trivialControlledX),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"InverseX", MQT_NAMED_BUILDER(inverseX),
                    MQT_NAMED_BUILDER(x)},
        QCOTestCase{"InverseMultipleControlledX",
                    MQT_NAMED_BUILDER(inverseMultipleControlledX),
                    MQT_NAMED_BUILDER(multipleControlledX)},
        QCOTestCase{"TwoX", MQT_NAMED_BUILDER(twoX),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"ControlledTwoX", MQT_NAMED_BUILDER(controlledTwoX),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"InverseTwoX", MQT_NAMED_BUILDER(inverseTwoX),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfX", MQT_NAMED_BUILDER(powHalfX),
                    MQT_NAMED_BUILDER(powHalfXRef)},
        QCOTestCase{"PowNegHalfXToSXdg", MQT_NAMED_BUILDER(powNegHalfX),
                    MQT_NAMED_BUILDER(sxdg)},
        QCOTestCase{"PowThirdXGeneral", MQT_NAMED_BUILDER(powThirdX),
                    MQT_NAMED_BUILDER(powThirdXRef)}));
/// @}

/// \name QCO/Operations/StandardGates/XxMinusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXMinusYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"XXMinusYY", MQT_NAMED_BUILDER(xxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"SingleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(singleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(singleControlledXxMinusYY)},
        QCOTestCase{"MultipleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"NestedControlledXXMinusYY",
                    MQT_NAMED_BUILDER(nestedControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"TrivialControlledXXMinusYY",
                    MQT_NAMED_BUILDER(trivialControlledXxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"InverseXXMinusYY", MQT_NAMED_BUILDER(inverseXxMinusYY),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"InverseMultipleControlledXXMinusYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledXxMinusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxMinusYY)},
        QCOTestCase{"TwoXXMinusYYOppositePhase",
                    MQT_NAMED_BUILDER(twoXxMinusYYOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoXXMinusYYSwappedTargets",
                    MQT_NAMED_BUILDER(twoXxMinusYYSwappedTargets),
                    MQT_NAMED_BUILDER(xxMinusYY)},
        QCOTestCase{"PowXxMinusYYScaled", MQT_NAMED_BUILDER(powXxMinusYYScaled),
                    MQT_NAMED_BUILDER(powXxMinusYYScaledRef)}));
/// @}

/// \name QCO/Operations/StandardGates/XxPlusYyOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOXXPlusYYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"XXPlusYY", MQT_NAMED_BUILDER(xxPlusYY),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"SingleControlledXXPlusYY",
                    MQT_NAMED_BUILDER(singleControlledXxPlusYY),
                    MQT_NAMED_BUILDER(singleControlledXxPlusYY)},
        QCOTestCase{"MultipleControlledXXPlusYY",
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCOTestCase{"NestedControlledXXPlusYY",
                    MQT_NAMED_BUILDER(nestedControlledXxPlusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCOTestCase{"TrivialControlledXXPlusYY",
                    MQT_NAMED_BUILDER(trivialControlledXxPlusYY),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"InverseXXPlusYY", MQT_NAMED_BUILDER(inverseXxPlusYY),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"InverseMultipleControlledXXPlusYY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledXxPlusYY),
                    MQT_NAMED_BUILDER(multipleControlledXxPlusYY)},
        QCOTestCase{"TwoXXPlusYYOppositePhase",
                    MQT_NAMED_BUILDER(twoXxPlusYYOppositePhase),
                    MQT_NAMED_BUILDER(alloc2QubitRegister)},
        QCOTestCase{"TwoXXPlusYYSwappedTargets",
                    MQT_NAMED_BUILDER(twoXxPlusYYSwappedTargets),
                    MQT_NAMED_BUILDER(xxPlusYY)},
        QCOTestCase{"PowXxPlusYYScaled", MQT_NAMED_BUILDER(powXxPlusYYScaled),
                    MQT_NAMED_BUILDER(powXxPlusYYScaledRef)}));
/// @}

/// \name QCO/Operations/StandardGates/YOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOYOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Y", MQT_NAMED_BUILDER(y), MQT_NAMED_BUILDER(y)},
        QCOTestCase{"SingleControlledY", MQT_NAMED_BUILDER(singleControlledY),
                    MQT_NAMED_BUILDER(singleControlledY)},
        QCOTestCase{"MultipleControlledY",
                    MQT_NAMED_BUILDER(multipleControlledY),
                    MQT_NAMED_BUILDER(multipleControlledY)},
        QCOTestCase{"NestedControlledY", MQT_NAMED_BUILDER(nestedControlledY),
                    MQT_NAMED_BUILDER(multipleControlledY)},
        QCOTestCase{"TrivialControlledY", MQT_NAMED_BUILDER(trivialControlledY),
                    MQT_NAMED_BUILDER(y)},
        QCOTestCase{"InverseY", MQT_NAMED_BUILDER(inverseY),
                    MQT_NAMED_BUILDER(y)},
        QCOTestCase{"InverseMultipleControlledY",
                    MQT_NAMED_BUILDER(inverseMultipleControlledY),
                    MQT_NAMED_BUILDER(multipleControlledY)},
        QCOTestCase{"TwoY", MQT_NAMED_BUILDER(twoY),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfY", MQT_NAMED_BUILDER(powHalfY),
                    MQT_NAMED_BUILDER(powHalfYRef)}));
/// @}

/// \name QCO/Operations/StandardGates/ZOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOZOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"Z", MQT_NAMED_BUILDER(z), MQT_NAMED_BUILDER(z)},
        QCOTestCase{"SingleControlledZ", MQT_NAMED_BUILDER(singleControlledZ),
                    MQT_NAMED_BUILDER(singleControlledZ)},
        QCOTestCase{"MultipleControlledZ",
                    MQT_NAMED_BUILDER(multipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"NestedControlledZ", MQT_NAMED_BUILDER(nestedControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TrivialControlledZ", MQT_NAMED_BUILDER(trivialControlledZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseZ", MQT_NAMED_BUILDER(inverseZ),
                    MQT_NAMED_BUILDER(z)},
        QCOTestCase{"InverseMultipleControlledZ",
                    MQT_NAMED_BUILDER(inverseMultipleControlledZ),
                    MQT_NAMED_BUILDER(multipleControlledZ)},
        QCOTestCase{"TwoZ", MQT_NAMED_BUILDER(twoZ),
                    MQT_NAMED_BUILDER(alloc1QubitRegister)},
        QCOTestCase{"PowHalfZ", MQT_NAMED_BUILDER(powHalfZ),
                    MQT_NAMED_BUILDER(s)},
        QCOTestCase{"NormalizeAngleWrapZ", MQT_NAMED_BUILDER(powThreeHalvesZ),
                    MQT_NAMED_BUILDER(sdg)},
        QCOTestCase{"PowThirdZToP", MQT_NAMED_BUILDER(powThirdZ),
                    MQT_NAMED_BUILDER(powThirdZRef)}));
/// @}

/// \name QCO/Operations/MeasureOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOMeasureOpTest, QCOTest,
    testing::Values(
        QCOTestCase{"SingleMeasurementToSingleBit",
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit),
                    MQT_NAMED_BUILDER(singleMeasurementToSingleBit)},
        QCOTestCase{"RepeatedMeasurementToSameBit",
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit),
                    MQT_NAMED_BUILDER(repeatedMeasurementToSameBit)},
        QCOTestCase{"RepeatedMeasurementToDifferentBits",
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits),
                    MQT_NAMED_BUILDER(repeatedMeasurementToDifferentBits)},
        QCOTestCase{
            "MultipleClassicalRegistersAndMeasurements",
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements),
            MQT_NAMED_BUILDER(multipleClassicalRegistersAndMeasurements)}));
/// @}

/// \name QCO/Operations/ResetOp.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOResetOpTest, QCOTest,
    testing::Values(QCOTestCase{"ResetQubitWithoutOp",
                                MQT_NAMED_BUILDER(resetQubitWithoutOp),
                                MQT_NAMED_BUILDER(allocQubit)},
                    QCOTestCase{"ResetMultipleQubitsWithoutOp",
                                MQT_NAMED_BUILDER(resetMultipleQubitsWithoutOp),
                                MQT_NAMED_BUILDER(alloc2QubitRegister)},
                    QCOTestCase{"RepeatedResetWithoutOp",
                                MQT_NAMED_BUILDER(repeatedResetWithoutOp),
                                MQT_NAMED_BUILDER(allocQubit)},
                    QCOTestCase{"ResetQubitAfterSingleOp",
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp),
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp)},
                    QCOTestCase{
                        "ResetMultipleQubitsAfterSingleOp",
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp),
                        MQT_NAMED_BUILDER(resetMultipleQubitsAfterSingleOp)},
                    QCOTestCase{"RepeatedResetAfterSingleOp",
                                MQT_NAMED_BUILDER(repeatedResetAfterSingleOp),
                                MQT_NAMED_BUILDER(resetQubitAfterSingleOp)}));
/// @}

/// \name QCO/QubitManagement/QubitManagement.cpp
/// @{
INSTANTIATE_TEST_SUITE_P(
    QCOQubitManagementTest, QCOTest,
    testing::Values(
        QCOTestCase{"AllocQubit", MQT_NAMED_BUILDER(allocQubitNoMeasure),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"StaticQubitsNoMeasure",
                    MQT_NAMED_BUILDER(staticQubitsNoMeasure),
                    MQT_NAMED_BUILDER(emptyQCO)},
        QCOTestCase{"StaticQubitsWithOps",
                    MQT_NAMED_BUILDER(staticQubitsWithOps),
                    MQT_NAMED_BUILDER(staticQubitsWithOps)},
        QCOTestCase{"StaticQubitsWithParametricOps",
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps),
                    MQT_NAMED_BUILDER(staticQubitsWithParametricOps)},
        QCOTestCase{"StaticQubitsWithTwoTargetOps",
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps),
                    MQT_NAMED_BUILDER(staticQubitsWithTwoTargetOps)},
        QCOTestCase{"StaticQubitsWithCtrl",
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl),
                    MQT_NAMED_BUILDER(staticQubitsWithCtrl)},
        QCOTestCase{"StaticQubitsWithInv",
                    MQT_NAMED_BUILDER(staticQubitsWithInv),
                    MQT_NAMED_BUILDER(staticQubitsWithInv)},
        QCOTestCase{"AllocSinkPair", MQT_NAMED_BUILDER(allocSinkPair),
                    MQT_NAMED_BUILDER(allocQubitNoMeasure)}));
/// @}

/// \name UnrollModifiers
/// @{
static LogicalResult runUnrollModifiers(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(mlir::mqt::createUnrollModifiers());
  return pm.run(moduleOp);
}

/// Unrolls @p program and checks that it matches @p reference.
static void
expectUnrollsTo(MLIRContext* context,
                const function_ref<Value(QCOProgramBuilder&)> program,
                const function_ref<Value(QCOProgramBuilder&)> reference,
                void (*checkStructure)(ModuleOp) = nullptr) {
  auto moduleOp = QCOProgramBuilder::build(context, program);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runUnrollModifiers(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  if (checkStructure != nullptr) {
    checkStructure(*moduleOp);
  }
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(moduleOp.get())));

  auto referenceOp = QCOProgramBuilder::build(context, reference);
  ASSERT_TRUE(referenceOp);
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(referenceOp.get())));

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(moduleOp.get(), referenceOp.get()));
}

static SmallVector<MeasureOp> getMeasurements(ModuleOp moduleOp) {
  SmallVector<MeasureOp> measurements;
  moduleOp.walk([&](MeasureOp op) { measurements.push_back(op); });
  return measurements;
}

static void checkCtrlTwoStructure(ModuleOp moduleOp) {
  SmallVector<CtrlOp> modifiers;
  moduleOp.walk([&](CtrlOp op) { modifiers.push_back(op); });
  ASSERT_EQ(modifiers.size(), 2);
  EXPECT_EQ(modifiers[0].getNumTargets(), 1);
  EXPECT_EQ(modifiers[1].getNumTargets(), 2);
  EXPECT_EQ(modifiers[0].getNumBodyUnitaries(), 1);
  EXPECT_EQ(modifiers[1].getNumBodyUnitaries(), 1);
  EXPECT_EQ(modifiers[1].getControlsIn(), modifiers[0].getControlsOut());
  EXPECT_EQ(modifiers[1].getTargetsIn()[0], modifiers[0].getTargetsOut()[0]);

  auto measurements = getMeasurements(moduleOp);
  ASSERT_EQ(measurements.size(), 4);
  EXPECT_EQ(measurements[0].getQubitIn(), modifiers[1].getControlsOut()[0]);
  EXPECT_EQ(measurements[1].getQubitIn(), modifiers[1].getControlsOut()[1]);
  EXPECT_EQ(measurements[2].getQubitIn(), modifiers[1].getTargetsOut()[0]);
  EXPECT_EQ(measurements[3].getQubitIn(), modifiers[1].getTargetsOut()[1]);
}

static void checkCtrlThreeStructure(ModuleOp moduleOp) {
  SmallVector<CtrlOp> modifiers;
  moduleOp.walk([&](CtrlOp op) { modifiers.push_back(op); });
  ASSERT_EQ(modifiers.size(), 3);
  EXPECT_EQ(modifiers[0].getNumTargets(), 1);
  EXPECT_EQ(modifiers[1].getNumTargets(), 2);
  EXPECT_EQ(modifiers[2].getNumTargets(), 1);
  EXPECT_EQ(modifiers[1].getTargetsIn()[0], modifiers[0].getTargetsOut()[0]);
  EXPECT_EQ(modifiers[2].getTargetsIn()[0], modifiers[1].getTargetsOut()[0]);

  auto measurements = getMeasurements(moduleOp);
  ASSERT_EQ(measurements.size(), 3);
  EXPECT_EQ(measurements[0].getQubitIn(), modifiers[2].getControlsOut()[0]);
  EXPECT_EQ(measurements[1].getQubitIn(), modifiers[1].getTargetsOut()[1]);
  EXPECT_EQ(measurements[2].getQubitIn(), modifiers[2].getTargetsOut()[0]);
}

static void checkInvStructure(ModuleOp moduleOp) {
  SmallVector<InvOp> modifiers;
  moduleOp.walk([&](InvOp op) { modifiers.push_back(op); });
  ASSERT_EQ(modifiers.size(), 2);
  ASSERT_EQ(modifiers[0].getNumBodyUnitaries(), 1);
  ASSERT_EQ(modifiers[1].getNumBodyUnitaries(), 1);
  EXPECT_TRUE(isa<RXXOp>(modifiers[0].getBodyUnitary(0).getOperation()));
  EXPECT_TRUE(isa<XOp>(modifiers[1].getBodyUnitary(0).getOperation()));
  EXPECT_EQ(modifiers[0].getNumQubits(), 2);
  EXPECT_EQ(modifiers[1].getNumQubits(), 1);
  EXPECT_EQ(modifiers[1].getQubitsIn()[0], modifiers[0].getResults()[0]);

  auto measurements = getMeasurements(moduleOp);
  ASSERT_EQ(measurements.size(), 2);
  EXPECT_EQ(measurements[0].getQubitIn(), modifiers[1].getResults()[0]);
  EXPECT_EQ(measurements[1].getQubitIn(), modifiers[0].getResults()[1]);
}

static void checkSplitPowStructure(ModuleOp moduleOp) {
  SmallVector<PowOp> modifiers;
  moduleOp.walk([&](PowOp op) { modifiers.push_back(op); });
  ASSERT_EQ(modifiers.size(), 2);
  for (auto modifier : modifiers) {
    EXPECT_EQ(modifier.getNumQubits(), 1);
    EXPECT_EQ(modifier.getNumBodyUnitaries(), 1);
  }

  auto measurements = getMeasurements(moduleOp);
  ASSERT_EQ(measurements.size(), 2);
  EXPECT_EQ(measurements[0].getQubitIn(), modifiers[0].getResult(0));
  EXPECT_EQ(measurements[1].getQubitIn(), modifiers[1].getResult(0));
}

static void checkPreservedPowStructure(ModuleOp moduleOp) {
  SmallVector<PowOp> modifiers;
  moduleOp.walk([&](PowOp op) { modifiers.push_back(op); });
  ASSERT_EQ(modifiers.size(), 1);
  EXPECT_EQ(modifiers[0].getNumQubits(), 2);
  EXPECT_EQ(modifiers[0].getNumBodyUnitaries(), 2);
}

/// Reference for `ctrlThree` after unrolling.
static Value ctrlThreeUnrolled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto first = b.ctrl(q[0], {q[2]}, [&](ValueRange targets) {
    return SmallVector{b.x(targets[0])};
  });
  auto second = b.ctrl(first.first[0], {first.second[0], q[1]},
                       [&](ValueRange targets) -> SmallVector<Value> {
                         auto [t0, t1] = b.dcx(targets[0], targets[1]);
                         return {t0, t1};
                       });
  auto third =
      b.ctrl(second.first[0], {second.second[0]},
             [&](ValueRange targets) { return SmallVector{b.y(targets[0])}; });
  return measureRegister(b,
                         {third.first[0], second.second[1], third.second[0]});
}

/// Reference for `ctrlInvTwo` after unrolling.
static Value ctrlInvTwoUnrolled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto first = b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    return llvm::to_vector(
        b.inv(targets, [&](ValueRange qubits) -> SmallVector<Value> {
          auto [t0, t1] = b.rxx(0.123, qubits[0], qubits[1]);
          return {t0, t1};
        }));
  });
  auto second =
      b.ctrl(first.first[0], {first.second[0]}, [&](ValueRange targets) {
        return llvm::to_vector(b.inv(targets, [&](ValueRange qubits) {
          return SmallVector{b.x(qubits[0])};
        }));
      });
  return measureRegister(b,
                         {second.first[0], second.second[0], first.second[1]});
}

/// Applies a control modifier to an inverse modifier with two operations,
/// followed by a further operation.
static Value ctrlTwoInvTwo(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto res = b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    auto inner = b.inv(targets, [&](ValueRange qubits) -> SmallVector<Value> {
      auto i0 = b.x(qubits[0]);
      auto [t0, t1] = b.rxx(0.123, i0, qubits[1]);
      return {t0, t1};
    });
    return SmallVector{b.h(inner[0]), inner[1]};
  });
  return measureRegister(b, {res.first[0], res.second[0], res.second[1]});
}

/// Reference for `ctrlTwoInvTwo` after unrolling.
static Value ctrlTwoInvTwoUnrolled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(3);
  auto first = b.ctrl(q[0], {q[1], q[2]}, [&](ValueRange targets) {
    return llvm::to_vector(
        b.inv(targets, [&](ValueRange qubits) -> SmallVector<Value> {
          auto [t0, t1] = b.rxx(0.123, qubits[0], qubits[1]);
          return {t0, t1};
        }));
  });
  auto second =
      b.ctrl(first.first[0], {first.second[0]}, [&](ValueRange targets) {
        return llvm::to_vector(b.inv(targets, [&](ValueRange qubits) {
          return SmallVector{b.x(qubits[0])};
        }));
      });
  auto third =
      b.ctrl(second.first[0], {second.second[0]},
             [&](ValueRange targets) { return SmallVector{b.h(targets[0])}; });
  return measureRegister(b, {third.first[0], third.second[0], first.second[1]});
}
/// Reference for `powTwoDisjoint` after unrolling.
static Value powTwoDisjointUnrolled(QCOProgramBuilder& b) {
  auto q = b.allocQubitRegister(2);
  auto first = b.pow(2.0, {q[0]}, [&](ValueRange qubits) {
    return SmallVector{b.s(qubits[0])};
  });
  auto second = b.pow(2.0, {q[1]}, [&](ValueRange qubits) {
    return SmallVector{b.t(qubits[0])};
  });
  return measureRegister(b, {first[0], second[0]});
}

TEST_F(QCOTest, UnrollModifiersSplitsCtrl) {
  expectUnrollsTo(context.get(), ctrlTwo, ctrlTwoUnrolled,
                  checkCtrlTwoStructure);
}

TEST_F(QCOTest, UnrollModifiersSplitsCtrlWithReorderedTargets) {
  expectUnrollsTo(context.get(), ctrlThree, ctrlThreeUnrolled,
                  checkCtrlThreeStructure);
}

TEST_F(QCOTest, UnrollModifiersReversesInv) {
  expectUnrollsTo(context.get(), invTwo, invTwoUnrolled, checkInvStructure);
}

TEST_F(QCOTest, UnrollModifiersUnrollsNestedModifiers) {
  expectUnrollsTo(context.get(), ctrlInvTwo, ctrlInvTwoUnrolled);
}

TEST_F(QCOTest, UnrollModifiersUnrollsNestedModifiersAndTrailingOperation) {
  expectUnrollsTo(context.get(), ctrlTwoInvTwo, ctrlTwoInvTwoUnrolled);
}

TEST_F(QCOTest, UnrollModifiersSplitsDisjointPow) {
  expectUnrollsTo(context.get(), powTwoDisjoint, powTwoDisjointUnrolled,
                  checkSplitPowStructure);
}

TEST_F(QCOTest, UnrollModifiersLeavesOverlappingPowUntouched) {
  expectUnrollsTo(context.get(), powTwo, powTwo, checkPreservedPowStructure);
}

TEST_F(QCOTest, UnrollModifiersLeavesNonIntegerPowUntouched) {
  expectUnrollsTo(context.get(), powHalfDisjoint, powHalfDisjoint,
                  checkPreservedPowStructure);
}
/// @}
