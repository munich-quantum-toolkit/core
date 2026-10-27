/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QCOToQC/QCOToQC.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Support/Passes.h"

#include "Conversion/ConversionTestCases.h"
#include "Support/IRVerification.h"
#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "qco_programs.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <array>
#include <memory>
#include <string>
#include <tuple>

using namespace mlir;

namespace {

struct QCOToQCTestCase {
  std::string name;
  ::mqt::test::NamedMLIRBuilder<qco::QCOProgramBuilder> programBuilder;
  ::mqt::test::NamedMLIRBuilder<qc::QCProgramBuilder> referenceBuilder;
};

class QCOToQCTest : public testing::TestWithParam<QCOToQCTestCase> {
protected:
  std::unique_ptr<MLIRContext> context;

  QCOToQCTest() {
    // Register all necessary dialects
    DialectRegistry registry;
    registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                    arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
};

} // namespace

static LogicalResult runQCOToQCConversion(ModuleOp moduleOp) {
  PassManager pm(moduleOp.getContext());
  pm.addPass(createQCOToQC());
  return pm.run(moduleOp);
}

TEST(QCOToQCRegressionTest, PreservesClassicalOperations) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect, LLVM::LLVMDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main(%x: i64) -> i1 {
      %same = llvm.icmp "eq" %x, %x : i64
      return %same : i1
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
}

TEST(QCOToQCRegressionTest, ReportsUnsupportedQubitConsumer) {
  MLIRContext context;
  context.loadDialect<qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                      func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main(%condition: i1) attributes {mqt.entry_point} {
      %left = qco.alloc : !qco.qubit
      %right = qco.alloc : !qco.qubit
      %selected = arith.select %condition, %left, %right : !qco.qubit
      qco.sink %selected : !qco.qubit
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  bool sawError = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawError |= diagnostic.getSeverity() == DiagnosticSeverity::Error;
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_TRUE(sawError);
}

TEST(QCOToQCRegressionTest, RejectsUnsupportedDynamicTensorOwnership) {
  MLIRContext context;
  context
      .loadDialect<qco::QCODialect, qtensor::QTensorDialect,
                   arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  for (const auto* body : {
           R"mlir(%q = qco.alloc : !qco.qubit
             %tensor = qtensor.from_elements %q : tensor<1x!qco.qubit>
             qtensor.dealloc %tensor : tensor<1x!qco.qubit>)mlir",
           R"mlir(%size = arith.constant 2 : index
             %index = arith.constant 1 : index
             %tensor = qtensor.alloc(%size) : tensor<2x!qco.qubit>
             %rest, %q = qtensor.extract %tensor[%index] : tensor<2x!qco.qubit>
             qtensor.dealloc %rest : tensor<2x!qco.qubit>
             qco.sink %q : !qco.qubit)mlir",
           R"mlir(%size = arith.constant 2 : index
             %index = arith.constant 1 : index
             %tensor = qtensor.alloc(%size) : tensor<?x!qco.qubit>
             %rest, %q = qtensor.extract %tensor[%index] : tensor<?x!qco.qubit>
             qco.sink %q : !qco.qubit
             qtensor.dealloc %rest : tensor<?x!qco.qubit>)mlir",
           R"mlir(%size = arith.constant 2 : index
             %zero = arith.constant 0 : index
             %one = arith.constant 1 : index
             %tensor = qtensor.alloc(%size) : tensor<2x!qco.qubit>
             %rest, %q = qtensor.extract %tensor[%zero] : tensor<2x!qco.qubit>
             %next = scf.for %i = %zero to %one step %one
                 iter_args(%incomplete = %rest) -> (tensor<2x!qco.qubit>) {
               scf.yield %incomplete : tensor<2x!qco.qubit>
             }
             %full = qtensor.insert %q into %next[%zero] : tensor<2x!qco.qubit>
             qtensor.dealloc %full : tensor<2x!qco.qubit>)mlir",
       }) {
    auto moduleOp = parseSourceString<ModuleOp>(
        std::string(
            "module { func.func @main() attributes {mqt.entry_point} {") +
            body + " return } }",
        &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
    std::string diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      diagnostics += diagnostic.str();
      return success();
    });
    EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
    EXPECT_NE(diagnostics.find("QCO-to-QC requires"), std::string::npos);
    EXPECT_TRUE(succeeded(verify(*moduleOp)));
    EXPECT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  }
}

TEST(QCOToQCRegressionTest, RejectsBorrowedTensorConsumption) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, qtensor::QTensorDialect,
                      func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @release(%tensor: tensor<2x!qco.qubit>) {
      qtensor.dealloc %tensor : tensor<2x!qco.qubit>
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_NE(
      diagnostics.find(
          "must return one trailing quantum value for each quantum argument"),
      std::string::npos);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(failed(qco::verifyLinearity(*moduleOp)));
}

TEST(QCOToQCRegressionTest, RejectsIncompleteOrReorderedBorrowedRegisters) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, qtensor::QTensorDialect,
                      arith::ArithDialect, func::FuncDialect>();
  for (const auto* source : {
           R"mlir(module {
             func.func private @missing(%reg: tensor<2x!qco.qubit>)
                 -> tensor<2x!qco.qubit> {
               %c0 = arith.constant 0 : index
               %rest, %q = qtensor.extract %reg[%c0] : tensor<2x!qco.qubit>
               qco.sink %q : !qco.qubit
               return %rest : tensor<2x!qco.qubit>
             }
           })mlir",
           R"mlir(module {
             func.func private @reordered(%a: tensor<2x!qco.qubit>, %b: tensor<2x!qco.qubit>)
                 -> (tensor<2x!qco.qubit>, tensor<2x!qco.qubit>) {
               return %b, %a : tensor<2x!qco.qubit>, tensor<2x!qco.qubit>
             }
           })mlir",
       }) {
    SCOPED_TRACE(source);
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    auto original = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(original);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic&) { return success(); });
    EXPECT_EQ(failed(qco::verifyLinearity(*moduleOp)),
              StringRef(source).contains("@reordered"));
    EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
    EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
        moduleOp->getOperation(), original->getOperation(),
        OperationEquivalence::Flags::None));
  }
}

TEST(QCOToQCRegressionTest, FindsAllocationModeBeforeConvertingCaller) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
 func.func @main() attributes {mqt.entry_point} {
  %q = func.call @make() : () -> !qco.qubit
  qco.sink %q : !qco.qubit
  return
 }
 func.func private @make() -> !qco.qubit {
  %q = qco.alloc : !qco.qubit
  return %q : !qco.qubit
 }
}
)mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto caller = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto call = *caller.getOps<func::CallOp>().begin();
  auto dealloc = *caller.getOps<qc::DeallocOp>().begin();
  EXPECT_EQ(dealloc.getQubit(), call.getResult(0));
}

TEST(QCOToQCRegressionTest, StripsPositionalQubitResultsFromUnitaryCalls) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func private @flip(%q: !qco.qubit) -> !qco.qubit
      attributes {mqt.unitary} {
    %out = qco.x %q : !qco.qubit
    return %out : !qco.qubit
  }
  func.func @caller(%q: !qco.qubit) -> !qco.qubit {
    %out = qco.call @flip(%q) : (!qco.qubit) -> !qco.qubit
    return %out : !qco.qubit
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  for (auto function : moduleOp->getOps<func::FuncOp>()) {
    EXPECT_EQ(function.getNumResults(), 0U);
  }
  std::size_t calls = 0;
  moduleOp->walk([&](qc::CallOp) { ++calls; });
  EXPECT_EQ(calls, 1U);
}

TEST(QCOToQCRegressionTest, StripsPositionalQubitResultsFromGenericCalls) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func private @reset(%q: !qco.qubit) -> (i1, !qco.qubit) {
    %out = qco.reset %q : !qco.qubit
    %flag = arith.constant true
    return %flag, %out : i1, !qco.qubit
  }
  func.func @caller(%q: !qco.qubit) -> (i1, !qco.qubit) {
    %flag, %out = func.call @reset(%q)
        : (!qco.qubit) -> (i1, !qco.qubit)
    return %flag, %out : i1, !qco.qubit
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  for (auto function : moduleOp->getOps<func::FuncOp>()) {
    ASSERT_EQ(function.getNumResults(), 1U);
    EXPECT_TRUE(function.getResultTypes().front().isInteger(1));
  }
  auto main = moduleOp->lookupSymbol<func::FuncOp>("caller");
  ASSERT_TRUE(main);
  auto call = *main.getBody().getOps<func::CallOp>().begin();
  ASSERT_EQ(call.getNumResults(), 1U);
  EXPECT_TRUE(call.getResult(0).getType().isInteger(1));
}

TEST(QCOToQCRegressionTest, RejectsUnrepresentableCallResultAttributes) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr auto sources = std::to_array<llvm::StringLiteral>({
      R"mlir(
module {
  func.func private @reset(%q: !qco.qubit) -> (i1, !qco.qubit) {
    %out = qco.reset %q : !qco.qubit
    %flag = arith.constant true
    return %flag, %out : i1, !qco.qubit
  }
  func.func @caller(%q: !qco.qubit) -> (i1, !qco.qubit) {
    %flag, %out = func.call @reset(%q) {
        res_attrs = [{}, {tag = "wire"}]
      } : (!qco.qubit) -> (i1, !qco.qubit)
    return %flag, %out : i1, !qco.qubit
  }
}
)mlir",
      R"mlir(
module {
  func.func private @flip(%q: !qco.qubit) -> !qco.qubit
      attributes {mqt.unitary} {
    %out = qco.x %q : !qco.qubit
    return %out : !qco.qubit
  }
  func.func @caller(%q: !qco.qubit) -> !qco.qubit {
    %out = qco.call @flip(%q) {res_attrs = [{tag = "wire"}]}
        : (!qco.qubit) -> !qco.qubit
    return %out : !qco.qubit
  }
}
)mlir",
  });

  for (const auto source : sources) {
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    bool sawExpectedDiagnostic = false;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      sawExpectedDiagnostic |=
          StringRef(diagnostic.str()).contains("cannot preserve");
      return success();
    });
    EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
    EXPECT_TRUE(sawExpectedDiagnostic);
  }
}

TEST(QCOToQCRegressionTest, RejectsUnrepresentableFunctionResultAttributes) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  qco::QCOProgramBuilder builder(&context);
  builder.initialize();
  auto function = builder.createFunction(
      "passthrough", TypeRange{qco::QubitType::get(&context)},
      [](ValueRange arguments) {
        return SmallVector<Value>{arguments.front()};
      });
  function.setResultAttr(0, "test.tag", StringAttr::get(&context, "wire"));
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |=
        StringRef(diagnostic.str()).contains("cannot preserve attributes");
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST(QCOToQCRegressionTest, RejectsNonPositionalQubitResults) {
  DialectRegistry registry;
  registry.insert<qco::QCODialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @swap(%left: !qco.qubit, %right: !qco.qubit)
      -> (!qco.qubit, !qco.qubit) {
    return %right, %left : !qco.qubit, !qco.qubit
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    sawExpectedDiagnostic |= StringRef(diagnostic.str())
                                 .contains("must return its quantum arguments "
                                           "positionally");
    return success();
  });
  EXPECT_TRUE(failed(qco::verifyLinearity(*moduleOp)));
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_TRUE(sawExpectedDiagnostic);
}

TEST(QCOToQCRegressionTest, RejectsMissingPositionalQubitResults) {
  DialectRegistry registry;
  registry.insert<qco::QCODialect, arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @bad(%q: !qco.qubit) -> i1 {
      qco.sink %q : !qco.qubit
      %flag = arith.constant true
      return %flag : i1
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  std::string diagnosticText;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    diagnosticText += diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(qco::verifyLinearity(*moduleOp)));
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_NE(
      diagnosticText.find(
          "must return one trailing quantum value for each quantum argument"),
      std::string::npos);
}

TEST(QCOToQCRegressionTest, RejectsNonPositionalRegionResults) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main(%condition: i1) attributes {mqt.entry_point} {
      %left = qco.alloc : !qco.qubit
      %right = qco.alloc : !qco.qubit
      %results:2 = qco.if %condition args(%a = %left, %b = %right)
          -> (!qco.qubit, !qco.qubit) {
        qco.yield %b, %a : !qco.qubit, !qco.qubit
      } else args(%a = %left, %b = %right) {
        qco.yield %a, %b : !qco.qubit, !qco.qubit
      }
      qco.sink %results#0 : !qco.qubit
      qco.sink %results#1 : !qco.qubit
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  OwningOpRef<ModuleOp> original = moduleOp->clone();
  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
    diagnostics += diagnostic.str();
    return success();
  });
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_NE(diagnostics.find("positional quantum state correspondence"),
            std::string::npos);
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      moduleOp->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest, PreservesFlatControlledGateWires) {
  MLIRContext context;
  context.loadDialect<qc::QCDialect, qco::QCODialect, func::FuncDialect>();
  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main() attributes {mqt.entry_point} {
      %left = qco.static 0 : !qco.qubit
      %right = qco.static 1 : !qco.qubit
      %control, %target = qco.ctrl(%left) targets(%arg = %right) {
        %flipped = qco.z %arg : !qco.qubit
        qco.yield %flipped : !qco.qubit
      } : ({!qco.qubit}, {!qco.qubit})
      %flipped = qco.x %control : !qco.qubit
      qco.sink %flipped : !qco.qubit
      qco.sink %target : !qco.qubit
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = moduleOp->lookupSymbol<func::FuncOp>("main");
  auto qubits = llvm::to_vector(function.getOps<qc::StaticOp>());
  ASSERT_EQ(qubits.size(), 2U);
  ASSERT_FALSE(function.getOps<qc::CtrlOp>().empty());
  auto controlled = *function.getOps<qc::CtrlOp>().begin();
  EXPECT_EQ(controlled.getControls().front(), qubits[0].getResult());
  EXPECT_EQ(controlled.getTargets().front(), qubits[1].getResult());
  ASSERT_FALSE(function.getOps<qc::XOp>().empty());
  auto flipped = *function.getOps<qc::XOp>().begin();
  EXPECT_EQ(flipped->getOperand(0), qubits[0].getResult());
}

TEST(QCOToQCRegressionTest, ChecksFlatTensorSlotCorrespondence) {
  MLIRContext context;
  context.loadDialect<qco::QCODialect, qtensor::QTensorDialect,
                      arith::ArithDialect, func::FuncDialect>();
  for (const auto slot : {0, 1}) {
    SCOPED_TRACE(slot);
    auto source = llvm::formatv(R"mlir(module {{
      func.func @main() attributes {{mqt.entry_point} {{
        %c0 = arith.constant 0 : index
        %slot = arith.constant {0} : index
        %left = qco.static 0 : !qco.qubit
        %right = qco.static 1 : !qco.qubit
        %control = qco.static 2 : !qco.qubit
        %tensor = qtensor.from_elements %left, %right : tensor<2x!qco.qubit>
        %rest, %qubit = qtensor.extract %tensor[%c0] : tensor<2x!qco.qubit>
        %control_out, %flipped = qco.ctrl(%control) targets(%arg = %qubit) {{
          %out = qco.x %arg : !qco.qubit
          qco.yield %out : !qco.qubit
        } : ({{!qco.qubit}, {{!qco.qubit})
        qco.sink %control_out : !qco.qubit
        %full = qtensor.insert %flipped into %rest[%slot] : tensor<2x!qco.qubit>
        qtensor.dealloc %full : tensor<2x!qco.qubit>
        return
      }
    })mlir",
                                slot)
                      .str();
    auto moduleOp = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(qco::verifyLinearity(*moduleOp)));
    OwningOpRef<ModuleOp> original = moduleOp->clone();
    std::string diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diagnostic) {
      diagnostics += diagnostic.str();
      return success();
    });
    auto result = runQCOToQCConversion(*moduleOp);
    if (slot == 0) {
      ASSERT_TRUE(succeeded(result));
      EXPECT_TRUE(succeeded(verify(*moduleOp)));
    } else {
      EXPECT_TRUE(failed(result));
      EXPECT_NE(diagnostics.find("original register slot"), std::string::npos);
      EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
          moduleOp->getOperation(), original->getOperation(),
          OperationEquivalence::Flags::None));
    }
  }
}

TEST(QCOToQCRegressionTest, PreservesDistinctResultsOfIndexProducer) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  auto moduleOp = parseSourceString<ModuleOp>(R"mlir(module {
    func.func @main(%condition: i1) attributes {mqt.entry_point} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %indices:2 = scf.if %condition -> (index, index) {
        scf.yield %c0, %c1 : index, index
      } else {
        scf.yield %c1, %c0 : index, index
      }
      %tensor = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
      %t1, %left = qtensor.extract %tensor[%indices#0] : tensor<2x!qco.qubit>
      %t2, %right = qtensor.extract %t1[%indices#1] : tensor<2x!qco.qubit>
      %flipped = qco.x %left : !qco.qubit
      %same = arith.addi %indices#0, %c0 : index
      %t3 = qtensor.insert %flipped into %t2[%same] : tensor<2x!qco.qubit>
      %t4 = qtensor.insert %right into %t3[%indices#1] : tensor<2x!qco.qubit>
      qtensor.dealloc %t4 : tensor<2x!qco.qubit>
      return
    }
  })mlir",
                                              &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto loads = llvm::to_vector(function.getOps<memref::LoadOp>());
  ASSERT_EQ(loads.size(), 2U);
  auto indices = *function.getOps<scf::IfOp>().begin();
  EXPECT_EQ(loads[0].getIndices().front(), indices.getResult(0));
  EXPECT_EQ(loads[1].getIndices().front(), indices.getResult(1));
  EXPECT_TRUE(function.getOps<memref::StoreOp>().empty());
}

TEST(QCOToQCRegressionTest, RejectsQTensorSlotMovesBeforeRewriting) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q0 = qco.static 0 : !qco.qubit
    %q1 = qco.static 1 : !qco.qubit
    %tensor0 = qtensor.from_elements %q0, %q1 : tensor<2x!qco.qubit>
    %tensor1, %before = qtensor.extract %tensor0[%c0] : tensor<2x!qco.qubit>
    %another_c0 = arith.constant 0 : index
    %tensor2 = qtensor.insert %before into %tensor1[%another_c0] : tensor<2x!qco.qubit>
    %tensor3 = scf.for %iv = %c0 to %c1 step %c1
        iter_args(%tensor = %tensor2) -> (tensor<2x!qco.qubit>) {
      %tensor4, %left = qtensor.extract %tensor[%c0] : tensor<2x!qco.qubit>
      %tensor5, %right = qtensor.extract %tensor4[%c1] : tensor<2x!qco.qubit>
      %tensor6 = qtensor.insert %left into %tensor5[%c1] : tensor<2x!qco.qubit>
      %tensor7 = qtensor.insert %right into %tensor6[%iv] : tensor<2x!qco.qubit>
      scf.yield %tensor7 : tensor<2x!qco.qubit>
    }
    %tensor8, %at0 = qtensor.extract %tensor3[%c0] : tensor<2x!qco.qubit>
    %tensor9, %at1 = qtensor.extract %tensor8[%c1] : tensor<2x!qco.qubit>
    qco.sink %at0 : !qco.qubit
    qco.sink %at1 : !qco.qubit
    qtensor.dealloc %tensor9 : tensor<2x!qco.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  OwningOpRef<ModuleOp> original = moduleOp->clone();
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic&) { return success(); });
  EXPECT_TRUE(failed(runQCOToQCConversion(*moduleOp)));
  EXPECT_TRUE(OperationEquivalence::isEquivalentTo(
      moduleOp->getOperation(), original->getOperation(),
      OperationEquivalence::Flags::None));
}

TEST(QCOToQCRegressionTest, RetainsQubitRegisterName) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  qtensor::QTensorDialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  qco::QCOProgramBuilder builder(&context);
  builder.initialize();
  std::ignore = builder.allocQubitRegister(2, "named_qubits");
  auto moduleOp = builder.finalize();
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));

  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) {
    if (isa<qc::QubitType>(op.getType().getElementType())) {
      allocation = op;
    }
  });
  ASSERT_TRUE(allocation);
  const auto name = allocation->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

TEST(QCOToQCRegressionTest, RetainsDynamicQubitRegisterName) {
  DialectRegistry registry;
  registry.insert<mlir::mqt::MQTDialect, qc::QCDialect, qco::QCODialect,
                  qtensor::QTensorDialect, arith::ArithDialect,
                  func::FuncDialect, memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%size: index) attributes {mqt.entry_point} {
    %reg = qtensor.alloc(%size) {mqt.register_name = "named_qubits"} : tensor<?x!qco.qubit>
    qtensor.dealloc %reg : tensor<?x!qco.qubit>
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));

  memref::AllocOp allocation;
  moduleOp->walk([&](memref::AllocOp op) { allocation = op; });
  ASSERT_TRUE(allocation);
  EXPECT_TRUE(allocation.getType().isDynamicDim(0));
  ASSERT_EQ(allocation.getDynamicSizes().size(), 1);
  EXPECT_EQ(allocation.getDynamicSizes().front(),
            allocation->getBlock()->getArgument(0));
  const auto name = allocation->getAttrOfType<StringAttr>(
      mlir::mqt::MQTDialect::RegisterNameAttrHelper::getNameStr());
  ASSERT_TRUE(name);
  EXPECT_EQ(name.getValue(), "named_qubits");
}

static Value
aliasSafeNestedForLoopCtrlOpWithExtractedQubit(qc::QCProgramBuilder& b) {
  auto reg = b.allocQubitRegister(4);
  auto c0 = arith::ConstantIndexOp::create(b, 0);
  auto control = b.loadQubit(reg.value, c0);
  b.h(control);
  b.scfFor(1, 4, 1, [&](Value iv) {
    auto target = b.loadQubit(reg.value, iv);
    b.h(target);
    // Sequence loads explicitly; function argument evaluation order varies.
    auto loopControl = b.loadQubit(reg.value, c0);
    target = b.loadQubit(reg.value, iv);
    b.cx(loopControl, target);
  });
  auto result = b.allocClassicalBitRegister(1);
  b.measure(b.loadQubit(reg.value, c0), result, 0);
  return result;
}

TEST(QCOToQCRegressionTest, PreservesClassicalIfResult) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
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
      %q2 = qco.reset %arg : !qco.qubit
      qco.yield %then, %q2 : i64, !qco.qubit
    } else args(%arg = %q0) {
      %q2 = qco.reset %arg : !qco.qubit
      qco.yield %else, %q2 : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::IfOp ifOp;
  moduleOp->walk([&](scf::IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getNumResults(), 1);
  EXPECT_TRUE(ifOp.getResult(0).getType().isInteger(64));

  auto main = moduleOp->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), ifOp.getResult(0));
  for (auto& region : ifOp->getRegions()) {
    EXPECT_EQ(llvm::range_size(region.getOps<qc::ResetOp>()), 1);
  }

  bool containsQCOOperations = false;
  moduleOp->walk([&](Operation* operation) {
    containsQCOOperations |=
        operation->getDialect() == context.getLoadedDialect<qco::QCODialect>();
  });
  EXPECT_FALSE(containsQCOOperations);
}

TEST(QCOToQCRegressionTest, PreservesClassicalIndexSwitchResult) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main(%index: index) -> i64
      attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %result, %q1 = qco.index_switch %index -> (i64, !qco.qubit)
    case 0 args(%arg0 = %q0) {
      %q2 = qco.reset %arg0 : !qco.qubit
      %case = arith.constant 1 : i64
      qco.yield %case, %q2 : i64, !qco.qubit
    }
    default args(%arg0 = %q0) {
      %q2 = qco.reset %arg0 : !qco.qubit
      %default = arith.constant 2 : i64
      qco.yield %default, %q2 : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::IndexSwitchOp switchOp;
  moduleOp->walk([&](scf::IndexSwitchOp candidate) { switchOp = candidate; });
  ASSERT_TRUE(switchOp);
  ASSERT_EQ(switchOp.getNumResults(), 1);
  EXPECT_TRUE(switchOp.getResult(0).getType().isInteger(64));

  auto main = moduleOp->lookupSymbol<func::FuncOp>("main");
  ASSERT_TRUE(main);
  auto returnOp = cast<func::ReturnOp>(main.getBody().front().getTerminator());
  EXPECT_EQ(returnOp.getOperand(0), switchOp.getResult(0));
  for (auto& region : switchOp->getRegions()) {
    EXPECT_EQ(llvm::range_size(region.getOps<qc::ResetOp>()), 1);
  }

  bool containsQCOOperations = false;
  moduleOp->walk([&](Operation* operation) {
    containsQCOOperations |=
        operation->getDialect() == context.getLoadedDialect<qco::QCODialect>();
  });
  EXPECT_FALSE(containsQCOOperations);
}

TEST(QCOToQCRegressionTest, PreservesClassicalForLoopState) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    %initial = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %result, %q1 = scf.for %iv = %lb to %ub step %step
        iter_args(%value = %initial, %q = %q0) -> (i64, !qco.qubit) {
      %next = arith.addi %value, %one : i64
      %q2 = qco.reset %q : !qco.qubit
      scf.yield %next, %q2 : i64, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::ForOp loop;
  moduleOp->walk([&](scf::ForOp candidate) { loop = candidate; });
  ASSERT_TRUE(loop);
  EXPECT_EQ(llvm::range_size(loop.getBody()->getOps<qc::ResetOp>()), 1);
  ASSERT_EQ(loop.getInitArgs().size(), 1);
  EXPECT_TRUE(loop.getInitArgs().front().getType().isInteger(64));
  ASSERT_EQ(loop.getNumResults(), 1);
  EXPECT_TRUE(loop.getResult(0).getType().isInteger(64));
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  ASSERT_EQ(yield.getNumOperands(), 1);
  EXPECT_TRUE(yield.getOperand(0).getType().isInteger(64));
}

TEST(QCOToQCRegressionTest, PreservesTypeChangingClassicalWhileState) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                  scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %q0 = qco.alloc : !qco.qubit
    %initial = arith.constant 1.0 : f32
    %result, %q1 = scf.while (%input = %initial, %q = %q0)
        : (f32, !qco.qubit) -> (i64, !qco.qubit) {
      %q2 = qco.h %q : !qco.qubit
      %condition = arith.constant false
      %next = arith.constant 7 : i64
      scf.condition(%condition) %next, %q2 : i64, !qco.qubit
    } do {
    ^bb0(%input: i64, %q: !qco.qubit):
      %q2 = qco.x %q : !qco.qubit
      %next = arith.sitofp %input : i64 to f32
      scf.yield %next, %q2 : f32, !qco.qubit
    }
    qco.sink %q1 : !qco.qubit
    return %result : i64
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  scf::WhileOp loop;
  moduleOp->walk([&](scf::WhileOp candidate) { loop = candidate; });
  ASSERT_TRUE(loop);
  ASSERT_EQ(loop.getInits().size(), 1);
  EXPECT_TRUE(loop.getInits().front().getType().isF32());
  ASSERT_EQ(loop.getNumResults(), 1);
  EXPECT_TRUE(loop.getResult(0).getType().isInteger(64));
  auto condition =
      cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
  ASSERT_EQ(condition.getArgs().size(), 1);
  EXPECT_TRUE(condition.getArgs().front().getType().isInteger(64));
  auto yield = cast<scf::YieldOp>(loop.getAfterBody()->getTerminator());
  ASSERT_EQ(yield.getNumOperands(), 1);
  EXPECT_TRUE(yield.getOperand(0).getType().isF32());
}

TEST(QCOToQCRegressionTest, PreservesResetQubitsInWhileRegions) {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                  func::FuncDialect, scf::SCFDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  func.func @main() {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %out:2 = scf.while (%a = %q0, %b = %q1)
        : (!qco.qubit, !qco.qubit) -> (!qco.qubit, !qco.qubit) {
      %a0 = qco.reset %a : !qco.qubit
      %b0 = qco.reset %b : !qco.qubit
      %condition = arith.constant false
      scf.condition(%condition) %a0, %b0 : !qco.qubit, !qco.qubit
    } do {
    ^bb0(%a: !qco.qubit, %b: !qco.qubit):
      %a0 = qco.reset %a : !qco.qubit
      %b0 = qco.reset %b : !qco.qubit
      scf.yield %a0, %b0 : !qco.qubit, !qco.qubit
    }
    qco.sink %out#0 : !qco.qubit
    qco.sink %out#1 : !qco.qubit
    return
  }
}
)mlir";

  auto moduleOp = parseSourceString<ModuleOp>(source, &context);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOToQCConversion(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  auto function = *moduleOp->getOps<func::FuncOp>().begin();
  auto allocations = llvm::to_vector(function.getOps<qc::AllocOp>());
  ASSERT_EQ(allocations.size(), 2U);
  auto loops = llvm::to_vector(function.getOps<scf::WhileOp>());
  ASSERT_EQ(loops.size(), 1U);
  auto loop = loops.front();
  EXPECT_EQ(loop.getNumOperands(), 0U);
  EXPECT_EQ(loop.getNumResults(), 0U);
  for (auto& region : loop->getRegions()) {
    EXPECT_EQ(region.front().getNumArguments(), 0U);
    auto resets = llvm::to_vector(region.getOps<qc::ResetOp>());
    ASSERT_EQ(resets.size(), 2U);
    EXPECT_EQ(resets[0].getQubit(), allocations[0].getResult());
    EXPECT_EQ(resets[1].getQubit(), allocations[1].getResult());
  }
}

TEST_P(QCOToQCTest, ProgramEquivalence) {
  SCOPED_TRACE(llvm::formatv(
                   "QCOToQC: {0}, original={1}, reference={2}", GetParam().name,
                   ::mqt::test::displayName(GetParam().programBuilder.name),
                   ::mqt::test::displayName(GetParam().referenceBuilder.name))
                   .str());
  const auto& [nameStr, programBuilder, referenceBuilder] = GetParam();
  const auto name = " (" + nameStr + ")";
  ::mqt::test::DeferredPrinter printer;

  auto program = ::mqt::test::buildMLIRProgram(context.get(), programBuilder);
  ASSERT_TRUE(program);
  printer.record(program.get(), "Original QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCOCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized QCO IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(succeeded(runQCOToQCConversion(program.get())));
  printer.record(program.get(), "Converted QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(program.get()).succeeded());
  printer.record(program.get(), "Canonicalized Converted QC IR" + name);
  EXPECT_TRUE(verify(*program).succeeded());

  auto reference =
      ::mqt::test::buildMLIRProgram(context.get(), referenceBuilder);
  ASSERT_TRUE(reference);
  printer.record(reference.get(), "Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(runQCCleanupPipeline(reference.get()).succeeded());
  printer.record(reference.get(), "Canonicalized Reference QC IR" + name);
  EXPECT_TRUE(verify(*reference).succeeded());

  EXPECT_TRUE(
      areModulesEquivalentWithPermutations(program.get(), reference.get()));
}

static SmallVector<QCOToQCTestCase> commonConversionCases() {
  SmallVector<QCOToQCTestCase> cases;
  for (const auto& testCase : ::mqt::test::COMMON_QC_QCO_CONVERSION_CASES) {
    cases.push_back({
        .name = std::get<0>(testCase),
        .programBuilder = std::get<2>(testCase),
        .referenceBuilder = std::get<1>(testCase),
    });
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(CommonCases, QCOToQCTest,
                         testing::ValuesIn(commonConversionCases()));

INSTANTIATE_TEST_SUITE_P(
    SpecialCases, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"AllocDeallocPair",
                        MQT_NAMED_BUILDER(qco::allocSinkPair),
                        MQT_NAMED_BUILDER(qc::allocDeallocPair)},
        QCOToQCTestCase{"InverseDCX", MQT_NAMED_BUILDER(qco::inverseDcx),
                        MQT_NAMED_BUILDER(qc::dcx)},
        QCOToQCTestCase{"InverseMultipleControlledDCX",
                        MQT_NAMED_BUILDER(qco::inverseMultipleControlledDcx),
                        MQT_NAMED_BUILDER(qc::multipleControlledDcx)},
        QCOToQCTestCase{"RepeatedResetAfterSingleOp",
                        MQT_NAMED_BUILDER(qco::repeatedResetAfterSingleOp),
                        MQT_NAMED_BUILDER(qc::resetQubitAfterSingleOp)},
        QCOToQCTestCase{"SimpleIfOp", MQT_NAMED_BUILDER(qco::simpleIf),
                        MQT_NAMED_BUILDER(qc::simpleIf)},
        QCOToQCTestCase{"IfElse", MQT_NAMED_BUILDER(qco::ifElse),
                        MQT_NAMED_BUILDER(qc::ifElse)},
        QCOToQCTestCase{"IfTwoQubits", MQT_NAMED_BUILDER(qco::ifTwoQubits),
                        MQT_NAMED_BUILDER(qc::ifTwoQubits)},
        QCOToQCTestCase{"IfWithMeasurement",
                        MQT_NAMED_BUILDER(qco::ifWithMeasurement),
                        MQT_NAMED_BUILDER(qc::ifWithMeasurement)},
        QCOToQCTestCase{"IfWithCreg", MQT_NAMED_BUILDER(qco::ifWithCreg),
                        MQT_NAMED_BUILDER(qc::ifWithCreg)},
        QCOToQCTestCase{"SimpleIndexSwitchOp",
                        MQT_NAMED_BUILDER(qco::simpleIndexSwitch),
                        MQT_NAMED_BUILDER(qc::simpleIndexSwitch)},
        QCOToQCTestCase{"IndexSwitchMultiCase",
                        MQT_NAMED_BUILDER(qco::indexSwitchMultiCase),
                        MQT_NAMED_BUILDER(qc::indexSwitchMultiCase)},
        QCOToQCTestCase{"NestedForLoopWhileOp",
                        MQT_NAMED_BUILDER(qco::nestedForLoopWhileOp),
                        MQT_NAMED_BUILDER(qc::nestedForLoopWhileOp)},
        QCOToQCTestCase{"NestedForLoopSwitchOp",
                        MQT_NAMED_BUILDER(qco::nestedForLoopSwitchOp),
                        MQT_NAMED_BUILDER(qc::nestedForLoopSwitchOp)},
        QCOToQCTestCase{
            "NestedForLoopCtrlOpWithExtractedQubit",
            MQT_NAMED_BUILDER(qco::nestedForLoopCtrlOpWithExtractedQubit),
            MQT_NAMED_BUILDER(
                aliasSafeNestedForLoopCtrlOpWithExtractedQubit)}));
