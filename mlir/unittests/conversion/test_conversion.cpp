/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <functional>
#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

using namespace mlir;

class ConversionTest : public ::testing::Test {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> buildQCIR(
      const std::function<void(mlir::qc::QCProgramBuilder&)>& buildFunc) const {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    return module;
  }
  [[nodiscard]] OwningOpRef<ModuleOp> buildQCOIR(
      const std::function<void(qco::QCOProgramBuilder&)>& buildFunc) const {
    qco::QCOProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    return module;
  }
};

static std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>& module) {
  std::string outputString;
  llvm::raw_string_ostream os(outputString);
  module->print(os);
  os.flush();
  return outputString;
}

TEST_F(ConversionTest, ScfForQCToQCOTest) {
  // Test conversion from qc to qco for scf.for operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](Value /*iv*/) {
      b.h(q0);
      b.x(q0);
      b.h(q0);
    });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QC-QCO conversion for scf.for";
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes = b.scfFor(
        c0, c2, c1, {q0},
        [&](Value /*iv*/, ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto q1 = b.h(iterArgs[0]);
          auto q2 = b.x(q1);
          auto q3 = b.h(q2);
          return {q3};
        });
    b.h(scfForRes[0]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfForQCOToQCTest) {
  // Test conversion from qco to qc for scf.for operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes = b.scfFor(
        c0, c2, c1, {q0},
        [&](Value /*iv*/, ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto q1 = b.h(iterArgs[0]);
          auto q2 = b.x(q1);
          auto q3 = b.h(q2);
          return {q3};
        });
    b.h(scfForRes[0]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QCO-QC conversion for scf.for";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](Value /*iv*/) {
      b.h(q0);
      b.x(q0);
      b.h(q0);
    });
    b.h(q0);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfWhileQCToQCOTest) {
  // Test conversion from qc to qco for scf.while operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.scfWhile(
        [&] {
          auto measureResult = b.measure(q0);
          b.scfCondition(measureResult);
        },
        [&] {
          b.h(q0);
          b.y(q0);
        });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QC-QCO conversion for scf.while";
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto scfWhileResult = b.scfWhile(
        ValueRange{q0},
        [&](ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto [q1, measureResult] = b.measure(iterArgs[0]);
          b.scfCondition(measureResult, q1);
          return {q1};
        },
        [&](ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto q1 = b.h(iterArgs[0]);
          auto q2 = b.y(q1);
          return {q2};
        });
    b.h(scfWhileResult[0]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfWhileQCOToQCTest) {
  // Test conversion from qco to qc for scf.while operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto scfWhileResult = b.scfWhile(
        ValueRange{q0},
        [&](ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto [q1, measureResult] = b.measure(iterArgs[0]);
          b.scfCondition(measureResult, q1);
          return {q1};
        },
        [&](ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto q1 = b.h(iterArgs[0]);
          auto q2 = b.y(q1);
          return {q2};
        });
    b.h(scfWhileResult[0]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QCO-QC conversion for scf.while";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.scfWhile(
        [&] {
          auto measureResult = b.measure(q0);

          b.scfCondition(measureResult);
        },
        [&] {
          b.h(q0);
          b.y(q0);
        });
    b.h(q0);
  });
  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfQCToQCOTest) {
  // Test conversion from qc to qco for scf.if operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    b.scfIf(
        measure,
        [&] {
          b.h(q0);
          b.y(q0);
        },
        [&] {
          b.y(q0);
          b.h(q0);
        });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QC-QCO conversion for scf.if";
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto [q1, measureResult] = b.measure(q0);
    auto scfIfResult = b.scfIf(
        measureResult, {q1},
        [&]() -> llvm::SmallVector<Value> {
          auto q2 = b.h(q1);
          auto q3 = b.y(q2);
          return {q3};
        },
        [&]() -> llvm::SmallVector<Value> {
          auto q2 = b.y(q1);
          auto q3 = b.h(q2);
          return {q3};
        });
    b.h(scfIfResult[0]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfQCOToQCTest) {
  // Test conversion from qco to qc for scf.if operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto [q1, measureResult] = b.measure(q0);
    auto scfIfResult = b.scfIf(
        measureResult, {q1},
        [&]() -> llvm::SmallVector<Value> {
          auto q2 = b.h(q1);
          auto q3 = b.y(q2);
          return {q3};
        },
        [&]() -> llvm::SmallVector<Value> {
          auto q2 = b.y(q1);
          auto q3 = b.h(q2);
          return {q3};
        });
    b.h(scfIfResult[0]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QCO-QC conversion for scf.if";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    b.scfIf(
        measure,
        [&] {
          b.h(q0);
          b.y(q0);
        },
        [&] {
          b.y(q0);
          b.h(q0);
        });
    b.h(q0);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfEmptyElseTest) {
  // Test conversion from qc to qco for scf.if operation without an else body
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    b.scfIf(measure, [&] {
      b.h(q0);
      b.y(q0);
    });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QC-QCO conversion for scf.if";
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto [q1, measureResult] = b.measure(q0);
    auto scfIfResult = b.scfIf(
        measureResult, {q1},
        [&]() -> llvm::SmallVector<Value> {
          auto q2 = b.h(q1);
          auto q3 = b.y(q2);
          return {q3};
        },
        [&]() -> llvm::SmallVector<Value> { return {q1}; });
    b.h(scfIfResult[0]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, FuncFuncQCToQCOTest) {
  // Test conversion from qc to qco for func.func operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.funcCall("test", q0);
    b.h(q0);
    b.funcFunc("test", q0.getType(), [&](ValueRange args) {
      b.h(args[0]);
      b.y(args[0]);
    });
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QC-QCO conversion for func.func";
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto q1 = b.funcCall("test", q0);
    b.h(q1[0]);
    b.funcFunc("test", q0.getType(), q0.getType(),
               [&](ValueRange args) -> llvm::SmallVector<Value> {
                 auto q2 = b.h(args[0]);
                 auto q3 = b.y(q2);
                 return {q3};
               });
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, FuncFuncQCOToQCTest) {
  // Test conversion from qco to qc for func.func operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto q1 = b.funcCall("test", q0);
    b.h(q1[0]);
    b.funcFunc("test", q0.getType(), q0.getType(),
               [&](ValueRange args) -> llvm::SmallVector<Value> {
                 auto q2 = b.h(args[0]);
                 auto q3 = b.y(q2);
                 return {q3};
               });
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QCO-QC conversion for func.func";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.funcCall("test", q0);
    b.h(q0);
    b.funcFunc("test", q0.getType(), [&](ValueRange args) {
      b.h(args[0]);
      b.y(args[0]);
    });
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfCtrlQCtoQCOTest) {
  // Test conversion from qc to qco for scf.for operation with nested ctrl
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto control = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](Value) {
      b.ctrl(control, [&] { b.h(q0); });
      b.x(q0);
      b.h(q0);
    });
    b.h(control);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QC-QCO conversion for scf nested";
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto control = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes =
        b.scfFor(c0, c2, c1, {q0, control},
                 [&](Value, ValueRange iterArgs) -> llvm::SmallVector<Value> {
                   auto [controls, targets] = b.ctrl(
                       iterArgs[1], iterArgs[0],
                       [&](ValueRange targets) -> llvm::SmallVector<Value> {
                         auto target = b.h(targets[0]);
                         return {target};
                       });
                   auto q1 = b.x(targets[0]);
                   auto q2 = b.h(q1);
                   return {q2, controls[0]};
                 });

    b.h(scfForRes[1]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfCtrlQCOtoQCTest) {
  // Test conversion from qco to qc for scf.for operation with nested ctrl
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto control = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes =
        b.scfFor(c0, c2, c1, {q0, control},
                 [&](Value, ValueRange iterArgs) -> llvm::SmallVector<Value> {
                   auto [controls, targets] = b.ctrl(
                       iterArgs[1], iterArgs[0],
                       [&](ValueRange targets) -> llvm::SmallVector<Value> {
                         auto target = b.h(targets[0]);
                         return {target};
                       });
                   auto q1 = b.x(targets[0]);
                   auto q2 = b.h(q1);
                   return {q2, controls[0]};
                 });

    b.h(scfForRes[1]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QCO-QC conversion for scf nested";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto control = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](Value) {
      b.ctrl(control, [&] { b.h(q0); });
      b.x(q0);
      b.h(q0);
    });
    b.h(control);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);

  ASSERT_EQ(outputString, checkString);
}
