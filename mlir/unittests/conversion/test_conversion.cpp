/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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

#include <gtest/gtest.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
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

static std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>* module) {
  std::string outputString;
  llvm::raw_string_ostream os(outputString);
  (*module)->print(os);
  os.flush();
  return outputString;
}

TEST_F(ConversionTest, ScfForTest) {
  // Test conversion from qc to qco for scf.for operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](OpBuilder& b) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      static_cast<mlir::qc::QCProgramBuilder&>(b).x(q0);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
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
        c0, c2, c1, ValueRange{q0},
        [&](OpBuilder& b, Value, ValueRange iterArgs) {
          auto
              q1 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(iterArgs[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).x(q1);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q2);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(
              ValueRange{q3});
          return q3;
        });
    b.h(scfForRes[0]);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfForTest2) {
  // Test conversion from qco to qc for scf.for operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes = b.scfFor(
        c0, c2, c1, ValueRange{q0},
        [&](OpBuilder& b, Value, ValueRange iterArgs) {
          auto
              q1 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(iterArgs[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).x(
              q1); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(
              q2); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(
              ValueRange{q3});
          return q3;
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
    b.scfFor(c0, c2, c1, [&](OpBuilder& b) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      static_cast<mlir::qc::QCProgramBuilder&>(b).x(q0);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
    });
    b.h(q0);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfWhileTest) {
  // Test conversion from qc to qco for scf.while operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.scfWhile(
        [&](OpBuilder& b) {
          auto
              measureResult = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qc::QCProgramBuilder&>(b).measure(q0);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).scfCondition(
              measureResult);
        },
        [&](OpBuilder&
                b) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
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
        [&](OpBuilder& b, ValueRange iterArgs) {
          auto
              [q1,
               measureResult] = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).measure(
                  iterArgs
                      [0]); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfCondition(
              measureResult, ValueRange{q1});
          return q1;
        },
        [&](OpBuilder& b, ValueRange iterArgs) {
          auto
              q1 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(
                  iterArgs
                      [0]); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(
              q1); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield({q2});
          return q2;
        });
    b.h(scfWhileResult[0]);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfWhileTest2) {
  // Test conversion from qco to qc for scf.while operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto scfWhileResult = b.scfWhile(
        ValueRange{q0},
        [&](OpBuilder& b, ValueRange iterArgs) {
          auto
              [q1,
               measureResult] = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).measure(
                  iterArgs
                      [0]); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfCondition(
              measureResult, ValueRange{q1});
          return q1;
        },
        [&](OpBuilder& b, ValueRange iterArgs) {
          auto
              q1 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(
                  iterArgs
                      [0]); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(
              q1); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield({q2});
          return q2;
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
        [&](OpBuilder& b) {
          auto
              measureResult = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qc::QCProgramBuilder&>(b).measure(
                  q0); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).scfCondition(
              measureResult);
        },
        [&](OpBuilder&
                b) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(
              q0); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        });
    b.h(q0);
  });
  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfTest) {
  // Test conversion from qc to qco for scf.if operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    b.scfIf(
        measure,
        [&](OpBuilder&
                b) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(
              q0); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        },
        [&](OpBuilder&
                b) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(
              q0); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
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
        measureResult, ValueRange{q1},
        [&](OpBuilder& b) {
          auto
              q2 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q1);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(
              q2); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q3);
          return q3;
        },
        [&](OpBuilder& b) {
          auto
              q2 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).y(
                  q1); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(
              q2); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q3);
          return q3;
        });
    b.h(scfIfResult[0]);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfTest2) {
  // Test conversion from qco to qc for scf.if operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto [q1, measureResult] = b.measure(q0);
    auto scfIfResult = b.scfIf(
        measureResult, ValueRange{q1},
        [&](OpBuilder& b) {
          auto
              q2 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q1);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(
              q2); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q3);
          return q3;
        },
        [&](OpBuilder& b) {
          auto
              q2 = // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
              static_cast<mlir::qco::QCOProgramBuilder&>(b).y(
                  q1); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(
              q2); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q3);
          return q3;
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
        [&](OpBuilder&
                b) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(
              q0); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        },
        [&](OpBuilder&
                b) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(
              q0); // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
        });
    b.h(q0);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, FuncFuncTest) {
  // Test conversion from qc to qco for func.func operation
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.funcCall("test", q0);
    b.h(q0);
    b.funcFunc(
        "test", q0.getType(),
        [&](OpBuilder& b,
            ValueRange
                args) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(args[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(args[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).funcReturn();
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
    b.funcFunc(
        "test", q0.getType(), q0.getType(), [&](OpBuilder& b, ValueRange args) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(args[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(q2);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).funcReturn(q3);
        });
  });
  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, FuncFuncTest2) {
  // Test conversion from qco to qc for func.func operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto q1 = b.funcCall("test", q0);
    b.h(q1[0]);
    b.funcFunc(
        "test", q0.getType(), q0.getType(), [&](OpBuilder& b, ValueRange args) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(args[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(q2);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qco::QCOProgramBuilder&>(b).funcReturn(q3);
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
    b.funcFunc(
        "test", q0.getType(),
        [&](OpBuilder& b,
            ValueRange
                args) { // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(args[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(args[0]);
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
          static_cast<mlir::qc::QCProgramBuilder&>(b).funcReturn();
        });
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}
