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
#include "mlir/IR/Builders.h"

#include <gtest/gtest.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>

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
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](OpBuilder& b) {
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
      static_cast<mlir::qc::QCProgramBuilder&>(b).x(q0);
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
    });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes = b.scfFor(
        c0, c2, c1, ValueRange{q0},
        [&](OpBuilder& b, Location, Value, ValueRange iterArgs) -> ValueRange {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(iterArgs[0]);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).x(q1);
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q2);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(
              ValueRange{q3});
          return {q3};
        });
    b.h(scfForRes[0]);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfForTest2) {
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    auto scfForRes = b.scfFor(
        c0, c2, c1, ValueRange{q0},
        [&](OpBuilder& b, Location, Value, ValueRange iterArgs) -> ValueRange {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(iterArgs[0]);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).x(q1);
          auto q3 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q2);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(
              ValueRange{q3});
          return {q3};
        });
    b.h(scfForRes[0]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto c0 = b.arithConstantIndex(0);
    auto c1 = b.arithConstantIndex(1);
    auto c2 = b.arithConstantIndex(2);
    b.scfFor(c0, c2, c1, [&](OpBuilder& b) {
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
      static_cast<mlir::qc::QCProgramBuilder&>(b).x(q0);
      static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
    });
    b.h(q0);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfWhileTest) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();

    b.scfWhile(
        [&](OpBuilder& b) {
          auto measure =
              static_cast<mlir::qc::QCProgramBuilder&>(b).measure(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).scfCondition(measure);
        },
        [&](OpBuilder& b) {
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto scfWhileResult = b.scfWhile(
        ValueRange{q0},
        [&](OpBuilder& b, Location, ValueRange iterArgs) {
          auto measure = static_cast<mlir::qco::QCOProgramBuilder&>(b).measure(
              iterArgs[0]);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfCondition(
              measure.second, ValueRange{measure.first});
          return ValueRange{measure.first};
        },
        [&](OpBuilder& b, Location, ValueRange iterArgs) {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(iterArgs[0]);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(q1);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield({q2});
          return ValueRange{q2};
        });
    b.h(scfWhileResult[0]);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfWhileTest2) {

  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto scfWhileResult = b.scfWhile(
        ValueRange{q0},
        [&](OpBuilder& b, Location, ValueRange iterArgs) {
          auto measure = static_cast<mlir::qco::QCOProgramBuilder&>(b).measure(
              iterArgs[0]);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfCondition(
              measure.second, ValueRange{measure.first});
          return ValueRange{measure.first};
        },
        [&](OpBuilder& b, Location, ValueRange iterArgs) {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(iterArgs[0]);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(q1);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield({q2});
          return ValueRange{q2};
        });
    b.h(scfWhileResult[0]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    b.scfWhile(
        [&](OpBuilder& b) {
          auto measure =
              static_cast<mlir::qc::QCProgramBuilder&>(b).measure(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).scfCondition(measure);
        },
        [&](OpBuilder& b) {
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        });
    b.h(q0);
  });
  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfTest) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    b.scfIf(
        measure,
        [&](OpBuilder& b) {
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        },
        [&](OpBuilder& b) {
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
        });
    b.h(q0);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(input.get()))) {
  }

  auto expectedOutput = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    auto scfIfResult = b.scfIf(
        measure.second, ValueRange{measure.first},
        [&](OpBuilder& b, Location) -> ValueRange {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(measure.first);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(q1);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q2);
          return {q2};
        },
        [&](OpBuilder& b, Location) -> ValueRange {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).y(measure.first);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q1);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q2);
          return {q2};
        });
    b.h(scfIfResult[0]);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfIfTest2) {

  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    auto scfIfResult = b.scfIf(
        measure.second, ValueRange{measure.first},
        [&](OpBuilder& b, Location) -> ValueRange {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).h(measure.first);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).y(q1);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q2);
          return {q2};
        },
        [&](OpBuilder& b, Location) -> ValueRange {
          auto q1 =
              static_cast<mlir::qco::QCOProgramBuilder&>(b).y(measure.first);
          auto q2 = static_cast<mlir::qco::QCOProgramBuilder&>(b).h(q1);
          static_cast<mlir::qco::QCOProgramBuilder&>(b).scfYield(q2);
          return {q2};
        });
    b.h(scfIfResult[0]);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto measure = b.measure(q0);
    b.scfIf(
        measure,
        [&](OpBuilder& b) {
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
        },
        [&](OpBuilder& b) {
          static_cast<mlir::qc::QCProgramBuilder&>(b).y(q0);
          static_cast<mlir::qc::QCProgramBuilder&>(b).h(q0);
        });
    b.h(q0);
  });

  const auto outputString = getOutputString(&input);
  const auto checkString = getOutputString(&expectedOutput);

  ASSERT_EQ(outputString, checkString);
}
