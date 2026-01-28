/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToJeff/QCToJeff.h"
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"

#include <functional>
#include <gtest/gtest.h>
#include <jeff/IR/JeffDialect.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

using namespace mlir;

class QCToJeffConversionTest : public ::testing::Test {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, jeff::JeffDialect,
                    mlir::qc::QCDialect>();

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

  static std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>& mod) {
    std::string outputString;
    llvm::raw_string_ostream outputStream(outputString);
    mod->print(outputStream);
    outputStream.flush();
    return outputString;
  }
};

TEST_F(QCToJeffConversionTest, Measure) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.measure(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.qubit_measure_nd"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, Id) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.id(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.i"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, X) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.x(q);
    b.x(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, Y) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.y(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.y"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, Z) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.z(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.z"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, H) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.h(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.h"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, S) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.s(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.s"), std::string::npos);
  ASSERT_NE(outputString.find("is_adjoint = false"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, Sdg) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.sdg(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.s"), std::string::npos);
  ASSERT_NE(outputString.find("is_adjoint = true"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, T) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.t(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.t"), std::string::npos);
  ASSERT_NE(outputString.find("is_adjoint = false"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, Tdg) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.tdg(q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.t"), std::string::npos);
  ASSERT_NE(outputString.find("is_adjoint = true"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, RX) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.rx(0.5, q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.rx"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, RY) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.ry(0.5, q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.ry"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, RZ) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.rz(0.5, q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.rz"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, P) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(1, "q");
    const auto q = reg[0];
    b.p(0.5, q);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.r1"), std::string::npos);
}

TEST_F(QCToJeffConversionTest, Bell) {
  auto input = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(2, "q");
    const auto q0 = reg[0];
    const auto q1 = reg[1];
    b.h(q0);
    b.cx(q0, q1);
  });

  PassManager pm(context.get());
  pm.addPass(createQCToJeff());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during QC-to-Jeff conversion";
  }

  const auto outputString = getOutputString(input);

  // ASSERT_EQ(outputString, "test");

  ASSERT_NE(outputString.find("jeff.h"), std::string::npos);
  ASSERT_NE(outputString.find("jeff.x"), std::string::npos);
  ASSERT_NE(outputString.find("num_ctrls = 1"), std::string::npos);
}
