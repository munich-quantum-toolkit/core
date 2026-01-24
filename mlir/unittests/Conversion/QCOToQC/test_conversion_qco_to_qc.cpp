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
#include "mlir/Dialect/QC/Builder/QCProgramBuilder.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <functional>
#include <gtest/gtest.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>
#include <string>

using namespace mlir;

class ConversionTest : public ::testing::Test {
protected:
  std::unique_ptr<mlir::MLIRContext> context;
  void SetUp() override {
    // Register all dialects needed for the full compilation pipeline
    DialectRegistry registry;
    registry.insert<mlir::qc::QCDialect, qco::QCODialect, arith::ArithDialect,
                    func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect,
                    tensor::TensorDialect, memref::MemRefDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }
  static void runCanonicalizationPass(ModuleOp module) {
    PassManager pm(module.getContext());
    pm.addPass(createCanonicalizerPass());
    if (pm.run(module).failed()) {
      llvm::errs() << "Failed to run canonicalization passes.\n";
    }
  }

  [[nodiscard]] OwningOpRef<ModuleOp> buildQCIR(
      const std::function<void(mlir::qc::QCProgramBuilder&)>& buildFunc) const {
    mlir::qc::QCProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPass(module.get());
    return module;
  }

  [[nodiscard]] OwningOpRef<ModuleOp> buildQCOIR(
      const std::function<void(qco::QCOProgramBuilder&)>& buildFunc) const {
    qco::QCOProgramBuilder builder(context.get());
    builder.initialize();
    buildFunc(builder);
    auto module = builder.finalize();
    runCanonicalizationPass(module.get());
    return module;
  }
};

static std::string getOutputString(mlir::OwningOpRef<mlir::ModuleOp>& module) {
  std::string outputString;
  llvm::raw_string_ostream os(outputString);

  auto* moduleOp = module->getOperation();
  const auto* qcDialect =
      moduleOp->getContext()->getLoadedDialect<qc::QCDialect>();
  const auto* scfDialect =
      moduleOp->getContext()->getLoadedDialect<scf::SCFDialect>();
  const auto* memrefDialect =
      moduleOp->getContext()->getLoadedDialect<memref::MemRefDialect>();

  moduleOp->walk([&](Operation* op) -> WalkResult {
    const auto* opDialect = op->getDialect();

    // Ignore dealloc operations as the order does not matter
    if (llvm::isa<qc::DeallocOp>(op)) {
      return WalkResult::advance();
    }
    // Only consider operations from the qc dialect, scf dialect or memref
    // dialect or func.call or func.return op
    if (opDialect == qcDialect || opDialect == scfDialect ||
        opDialect == memrefDialect || llvm::isa<func::ReturnOp>(op) ||
        llvm::isa<func::CallOp>(op)) {
      op->print(os);
    }
    return WalkResult::advance();
  });

  os.flush();
  return outputString;
}

TEST_F(ConversionTest, ScfForQCOToQCTest) {
  // Test conversion from qco to qc for scf.for operation
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto scfForRes = b.scfFor(
        0, 2, 1, {q0},
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
    b.scfFor(0, 2, 1, [&](Value /*iv*/) {
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

TEST_F(ConversionTest, ScfCtrlQCOtoQCTest) {
  // Test conversion from qco to qc for scf.for operation with nested ctrl
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto q0 = b.allocQubit();
    auto control = b.allocQubit();
    auto scfForRes =
        b.scfFor(0, 2, 1, {q0, control},
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
    b.scfFor(0, 2, 1, [&](Value) {
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

TEST_F(ConversionTest, ScfForTensorQCOtoQCTest) {
  // Test conversion from qco to qc for scf.for operation with a tensor
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4);
    auto tensor = b.tensorFromElements(reg);
    auto scfForRes = b.scfFor(
        0, 3, 1, {tensor},
        [&](Value iv, ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto extractedQubit = b.tensorExtract(iterArgs[0], iv);
          auto q4 = b.h(extractedQubit);
          auto newTensor = b.tensorInsert(q4, iterArgs[0], iv);
          return {newTensor};
        });
    auto extractedq0 = b.tensorExtract(scfForRes[0], 0);
    auto extractedq1 = b.tensorExtract(scfForRes[0], 1);
    auto extractedq2 = b.tensorExtract(scfForRes[0], 2);
    auto extractedq3 = b.tensorExtract(scfForRes[0], 3);
    b.swap(extractedq0, extractedq1);
    b.swap(extractedq2, extractedq3);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL()
        << "Conversion error during QCO-QC conversion for scf.for with tensor";
  }
  // Run the canonicalizer again to remove the additional constants
  pm.clear();
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during canonicalization";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg = b.allocQubitRegister(4);
    auto memref = b.memrefAlloc(reg);
    b.scfFor(0, 3, 1, [&](Value iv) {
      auto extractedQubit = b.memrefLoad(memref, iv);
      b.h(extractedQubit);
    });
    b.swap(reg[0], reg[1]);
    b.swap(reg[2], reg[3]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);
  ASSERT_EQ(outputString, checkString);
}

TEST_F(ConversionTest, ScfForNestedTensorQCOtoQCTest) {
  // Test conversion from qco to qc for scf.for operation with a nested tensor
  auto input = buildQCOIR([](mlir::qco::QCOProgramBuilder& b) {
    auto reg0 = b.allocQubitRegister(4, "q0");
    auto reg1 = b.allocQubitRegister(4, "q1");
    auto tensor0 = b.tensorFromElements(reg0);
    auto scfForRes = b.scfFor(
        0, 3, 1, {tensor0, reg1[0], reg1[1], reg1[2], reg1[3]},
        [&](Value iv, ValueRange iterArgs) -> llvm::SmallVector<Value> {
          auto extractedQubit = b.tensorExtract(iterArgs[0], iv);
          auto outerQubit = b.x(extractedQubit);
          auto tensor1 = b.tensorFromElements(
              {iterArgs[1], iterArgs[2], iterArgs[3], iterArgs[4]});
          auto innerResults = b.scfFor(
              0, 3, 1, {tensor1, outerQubit},
              [&](Value innerIv,
                  ValueRange innerIterArgs) -> llvm::SmallVector<Value> {
                auto innerQubit = b.tensorExtract(innerIterArgs[0], innerIv);
                auto ctrlOp = b.cx(innerIterArgs[1], innerQubit);
                auto innerTensor =
                    b.tensorInsert(ctrlOp.second, innerIterArgs[0], innerIv);
                return {innerTensor, ctrlOp.first};
              });
          auto extractedq0 = b.tensorExtract(innerResults[0], 0);
          auto extractedq1 = b.tensorExtract(innerResults[0], 1);
          auto extractedq2 = b.tensorExtract(innerResults[0], 2);
          auto extractedq3 = b.tensorExtract(innerResults[0], 3);
          auto tensor2 = b.tensorInsert(innerResults[1], iterArgs[0], iv);
          return {tensor2, extractedq0, extractedq1, extractedq2, extractedq3};
        });
    auto extractedq0 = b.tensorExtract(scfForRes[0], 0);
    auto extractedq1 = b.tensorExtract(scfForRes[0], 1);
    auto extractedq2 = b.tensorExtract(scfForRes[0], 2);
    auto extractedq3 = b.tensorExtract(scfForRes[0], 3);
    b.swap(extractedq0, extractedq1);
    b.swap(extractedq2, extractedq3);
  });

  PassManager pm(context.get());
  pm.addPass(createQCOToQC());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Conversion error during QCO-QC Conversion for scf.for with "
              "nested tensor";
  }
  // Run the canonicalizer again to remove the additional constants
  pm.clear();
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(input.get()))) {
    FAIL() << "Error during canonicalization";
  }

  auto expectedOutput = buildQCIR([](mlir::qc::QCProgramBuilder& b) {
    auto reg0 = b.allocQubitRegister(4, "q0");
    auto reg1 = b.allocQubitRegister(4, "q1");
    auto memref0 = b.memrefAlloc(reg0);
    b.scfFor(0, 3, 1, [&](Value iv) {
      auto extractedQubit = b.memrefLoad(memref0, iv);
      b.x(extractedQubit);
      auto memref1 = b.memrefAlloc(reg1);
      b.scfFor(0, 3, 1, [&](Value iv2) {
        auto q1 = b.memrefLoad(memref1, iv2);
        b.cx(extractedQubit, q1);
      });
    });
    b.swap(reg0[0], reg0[1]);
    b.swap(reg0[2], reg0[3]);
  });

  const auto outputString = getOutputString(input);
  const auto checkString = getOutputString(expectedOutput);
  ASSERT_EQ(outputString, checkString);
}
