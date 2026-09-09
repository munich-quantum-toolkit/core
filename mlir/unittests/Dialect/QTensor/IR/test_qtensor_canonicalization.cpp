/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Support/Passes.h"

#include <gtest/gtest.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

#include <array>
#include <cstddef>

using namespace mlir;
using namespace mlir::qco;

namespace {

class QTensorCanonicalizationTest : public testing::Test {
protected:
  MLIRContext context_;

  void SetUp() override {
    context_.loadDialect<qtensor::QTensorDialect, func::FuncDialect>();
  }
};

TEST_F(QTensorCanonicalizationTest,
       CanonicalizesConstantIndexQTensorIfToScalarQubits) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1) -> i1 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %tensor0 = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
        %flag, %tensor1 = qco.if %condition
            args(%arg0 = %tensor0) -> (i1, tensor<3x!qco.qubit>) {
          %tensor2, %q0 = qtensor.extract %arg0[%c0]
              : tensor<3x!qco.qubit>
          %tensor3, %q1 = qtensor.extract %tensor2[%c1]
              : tensor<3x!qco.qubit>
          %q2, %q3 = qco.swap %q0, %q1
              : !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit
          %tensor4 = qtensor.insert %q3 into %tensor3[%c1]
              : tensor<3x!qco.qubit>
          %tensor5 = qtensor.insert %q2 into %tensor4[%c0]
              : tensor<3x!qco.qubit>
          %true = arith.constant true
          qco.yield %true, %tensor5 : i1, tensor<3x!qco.qubit>
        } else args(%arg0 = %tensor0) {
          %tensor2, %q0 = qtensor.extract %arg0[%c2]
              : tensor<3x!qco.qubit>
          %q1 = qco.z %q0 : !qco.qubit -> !qco.qubit
          %tensor3 = qtensor.insert %q1 into %tensor2[%c2]
              : tensor<3x!qco.qubit>
          %false = arith.constant false
          qco.yield %false, %tensor3 : i1, tensor<3x!qco.qubit>
        } {test.marker = "preserved"}
        qtensor.dealloc %tensor1 : tensor<3x!qco.qubit>
        return %flag : i1
      }
    }
  )mlir";

  auto moduleOp = parseSourceString<ModuleOp>(mlirCode, &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  PassManager pm(&context_);
  pm.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(pm.run(*moduleOp)));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  IfOp ifOp;
  moduleOp->walk([&](IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getClassicalResults().size(), 1);
  ASSERT_EQ(ifOp.getQubits().size(), 3);
  ASSERT_EQ(ifOp.getLinearResults().size(), 3);
  EXPECT_EQ(
      cast<StringAttr>(ifOp->getDiscardableAttr("test.marker")).getValue(),
      "preserved");
  EXPECT_TRUE(llvm::all_of(ifOp.getQubits(), [](Value value) {
    return isa<QubitType>(value.getType());
  }));
  EXPECT_TRUE(llvm::all_of(ifOp.getLinearResults(), [](Value value) {
    return isa<QubitType>(value.getType());
  }));

  size_t nestedExtracts = 0;
  size_t nestedInserts = 0;
  size_t swaps = 0;
  size_t zs = 0;
  ifOp->walk([&](Operation* operation) {
    nestedExtracts += isa<qtensor::ExtractOp>(operation);
    nestedInserts += isa<qtensor::InsertOp>(operation);
    swaps += isa<SWAPOp>(operation);
    zs += isa<ZOp>(operation);
  });
  EXPECT_EQ(nestedExtracts, 0);
  EXPECT_EQ(nestedInserts, 0);
  EXPECT_EQ(swaps, 1);
  EXPECT_EQ(zs, 1);

  size_t extracts = 0;
  size_t inserts = 0;
  moduleOp->walk([&](qtensor::ExtractOp) { ++extracts; });
  moduleOp->walk([&](qtensor::InsertOp) { ++inserts; });
  EXPECT_EQ(extracts, 3);
  EXPECT_EQ(inserts, 3);
}

TEST_F(QTensorCanonicalizationTest, ScalarizesOnlyAccessedQTensorElements) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1) {
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %tensor0 = qtensor.alloc(%c3) : tensor<3x!qco.qubit>
        %tensor1 = qco.if %condition
            args(%arg0 = %tensor0) -> (tensor<3x!qco.qubit>) {
          %tensor2, %q0 = qtensor.extract %arg0[%c1]
              : tensor<3x!qco.qubit>
          %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
          %tensor3 = qtensor.insert %q1 into %tensor2[%c1]
              : tensor<3x!qco.qubit>
          qco.yield %tensor3 : tensor<3x!qco.qubit>
        } else args(%arg0 = %tensor0) {
          qco.yield %arg0 : tensor<3x!qco.qubit>
        }
        qtensor.dealloc %tensor1 : tensor<3x!qco.qubit>
        return
      }
    }
  )mlir";

  auto moduleOp = parseSourceString<ModuleOp>(mlirCode, &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(moduleOp.get())));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  IfOp ifOp;
  moduleOp->walk([&](IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getQubits().size(), 1);
  ASSERT_EQ(ifOp.getLinearResults().size(), 1);
  EXPECT_TRUE(llvm::all_of(ifOp.getQubits(), [](Value value) {
    return isa<QubitType>(value.getType());
  }));

  auto thenValues = ifOp.thenYield().getTargets();
  auto elseValues = ifOp.elseYield().getTargets();
  ASSERT_EQ(thenValues.size(), 1);
  ASSERT_EQ(elseValues.size(), 1);
  EXPECT_TRUE(isa<XOp>(thenValues[0].getDefiningOp()));
  EXPECT_EQ(elseValues[0], ifOp.elseBlock()->getArgument(0));

  size_t extracts = 0;
  size_t inserts = 0;
  moduleOp->walk([&](qtensor::ExtractOp) { ++extracts; });
  moduleOp->walk([&](qtensor::InsertOp) { ++inserts; });
  EXPECT_EQ(extracts, 1);
  EXPECT_EQ(inserts, 1);
}

TEST_F(QTensorCanonicalizationTest, ForwardsUnaccessedQTensorAroundIf) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1) {
        %c2 = arith.constant 2 : index
        %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
        %q0 = qco.alloc : !qco.qubit
        %tensor1, %q1 = qco.if %condition
            args(%tensor = %tensor0, %q = %q0)
            -> (tensor<2x!qco.qubit>, !qco.qubit) {
          %q2 = qco.h %q : !qco.qubit -> !qco.qubit
          qco.yield %tensor, %q2 : tensor<2x!qco.qubit>, !qco.qubit
        } else args(%tensor = %tensor0, %q = %q0) {
          qco.yield %tensor, %q : tensor<2x!qco.qubit>, !qco.qubit
        }
        qtensor.dealloc %tensor1 : tensor<2x!qco.qubit>
        qco.sink %q1 : !qco.qubit
        return
      }
    }
  )mlir";

  auto moduleOp = parseSourceString<ModuleOp>(mlirCode, &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(moduleOp.get())));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  IfOp ifOp;
  moduleOp->walk([&](IfOp candidate) { ifOp = candidate; });
  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getQubits().size(), 1);
  ASSERT_EQ(ifOp.getLinearResults().size(), 1);
  EXPECT_TRUE(isa<QubitType>(ifOp.getQubits()[0].getType()));
  EXPECT_TRUE(isa<QubitType>(ifOp.getLinearResults()[0].getType()));

  auto thenValues = ifOp.thenYield().getTargets();
  auto elseValues = ifOp.elseYield().getTargets();
  ASSERT_EQ(thenValues.size(), 1);
  ASSERT_EQ(elseValues.size(), 1);
  EXPECT_TRUE(isa<HOp>(thenValues[0].getDefiningOp()));
  for (auto [index, value] : llvm::enumerate(elseValues)) {
    EXPECT_EQ(value, ifOp.elseBlock()->getArgument(index));
  }
}

TEST_F(QTensorCanonicalizationTest,
       PreservesInterleavedResultOrderWhenScalarizingQTensors) {
  constexpr StringLiteral mlirCode = R"mlir(
    module {
      func.func @main(%condition: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %tensorA0 = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
        %middle0 = qco.alloc : !qco.qubit
        %tensorB0 = qtensor.alloc(%c1) : tensor<1x!qco.qubit>
        %tensorA1, %middle1, %tensorB1 =
            qco.if %condition
                args(%tensorA = %tensorA0, %middle = %middle0,
                     %tensorB = %tensorB0)
                -> (tensor<1x!qco.qubit>, !qco.qubit,
                    tensor<1x!qco.qubit>) {
          %tensorA2, %tensorAQubit = qtensor.extract %tensorA[%c0]
              : tensor<1x!qco.qubit>
          %tensorAQubitOut = qco.x %tensorAQubit
              : !qco.qubit -> !qco.qubit
          %tensorA3 = qtensor.insert %tensorAQubitOut into %tensorA2[%c0]
              : tensor<1x!qco.qubit>
          %middleOut = qco.y %middle : !qco.qubit -> !qco.qubit
          %tensorB2, %tensorBQubit = qtensor.extract %tensorB[%c0]
              : tensor<1x!qco.qubit>
          %tensorBQubitOut = qco.z %tensorBQubit
              : !qco.qubit -> !qco.qubit
          %tensorB3 = qtensor.insert %tensorBQubitOut into %tensorB2[%c0]
              : tensor<1x!qco.qubit>
          qco.yield %tensorA3, %middleOut, %tensorB3
              : tensor<1x!qco.qubit>, !qco.qubit, tensor<1x!qco.qubit>
        } else args(%tensorA = %tensorA0, %middle = %middle0,
                    %tensorB = %tensorB0) {
          qco.yield %tensorA, %middle, %tensorB
              : tensor<1x!qco.qubit>, !qco.qubit, tensor<1x!qco.qubit>
        }
        %middle2 = qco.t %middle1 : !qco.qubit -> !qco.qubit
        qtensor.dealloc %tensorA1 : tensor<1x!qco.qubit>
        qco.sink %middle2 : !qco.qubit
        qtensor.dealloc %tensorB1 : tensor<1x!qco.qubit>
        return
      }
    }
  )mlir";

  auto moduleOp = parseSourceString<ModuleOp>(mlirCode, &context_);
  ASSERT_TRUE(moduleOp);
  ASSERT_TRUE(succeeded(verify(*moduleOp)));
  ASSERT_TRUE(succeeded(runQCOCleanupPipeline(moduleOp.get())));
  ASSERT_TRUE(succeeded(verify(*moduleOp)));

  IfOp ifOp;
  TOp postMiddle;
  SmallVector<Value> insertedScalars;
  moduleOp->walk([&](IfOp candidate) { ifOp = candidate; });
  moduleOp->walk([&](TOp candidate) { postMiddle = candidate; });
  moduleOp->walk([&](qtensor::InsertOp insert) {
    insertedScalars.push_back(insert.getScalar());
  });

  ASSERT_TRUE(ifOp);
  ASSERT_EQ(ifOp.getQubits().size(), 3);
  ASSERT_EQ(ifOp.getLinearResults().size(), 3);
  EXPECT_TRUE(llvm::all_of(ifOp.getQubits(), [](Value value) {
    return isa<QubitType>(value.getType());
  }));
  EXPECT_TRUE(llvm::all_of(ifOp.getLinearResults(), [](Value value) {
    return isa<QubitType>(value.getType());
  }));

  ASSERT_TRUE(postMiddle);
  EXPECT_EQ(cast<UnitaryOpInterface>(postMiddle.getOperation())
                .getInputQubits()
                .front(),
            ifOp.getLinearResults()[0]);

  ASSERT_EQ(insertedScalars.size(), 2);
  EXPECT_TRUE(llvm::is_contained(insertedScalars, ifOp.getLinearResults()[1]));
  EXPECT_TRUE(llvm::is_contained(insertedScalars, ifOp.getLinearResults()[2]));

  auto thenValues = ifOp.thenYield().getTargets();
  ASSERT_EQ(thenValues.size(), 3);
  EXPECT_TRUE(isa<YOp>(thenValues[0].getDefiningOp()));
  EXPECT_TRUE(isa<XOp>(thenValues[1].getDefiningOp()));
  EXPECT_TRUE(isa<ZOp>(thenValues[2].getDefiningOp()));
}

TEST_F(QTensorCanonicalizationTest, LeavesUnsupportedQTensorIfUnchanged) {
  constexpr std::array<StringLiteral, 2> mlirCodes = {
      R"mlir(
        module {
          func.func @main(%condition: i1, %index: index) {
            %c2 = arith.constant 2 : index
            %tensor0 = qtensor.alloc(%c2) : tensor<2x!qco.qubit>
            %tensor1 = qco.if %condition
                args(%arg0 = %tensor0) -> (tensor<2x!qco.qubit>) {
              %tensor2, %q0 = qtensor.extract %arg0[%index]
                  : tensor<2x!qco.qubit>
              %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
              %tensor3 = qtensor.insert %q1 into %tensor2[%index]
                  : tensor<2x!qco.qubit>
              qco.yield %tensor3 : tensor<2x!qco.qubit>
            } else args(%arg0 = %tensor0) {
              qco.yield %arg0 : tensor<2x!qco.qubit>
            }
            qtensor.dealloc %tensor1 : tensor<2x!qco.qubit>
            return
          }
        }
      )mlir",
      R"mlir(
        module {
          func.func @main(%condition: i1, %size: index) {
            %c0 = arith.constant 0 : index
            %tensor0 = qtensor.alloc(%size) : tensor<?x!qco.qubit>
            %tensor1 = qco.if %condition
                args(%arg0 = %tensor0) -> (tensor<?x!qco.qubit>) {
              %tensor2, %q0 = qtensor.extract %arg0[%c0]
                  : tensor<?x!qco.qubit>
              %q1 = qco.x %q0 : !qco.qubit -> !qco.qubit
              %tensor3 = qtensor.insert %q1 into %tensor2[%c0]
                  : tensor<?x!qco.qubit>
              qco.yield %tensor3 : tensor<?x!qco.qubit>
            } else args(%arg0 = %tensor0) {
              qco.yield %arg0 : tensor<?x!qco.qubit>
            }
            qtensor.dealloc %tensor1 : tensor<?x!qco.qubit>
            return
          }
        }
      )mlir",
  };

  for (StringRef mlirCode : mlirCodes) {
    auto moduleOp = parseSourceString<ModuleOp>(mlirCode, &context_);
    ASSERT_TRUE(moduleOp);
    ASSERT_TRUE(succeeded(verify(*moduleOp)));
    ASSERT_TRUE(succeeded(runQCOCleanupPipeline(moduleOp.get())));
    ASSERT_TRUE(succeeded(verify(*moduleOp)));

    IfOp ifOp;
    moduleOp->walk([&](IfOp candidate) { ifOp = candidate; });
    ASSERT_TRUE(ifOp);
    ASSERT_EQ(ifOp.getQubits().size(), 1);
    EXPECT_TRUE(isa<RankedTensorType>(ifOp.getQubits().front().getType()));
    size_t nestedExtracts = 0;
    size_t nestedInserts = 0;
    ifOp->walk([&](Operation* operation) {
      nestedExtracts += isa<qtensor::ExtractOp>(operation);
      nestedInserts += isa<qtensor::InsertOp>(operation);
    });
    EXPECT_EQ(nestedExtracts, 1);
    EXPECT_EQ(nestedInserts, 1);
  }
}

} // namespace
