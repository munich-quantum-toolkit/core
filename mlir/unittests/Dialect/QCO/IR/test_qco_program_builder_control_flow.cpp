/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::qco;

namespace {
class QCOControlFlowBuilderTest : public testing::Test {
protected:
  MLIRContext context;

  QCOControlFlowBuilderTest() {
    context
        .loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  }
};
} // namespace

TEST_F(QCOControlFlowBuilderTest, IfYieldsClassicalValuesBeforeLinearResults) {
  QCOProgramBuilder builder(&context);
  builder.initialize({builder.getI1Type(), builder.getI64Type()});
  auto [tensor, qubit] = builder.qtensorExtract(builder.qtensorAlloc(1), 0);
  auto results = builder.qcoIf(
      true, {tensor, qubit},
      [&](ValueRange args) {
        auto [measured, bit] = builder.measure(builder.h(args[1]));
        return SmallVector<Value>{
            bit,
            builder.intConstant(1),
            args[0],
            measured,
        };
      },
      [&](ValueRange args) {
        return SmallVector<Value>{
            builder.boolConstant(false),
            builder.intConstant(0),
            args[0],
            args[1],
        };
      });
  ASSERT_EQ(results.size(), 4);
  auto ifOp = results[0].getDefiningOp<IfOp>();
  ASSERT_TRUE(ifOp);
  EXPECT_EQ(ifOp.getClassicalResults().size(), 2);
  EXPECT_EQ(ifOp.getLinearResults().size(), 2);
  EXPECT_EQ(results[0].getType(), builder.getI1Type());
  EXPECT_EQ(results[1].getType(), builder.getI64Type());
  auto measure = ifOp.thenYield().getOperand(0).getDefiningOp<MeasureOp>();
  ASSERT_TRUE(measure);
  EXPECT_EQ(ifOp.thenYield().getOperand(3), measure.getQubitOut());
  builder.h(results[3]);
  auto moduleOp = builder.finalize(results.take_front(2));
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
  moduleOp->walk([&](qtensor::InsertOp insert) {
    EXPECT_EQ(insert.getDest(), results[2]);
  });
}

TEST_F(QCOControlFlowBuilderTest, IfYieldsOnlyClassicalValues) {
  QCOProgramBuilder builder(&context);
  builder.initialize();
  auto reg = builder.allocClassicalBitRegister(1);
  auto results = builder.qcoIf(
      reg, 0, ValueRange{},
      [&](ValueRange) { return SmallVector<Value>{builder.intConstant(1)}; },
      [&](ValueRange) { return SmallVector<Value>{builder.intConstant(0)}; });
  ASSERT_EQ(results.size(), 1);
  auto ifOp = results[0].getDefiningOp<IfOp>();
  EXPECT_TRUE(ifOp.getLinearResults().empty());
  auto moduleOp = builder.finalize(results);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
}

TEST_F(QCOControlFlowBuilderTest, IfTracksNestedClassicalAndLinearResults) {
  QCOProgramBuilder builder(&context);
  builder.initialize({builder.getI1Type()});
  Value qubit = builder.allocQubit();
  const auto branch = [&](ValueRange args) {
    return SmallVector<Value>{builder.boolConstant(true), builder.x(args[0])};
  };
  auto results = builder.qcoIf(
      true, qubit,
      [&](ValueRange args) {
        return SmallVector<Value>(builder.qcoIf(true, args, branch, branch));
      },
      branch);
  builder.h(results[1]);
  auto moduleOp = builder.finalize(results[0]);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
}

TEST_F(QCOControlFlowBuilderTest, IfRejectsClassicalResultsAfterLinearResults) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(&context);
        builder.initialize();
        builder.qcoIf(
            true, ValueRange{builder.allocQubit()}, [&](ValueRange args) {
              return SmallVector<Value>{args[0], builder.boolConstant(true)};
            });
      },
      "Classical results must precede qubit and tensor results");
}

TEST_F(QCOControlFlowBuilderTest, IfRequiresElseForClassicalResults) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(&context);
        builder.initialize();
        builder.qcoIf(
            true, ValueRange{builder.allocQubit()}, [&](ValueRange args) {
              return SmallVector<Value>{builder.boolConstant(true), args[0]};
            });
      },
      "An else body is required when returning classical results");
}

TEST_F(QCOControlFlowBuilderTest, IfRequiresMatchingBranchResults) {
  for (bool omitClassical : {false, true}) {
    SCOPED_TRACE(omitClassical);
    EXPECT_DEATH(
        {
          QCOProgramBuilder builder(&context);
          builder.initialize();
          builder.qcoIf(
              true, ValueRange{builder.allocQubit()},
              [&](ValueRange args) {
                return SmallVector<Value>{builder.intConstant(1), args[0]};
              },
              [&](ValueRange args) {
                if (omitClassical) {
                  return SmallVector<Value>{args[0]};
                }
                return SmallVector<Value>{builder.boolConstant(true), args[0]};
              });
        },
        "Then and else bodies must return the same types");
  }
}

TEST_F(QCOControlFlowBuilderTest, IfRequiresOneLinearResultPerInput) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(&context);
        builder.initialize();
        builder.qcoIf(true, ValueRange{builder.allocQubit()}, [&](ValueRange) {
          return SmallVector<Value>{builder.boolConstant(true)};
        });
      },
      "Then body must return exactly one qubit or tensor per input value");
}

TEST_F(QCOControlFlowBuilderTest, IfRejectsClassicalInputs) {
  EXPECT_DEATH(
      {
        QCOProgramBuilder builder(&context);
        builder.initialize();
        builder.qcoIf(true, builder.boolConstant(true),
                      [](ValueRange args) { return SmallVector<Value>(args); });
      },
      "Elements must be qubit values");
}

TEST_F(QCOControlFlowBuilderTest, LoopsCarryClassicalValuesInAnyPosition) {
  for (bool useWhile : {false, true}) {
    for (bool carryTensor : {false, true}) {
      SCOPED_TRACE(useWhile);
      SCOPED_TRACE(carryTensor);
      QCOProgramBuilder builder(&context);
      builder.initialize();
      SmallVector<Value> inputs{builder.intConstant(0)};
      if (carryTensor) {
        inputs.push_back(builder.qtensorAlloc(1));
      }
      inputs.push_back(builder.floatConstant(1.0));
      const auto body = [&](ValueRange args) {
        SmallVector<Value> updated(args);
        updated[0] =
            arith::AddIOp::create(builder, args[0], builder.intConstant(1));
        updated.back() = arith::MulFOp::create(builder, args.back(),
                                               builder.floatConstant(0.5));
        return updated;
      };
      auto results =
          useWhile
              ? builder.scfWhile(
                    inputs,
                    [&](ValueRange args) {
                      auto updated = body(args);
                      builder.scfCondition(builder.boolConstant(false),
                                           updated);
                      return updated;
                    },
                    body)
              : builder.scfFor(0, 2, 1, inputs, [&](Value, ValueRange args) {
                  return body(args);
                });
      EXPECT_TRUE(
          llvm::equal(results.getTypes(), ValueRange(inputs).getTypes()));
      auto moduleOp = builder.finalize(results[0]);
      EXPECT_TRUE(succeeded(verify(*moduleOp)));
      EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
    }
  }
}

TEST_F(QCOControlFlowBuilderTest, WhileCarriesVqeStateThroughNestedFor) {
  QCOProgramBuilder builder(&context);
  builder.initialize();
  auto [tensor, qubit] = builder.qtensorExtract(builder.qtensorAlloc(1), 0);
  SmallVector<Value> inputs{
      builder.boolConstant(true),
      builder.floatConstant(1.0),
      builder.intConstant(1),
      qubit,
  };
  auto results = builder.scfWhile(
      inputs,
      [&](ValueRange args) {
        builder.scfCondition(args[0], args);
        return SmallVector<Value>(args);
      },
      [&](ValueRange args) {
        auto qubits =
            builder.scfFor(0, 2, 1, args[3], [&](Value, ValueRange iterArgs) {
              return SmallVector<Value>{builder.ry(args[1], iterArgs[0])};
            });
        auto [measured, bit] = builder.measure(qubits[0]);
        Value energy =
            arith::ExtUIOp::create(builder, builder.getI64Type(), bit);
        Value improved = arith::CmpIOp::create(
            builder, arith::CmpIPredicate::slt, energy, args[2]);
        Value angle =
            arith::MulFOp::create(builder, args[1], builder.floatConstant(0.5));
        return SmallVector<Value>{improved, angle, energy, measured};
      });
  auto whileOp = results[0].getDefiningOp<scf::WhileOp>();
  EXPECT_EQ(whileOp.getConditionOp().getCondition(),
            whileOp.getBeforeArguments()[0]);
  auto fullTensor = builder.qtensorInsert(builder.h(results[3]), tensor, 0);
  builder.qtensorDealloc(fullTensor);
  auto moduleOp = builder.finalize(results[2]);
  EXPECT_TRUE(succeeded(verify(*moduleOp)));
  EXPECT_TRUE(succeeded(verifyLinearity(*moduleOp)));
}

TEST_F(QCOControlFlowBuilderTest, LoopsRejectChangedClassicalTypes) {
  for (bool useWhile : {false, true}) {
    SCOPED_TRACE(useWhile);
    EXPECT_DEATH(
        {
          QCOProgramBuilder builder(&context);
          builder.initialize();
          auto input = builder.intConstant(1);
          const auto body = [&](ValueRange) {
            return SmallVector<Value>{builder.floatConstant(1.0)};
          };
          if (useWhile) {
            builder.scfWhile(
                input,
                [&](ValueRange args) {
                  builder.scfCondition(builder.boolConstant(false), args);
                  return SmallVector<Value>(args);
                },
                body);
          } else {
            builder.scfFor(0, 1, 1, input,
                           [&](Value, ValueRange args) { return body(args); });
          }
        },
        "Result types must match input types");
  }
}
