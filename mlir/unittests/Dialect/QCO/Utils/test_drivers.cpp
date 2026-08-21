/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <gtest/gtest.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

namespace {
class DriversFixture : public testing::Test {
protected:
  void SetUp() override {
    DialectRegistry registry;
    registry.insert<QCODialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();

    context = std::make_unique<MLIRContext>();
    context->appendDialectRegistry(registry);
    context->loadAllAvailableDialects();
  }

  /// Construct the test program.
  [[maybe_unused]] [[nodiscard]] OwningOpRef<ModuleOp> getProgram() const {
    QCOProgramBuilder builder(context.get());
    builder.initialize(SmallVector<Type>(4, builder.getI1Type()));

    SmallVector<Value> qubits(4);
    SmallVector<Value> bits(4);

    for (size_t i = 0; i < 4; ++i) {
      qubits[i] = builder.allocQubit();
    }

    qubits[0] = builder.h(qubits[0]);

    std::tie(qubits[0], qubits[1]) = builder.cx(qubits[0], qubits[1]);
    std::tie(qubits[2], qubits[3]) = builder.cx(qubits[2], qubits[3]);

    qubits[0] = builder.z(qubits[0]);
    qubits[2] = builder.h(qubits[2]);

    std::tie(qubits[1], qubits[2]) = builder.cx(qubits[1], qubits[2]);

    std::tie(qubits[0], qubits[1]) = builder.cx(qubits[0], qubits[1]);

    qubits[1] = builder.h(qubits[1]);

    qubits = builder.scfFor(0, 3, 1, qubits, [&](Value, ValueRange args) {
      return SmallVector<Value>{args};
    });

    qubits = builder.qcoIf(
        false, qubits,
        [&](ValueRange args) { return SmallVector<Value>{args}; },
        [&](ValueRange args) { return SmallVector<Value>{args}; });

    const auto identity = [](ValueRange args) { return llvm::to_vector(args); };
    qubits = builder.qcoIndexSwitch(0, qubits, SmallVector<int64_t>{0},
                                    {identity}, identity);

    qubits = builder.barrier(qubits);

    for (size_t i = 0; i < 4; ++i) {
      std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
    }

    return builder.finalize(bits);
  }

  /// Return the wires of the test program.
  static SmallVector<WireIterator> getWires(ModuleOp op) {
    auto func = mlir::mqt::getEntryPoint(op);
    SmallVector<WireIterator> wires;
    for (AllocOp op : func.getOps<AllocOp>()) {
      wires.emplace_back(op.getResult());
    }
    return wires;
  }

  std::unique_ptr<MLIRContext> context;
};
} // namespace

TEST_F(DriversFixture, ProgramWalkVisitsAllOps) {
  auto mod = getProgram();
  auto wires = getWires(*mod);
  size_t nvisited = 0;
  walkProgramGraph<WireDirection::Forward>(
      wires, [&](const Frontier& frontier, ReleasedOps& released) {
        for (const auto& [op, indices] : frontier) {
          ++nvisited;
          released.emplace_back(op);
        }
        return WalkResult::advance();
      });
  ASSERT_EQ(nvisited, 24);
  ASSERT_TRUE(llvm::all_of(wires, [](const WireIterator& it) {
    return it == std::default_sentinel;
  }));

  for_each(wires, [&](WireIterator& it) { --it; });

  nvisited = 0;
  walkProgramGraph<WireDirection::Backward>(
      wires, [&](const Frontier& frontier, ReleasedOps& released) {
        for (const auto& [op, indices] : frontier) {
          ++nvisited;
          released.emplace_back(op);
        }
        return WalkResult::advance();
      });
  ASSERT_EQ(nvisited, 24);
  ASSERT_TRUE(llvm::all_of(wires, [](const WireIterator& it) {
    return it == std::default_sentinel;
  }));
}

TEST_F(DriversFixture, StopProgramWalkWithInterrupt) {
  auto mod = getProgram();
  auto wires = getWires(*mod);
  size_t nvisited = 0;
  walkProgramGraph<WireDirection::Forward>(
      wires, [&](const Frontier& frontier, ReleasedOps& released) {
        for (const auto& [op, indices] : frontier) {
          ++nvisited;
          if (isa<BarrierOp>(op)) {
            return WalkResult::interrupt();
          }

          released.emplace_back(op);
        }
        return WalkResult::advance();
      });
  ASSERT_EQ(nvisited, 16);
}

TEST_F(DriversFixture, ProgramWalkTooFewWires) {
  auto mod = getProgram();
  SmallVector<WireIterator> wires{getWires(*mod).front()};
  ASSERT_DEATH(walkProgramGraph<WireDirection::Forward>(
                   wires,
                   [&](const Frontier& frontier, ReleasedOps& released) {
                     for_each(frontier.keys(), [&](Operation* op) {
                       released.emplace_back(op);
                     });
                     return WalkResult::advance();
                   }),
               "more input qubits than wires");
}

TEST_F(DriversFixture, ProgramWalkVisitsLayersCorrectly) {
  auto mod = getProgram();
  auto wires = getWires(*mod);
  SmallVector<DenseSet<std::pair<size_t, size_t>>> layers;
  const auto callback = [&](const Frontier& frontier, ReleasedOps& released) {
    for (const auto& [op, indices] : frontier) {
      if (indices.size() == 1) {
        released.emplace_back(op);
      }
    }

    if (released.empty()) {
      DenseSet<std::pair<size_t, size_t>> layer;
      layer.reserve(frontier.size());

      for (const auto& [op, indices] : frontier) {
        if (!isa<BarrierOp>(op) && isa<UnitaryOpInterface>(op)) {
          layer.insert(std::minmax(indices[0], indices[1]));
        }
        released.emplace_back(op);
      }

      if (!layer.empty()) {
        layers.emplace_back(std::move(layer));
      }
    }

    return WalkResult::advance();
  };

  walkProgramGraph<WireDirection::Forward>(wires, callback);

  ASSERT_EQ(layers.size(), 3);
  ASSERT_TRUE(layers[0].contains(std::make_pair(0, 1)));
  ASSERT_TRUE(layers[0].contains(std::make_pair(2, 3)));
  ASSERT_TRUE(layers[1].contains(std::make_pair(1, 2)));
  ASSERT_TRUE(layers[2].contains(std::make_pair(0, 1)));

  for_each(wires, [&](WireIterator& it) { --it; });

  layers.clear();

  walkProgramGraph<WireDirection::Backward>(wires, callback);

  ASSERT_EQ(layers.size(), 3);
  ASSERT_TRUE(layers[0].contains(std::make_pair(0, 1)));
  ASSERT_TRUE(layers[1].contains(std::make_pair(1, 2)));
  ASSERT_TRUE(layers[2].contains(std::make_pair(0, 1)));
  ASSERT_TRUE(layers[2].contains(std::make_pair(2, 3)));
}

TEST_F(DriversFixture, ProgramWalkRetainsUnreleasedReadyOperations) {
  QCOProgramBuilder builder(context.get());
  builder.initialize(SmallVector<Type>(3, builder.getI1Type()));

  SmallVector<WireIterator> wires;
  SmallVector<Value> qubits(3);
  SmallVector<Value> bits(3);

  for (size_t i = 0; i < 3; ++i) {
    qubits[i] = builder.allocQubit();
    wires.emplace_back(qubits[i]);
  }

  qubits[0] = builder.h(qubits[0]);
  std::tie(qubits[1], qubits[2]) = builder.cx(qubits[1], qubits[2]);

  for (size_t i = 0; i < 3; ++i) {
    std::tie(qubits[i], bits[i]) = builder.measure(qubits[i]);
  }

  [[maybe_unused]] auto mod = builder.finalize(bits);

  size_t iteration = 0;
  DenseSet<Operation*> prev;
  DenseSet<Operation*> curr;
  walkProgramGraph<WireDirection::Forward>(
      wires, [&](const Frontier& frontier, ReleasedOps& released) {
        if (iteration++ == 0) {
          EXPECT_GE(frontier.size(), 2U);
          if (frontier.size() < 2) {
            return WalkResult::interrupt();
          }

          released.emplace_back(*frontier.keys().begin());
          for (Operation* op : llvm::drop_begin(frontier.keys())) {
            prev.insert(op);
          }
          return WalkResult::advance();
        }

        for (Operation* op : frontier.keys()) {
          curr.insert(op);
        }

        return WalkResult::interrupt();
      });

  EXPECT_GE(curr.size(), prev.size());
  EXPECT_TRUE(
      llvm::all_of(prev, [&](Operation* op) { return curr.contains(op); }));
}
