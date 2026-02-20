/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Passes/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#define DEBUG_TYPE "mapping-pass"

namespace mlir::qco {
#define GEN_PASS_DEF_HEURISTICMAPPINGPASS
#include "mlir/Passes/Passes.h.inc"

struct HeuristicMappingPass
    : impl::HeuristicMappingPassBase<HeuristicMappingPass> {
private:
  struct Architecture {};

  /**
   * @brief Describes a bidirectional program-to-hardware qubit mapping.
   */
  struct Mapping {};

  struct Circuit {
    Circuit() = default;

    void extend(Value q) { qubits_.emplace_back(q); }

    template <typename Range> void extend(Range&& range) {
      llvm::append_range(qubits_, std::forward<Range>(range));
    }

    [[nodiscard]] std::size_t size() const { return qubits_.size(); }

  private:
    SmallVector<Value, 32> qubits_;
  };

public:
  using HeuristicMappingPassBase::HeuristicMappingPassBase;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    Architecture arch;

    for (auto func : getOperation().getOps<func::FuncOp>()) {
      Region& region = func.getFunctionBody();

      // Stage 1: Apply initial program-to-hardware mapping strategy.
      Circuit circ;
      circ.extend(llvm::map_range(region.getOps<AllocOp>(),
                                  [](AllocOp op) { return op.getResult(); }));

      Mapping mapping = computeInitialMapping(circ, arch);

      // Stage 2: Recomputing starting program-to-hardware mapping by
      // repeating forwards and backwards traversals.
      const std::size_t repeats = 10;
      for (std::size_t i = 0; i < repeats; ++i) {
        // mapping = forward(mapping, /*commit= */false)
        // mapping = backward(mapping)
      }

      // Stage 3: Commit mapping and final traversal.
      commitMapping(func, mapping, rewriter);
      // forward(mapping, /*commit= */true)
    }
  }

private:
  static Mapping computeInitialMapping(Circuit& circ, Architecture& arch) {
    // TODO
  }

  static void commitMapping(func::FuncOp& func, const Mapping& mapping,
                            IRRewriter& rewriter) {
    // TODO
    // rewriter.setInsertionPointToStart(body);
    // SmallVector<AllocOp> allocations(body->getOps<AllocOp>());
    // for (auto [i, op] : llvm::enumerate(allocations)) {
    //   rewriter.setInsertionPoint(op);
    //   std::ignore = rewriter.replaceOpWithNewOp<StaticOp>(op, i);
    // }
  };
};
} // namespace mlir::qco
