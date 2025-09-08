/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"

#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>

#define DEBUG_TYPE "transpilation-verification"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_TRANSPILATIONVERIFICATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

using namespace mlir;

/// @brief A function attribute that specifies an (QIR) entry point function.
constexpr llvm::StringLiteral ATTRIBUTE_ENTRY_POINT{"entry_point"};

namespace {
void forward(llvm::DenseMap<Value, std::size_t>& map, const Value in,
             const Value out) {
  assert(in != out);
  map[out] = map.at(in);
  map.erase(in);
}
} // namespace

/**
 * @brief This pass verifies that the constraints of a target architecture are
 * met.
 */
struct TranspilationVerificationPass final
    : impl::TranspilationVerificationPassBase<TranspilationVerificationPass> {
  void runOnOperation() override {

    llvm::DenseMap<Value, std::size_t> map;

    auto arch = getArchitecture(ArchitectureName::MQTTest);

    auto res = getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
      // As of now, we don't route non-entry functions. Hence, skip.
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
          return WalkResult::skip();
        }

        map.clear();

        return WalkResult::advance();
      }

      // As of now, we don't support conditionals. Hence, emit an error.
      if (auto cond = dyn_cast<scf::IfOp>(op)) {
        return WalkResult(cond.emitOpError() << "is currently not supported");
      }

      // As of now, we don't support loops with qubit dependencies. Hence, emit
      // an error.
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (loop.getRegionIterArgs().empty()) {
          return WalkResult::advance();
        }
        return WalkResult(
            loop.emitOpError()
            << "is currently not supported with qubit dependencies");
      }

      if (auto qubit = dyn_cast<QubitOp>(op)) {
        if (map.size() == arch.nqubits()) {
          return WalkResult(qubit->emitOpError()
                            << "requires " << (map.size() + 1)
                            << " qubits but target architecture '"
                            << arch.name() << "' only supports "
                            << arch.nqubits() << " qubits");
        }

        map[qubit.getQubit()] = qubit.getIndex();
        return WalkResult::advance();
      }

      if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
        return WalkResult(alloc->emitOpError()
                          << "is not allowed for transpiled program");
      }

      if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
        return WalkResult(dealloc->emitOpError()
                          << "is not allowed for transpiled program");
      }

      if (auto reset = dyn_cast<ResetOp>(op)) {
        forward(map, reset.getInQubit(), reset.getOutQubit());
        return WalkResult::advance();
      }

      if (auto u = dyn_cast<UnitaryInterface>(op)) {
        const std::size_t nacts = u.getAllInQubits().size();
        if (nacts == 0) {
          return WalkResult::advance();
        }

        if (nacts > 2) {
          return WalkResult(u->emitOpError() << "acts on more than two qubits");
        }

        const Value in0 = u.getAllInQubits()[0];
        const Value out0 = u.getAllOutQubits()[0];

        if (nacts == 1) {
          forward(map, in0, out0);
          return WalkResult::advance();
        }

        const Value in1 = u.getAllInQubits()[1];
        const Value out1 = u.getAllOutQubits()[1];

        if (!arch.areAdjacent(map.at(in0), map.at(in1))) {
          return WalkResult(u->emitOpError()
                            << "(" << map[in0] << "," << map[in1] << ")"
                            << " is not executable on target architecture '"
                            << arch.name() << "'");
        }

        forward(map, in0, out0);
        forward(map, in1, out1);

        return WalkResult::advance();
      }

      if (auto measure = dyn_cast<MeasureOp>(op)) {
        forward(map, measure.getInQubit(), measure.getOutQubit());
        return WalkResult::advance();
      }

      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace mqt::ir::opt
