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

#include <cassert>
#include <cstddef>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#define DEBUG_TYPE "routing-verification"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_ROUTINGVERIFICATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

using namespace mlir;

/**
 * @brief Maps SSA values to static qubit indices.
 */
using QubitIndexMap = llvm::DenseMap<Value, std::size_t>;

/**
 * @brief A function attribute that specifies an (QIR) entry point function.
 */
constexpr llvm::StringLiteral ATTRIBUTE_ENTRY_POINT{"entry_point"};

namespace {

bool forwardOne(llvm::DenseMap<Value, std::size_t>& map, const Value in,
                const Value out) {
  assert(in != out);
  if (!map.contains(in)) {
    return false;
  }
  map[out] = map[in];
  map.erase(in);
  return true;
}

bool forwardRange(llvm::DenseMap<Value, std::size_t>& map,
                  const ArrayRef<Value>& in, const ArrayRef<Value>& out) {
  assert(in.size() == out.size());
  for (std::size_t i = 0; i < in.size(); ++i) {
    if (!forwardOne(map, in[i], out[i])) {
      return false;
    }
  }
  return true;
}
} // namespace

/**
 * @brief This pass verifies that the constraints of a target architecture are
 * met.
 */
struct RoutingVerificationPass final
    : impl::RoutingVerificationPassBase<RoutingVerificationPass> {
  void runOnOperation() override {

    QubitIndexMap map;

    auto arch = getArchitecture(ArchitectureName::MQTTest);

    auto res = getOperation()->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        // As of now, we don't route non-entry functions. Hence, skip.
        if (!func->hasAttr(ATTRIBUTE_ENTRY_POINT)) {
          return WalkResult::skip();
        }

        map.clear();

        return WalkResult::advance();
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
        if (!forwardOne(map, reset.getInQubit(), reset.getOutQubit())) {
          return WalkResult(reset->emitOpError()
                            << "accesses invalid qubit-index map.");
        }
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
          if (!forwardOne(map, in0, out0)) {
            return WalkResult(u->emitOpError()
                              << "accesses invalid qubit-index map.");
          }
          return WalkResult::advance();
        }

        const Value in1 = u.getAllInQubits()[1];

        if (!arch.areAdjacent(map.at(in0), map.at(in1))) {
          return WalkResult(u->emitOpError()
                            << "(" << map[in0] << "," << map[in1] << ")"
                            << " is not executable on target architecture '"
                            << arch.name() << "'");
        }

        if (!forwardRange(map, u.getAllInQubits(), u.getAllOutQubits())) {
          return WalkResult(u->emitOpError()
                            << "accesses invalid qubit-index map.");
        }

        return WalkResult::advance();
      }

      if (auto measure = dyn_cast<MeasureOp>(op)) {
        if (!forwardOne(map, measure.getInQubit(), measure.getOutQubit())) {
          return WalkResult(measure->emitOpError()
                            << "accesses invalid qubit-index map.");
        }
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
