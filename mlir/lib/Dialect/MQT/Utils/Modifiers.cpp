/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/Modifiers.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cassert>

namespace mlir::mqt {

void printTargetAliasing(OpAsmPrinter& printer, Region& region,
                         OperandRange targetsIn) {
  printer << "(";
  if (region.empty()) {
    printer << ") ";
    printer.printRegion(region, false);
    return;
  }
  auto& entryBlock = region.front();

  for (unsigned i = 0; i < targetsIn.size(); ++i) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printOperand(entryBlock.getArgument(i));
    printer << " = ";
    printer.printOperand(targetsIn[i]);
  }
  printer << ") ";
  printer.printRegion(region, false);
}

Value getValueFromBlockArgument(Value qubit, ValueRange qubits) {
  if (auto blockArg = dyn_cast<BlockArgument>(qubit)) {
    assert(blockArg.getArgNumber() < qubits.size() &&
           "block argument index must be within qubits range");
    return qubits[blockArg.getArgNumber()];
  }
  return qubit;
}

LogicalResult hoistSupportingOpsBefore(Block& body, Operation* keep,
                                       Operation* target,
                                       RewriterBase& rewriter) {
  bool sawKeep = false;
  for (Operation& bodyOp : body) {
    if (&bodyOp == keep) {
      sawKeep = true;
    } else if (sawKeep && !bodyOp.hasTrait<OpTrait::IsTerminator>() &&
               !isPure(&bodyOp)) {
      return failure();
    }
  }
  for (auto& bodyOp : llvm::make_early_inc_range(body)) {
    if (&bodyOp != keep && !bodyOp.hasTrait<OpTrait::IsTerminator>()) {
      rewriter.moveOpBefore(&bodyOp, target);
    }
  }
  return success();
}

void inlineModifierBody(Operation* operation, Block& body,
                        ValueRange blockArgReplacements,
                        RewriterBase& rewriter) {
  auto* terminator = body.getTerminator();
  const auto results =
      llvm::map_to_vector(terminator->getOperands(), [&](Value yielded) {
        return getValueFromBlockArgument(yielded, blockArgReplacements);
      });
  rewriter.inlineBlockBefore(&body, operation, blockArgReplacements);
  rewriter.eraseOp(terminator);
  rewriter.replaceOp(operation, results);
}

SmallVector<Value> inlineBodyReturningYields(Block& source,
                                             ValueRange blockArgReplacements,
                                             RewriterBase& rewriter) {
  auto* destination = rewriter.getInsertionBlock();
  rewriter.inlineBlockBefore(&source, destination, destination->begin(),
                             blockArgReplacements);
  auto yielded = llvm::to_vector(destination->back().getOperands());
  rewriter.eraseOp(&destination->back());
  return yielded;
}

} // namespace mlir::mqt
