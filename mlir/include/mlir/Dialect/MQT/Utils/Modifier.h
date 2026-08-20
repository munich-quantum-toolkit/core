/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstddef>
#include <iterator>

namespace mlir::mqt {

/// Populate a modifier region and invoke @p emitBody.
template <typename QubitType>
inline void
buildModifierBody(OpBuilder& builder, OperationState& state,
                  const size_t numBlockArgs,
                  const function_ref<void(OpBuilder&, Block&)>& emitBody) {
  auto& block = state.regions.front()->emplaceBlock();
  auto qubitType = QubitType::get(builder.getContext());
  for (size_t i = 0; i < numBlockArgs; ++i) {
    block.addArgument(qubitType, state.location);
  }

  const OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);
  emitBody(builder, block);
}

/// Parse a modifier's block arguments and aliased qubit operands.
template <typename QubitType>
[[nodiscard]] ParseResult
parseTargetAliasing(OpAsmParser& parser, Region& region,
                    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  if (parser.parseLParen()) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> blockArgs;
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument newArg;
      OpAsmParser::UnresolvedOperand oldOperand;
      if (parser.parseArgument(newArg) || parser.parseEqual() ||
          parser.parseOperand(oldOperand)) {
        return failure();
      }
      operands.push_back(oldOperand);
      newArg.type = QubitType::get(parser.getBuilder().getContext());
      blockArgs.push_back(newArg);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) {
      return failure();
    }
  }

  return parser.parseRegion(region, blockArgs);
}

/// Print a modifier's block arguments and aliased qubit operands.
inline void printTargetAliasing(OpAsmPrinter& printer, Region& region,
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

/// Resolve a modifier block argument to the corresponding outer value.
inline Value getValueFromBlockArgument(Value qubit, ValueRange qubits) {
  if (auto blockArg = dyn_cast<BlockArgument>(qubit)) {
    assert(blockArg.getArgNumber() < qubits.size() &&
           "block argument index must be within qubits range");
    return qubits[blockArg.getArgNumber()];
  }
  return qubit;
}

/// Return the number of operations implementing @p UnitaryInterface.
template <typename UnitaryInterface>
[[nodiscard]] size_t getNumBodyUnitaries(Block& block) {
  return static_cast<size_t>(llvm::count_if(
      block, [](Operation& op) { return isa<UnitaryInterface>(op); }));
}

/// Return the indexed body unitary or report an invalid API use.
template <typename UnitaryInterface>
[[nodiscard]] UnitaryInterface getBodyUnitary(Block& block,
                                              const size_t index) {
  auto unitaries = llvm::make_filter_range(
      block, [](Operation& op) { return isa<UnitaryInterface>(op); });
  auto it = std::next(unitaries.begin(), static_cast<std::ptrdiff_t>(index));
  if (it == unitaries.end()) {
    llvm::reportFatalUsageError("Unitary index out of bounds");
  }
  return cast<UnitaryInterface>(*it);
}

/// Return the sole body unitary, or a null interface if there is not one.
template <typename UnitaryInterface>
[[nodiscard]] UnitaryInterface getSoleBodyUnitary(Block& block) {
  auto unitaries = llvm::make_filter_range(
      block, [](Operation& op) { return isa<UnitaryInterface>(op); });
  auto it = unitaries.begin();
  if (it == unitaries.end()) {
    return {};
  }
  auto unitary = cast<UnitaryInterface>(*it);
  if (++it != unitaries.end()) {
    return {};
  }
  return unitary;
}

/// Move a modifier body's support operations before @p target.
inline void hoistSupportingOpsBefore(Block& body, Operation* keep,
                                     Operation* target,
                                     RewriterBase& rewriter) {
  for (auto& bodyOp : llvm::make_early_inc_range(body)) {
    if (&bodyOp != keep && !bodyOp.hasTrait<OpTrait::IsTerminator>()) {
      rewriter.moveOpBefore(&bodyOp, target);
    }
  }
}

/// Inline a modifier body and replace the modifier with the yielded values.
inline void inlineModifierBody(Operation* operation, Block& body,
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

/// Inline @p source into the current block and return its yielded values.
inline SmallVector<Value>
inlineBodyReturningYields(Block& source, ValueRange blockArgReplacements,
                          RewriterBase& rewriter) {
  auto* destination = rewriter.getInsertionBlock();
  rewriter.inlineBlockBefore(&source, destination, destination->begin(),
                             blockArgReplacements);
  auto yielded = llvm::to_vector(destination->back().getOperands());
  rewriter.eraseOp(&destination->back());
  return yielded;
}

} // namespace mlir::mqt
