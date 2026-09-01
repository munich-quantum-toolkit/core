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
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

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
void printTargetAliasing(OpAsmPrinter& printer, Region& region,
                         OperandRange targetsIn);

/// Resolve a modifier block argument to the corresponding outer value.
[[nodiscard]] Value getValueFromBlockArgument(Value qubit, ValueRange qubits);

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

template <typename UnitaryInterface>
[[nodiscard]] bool containsUnitaryOperation(Operation* operation) {
  bool found = false;
  operation->walk(
      [&](Operation* nested) { found |= isa<UnitaryInterface>(nested); });
  return found;
}

/// Return whether @p operation or one of its nested operations is neither a
/// unitary nor a terminator.
template <typename UnitaryInterface>
[[nodiscard]] bool containsSupportingOperation(Operation* operation) {
  bool found = false;
  operation->walk([&](Operation* nested) {
    found |= !isa<UnitaryInterface>(nested) &&
             !nested->hasTrait<OpTrait::IsTerminator>();
  });
  return found;
}

/// Return whether top-level supporting operations may move before a modifier
/// without moving an impure operation across a unitary.
template <typename UnitaryInterface>
[[nodiscard]] bool canHoistSupportingOps(Block& body) {
  bool sawUnitary = false;
  for (Operation& operation : body) {
    if (isa<UnitaryInterface>(operation)) {
      if (containsSupportingOperation<UnitaryInterface>(&operation)) {
        return false;
      }
      sawUnitary = true;
      continue;
    }
    if (operation.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    if (containsUnitaryOperation<UnitaryInterface>(&operation) ||
        (sawUnitary && !isPure(&operation))) {
      return false;
    }
  }
  return true;
}

/// Move top-level non-unitary operations before @p target. Fail without
/// mutation if doing so could move an impure operation across a unitary.
template <typename UnitaryInterface>
[[nodiscard]] LogicalResult hoistSupportingOpsBefore(Block& body,
                                                     Operation* target,
                                                     RewriterBase& rewriter) {
  const auto isSupporting = [](Operation& operation) {
    return !isa<UnitaryInterface>(operation) &&
           !operation.hasTrait<OpTrait::IsTerminator>();
  };
  if (!canHoistSupportingOps<UnitaryInterface>(body)) {
    return failure();
  }
  for (Operation& operation : llvm::make_early_inc_range(body)) {
    if (isSupporting(operation)) {
      rewriter.moveOpBefore(&operation, target);
    }
  }
  return success();
}

/// Return the sole top-level body unitary, or a null interface if another
/// top-level operation contains nested unitaries.
template <typename UnitaryInterface>
[[nodiscard]] UnitaryInterface getSoleBodyUnitary(Block& block) {
  UnitaryInterface unitary;
  for (Operation& operation : block) {
    if (auto candidate = dyn_cast<UnitaryInterface>(operation)) {
      if (unitary) {
        return {};
      }
      unitary = candidate;
      continue;
    }
    if (containsUnitaryOperation<UnitaryInterface>(&operation)) {
      return {};
    }
  }
  return unitary;
}

/// Move a modifier body's support operations before @p target. Fail without
/// mutation if an impure support operation follows @p keep.
[[nodiscard]] LogicalResult hoistSupportingOpsBefore(Block& body,
                                                     Operation* keep,
                                                     Operation* target,
                                                     RewriterBase& rewriter);

/// Inline a modifier body and replace the modifier with the yielded values.
void inlineModifierBody(Operation* operation, Block& body,
                        ValueRange blockArgReplacements,
                        RewriterBase& rewriter);

/// Inline @p source into the current block and return its yielded values.
[[nodiscard]] SmallVector<Value>
inlineBodyReturningYields(Block& source, ValueRange blockArgReplacements,
                          RewriterBase& rewriter);

} // namespace mlir::mqt
