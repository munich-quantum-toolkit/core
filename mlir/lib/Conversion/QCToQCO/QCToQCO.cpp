/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/QCToQCO/QCToQCO.h"

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>

namespace mlir {
using namespace qco;
using namespace qc;

#define GEN_PASS_DEF_QCTOQCO
#include "mlir/Conversion/QCToQCO/QCToQCO.h.inc"

namespace {

/**
 * @brief State object for tracking qubit value flow during conversion
 *
 * @details
 * This struct maintains the mapping between QC dialect qubits (which use
 * reference semantics) and their corresponding QCO dialect qubit values
 * (which use value semantics). As the conversion progresses, each QC
 * qubit reference is mapped to its latest QCO SSA value.
 *
 * The key insight is that QC operations modify qubits in-place:
 * ```mlir
 * %q = qc.alloc : !qc.qubit
 * qc.h %q : !qc.qubit        // modifies %q in-place
 * qc.x %q : !qc.qubit        // modifies %q in-place
 * ```
 *
 * While QCO operations consume inputs and produce new outputs:
 * ```mlir
 * %q0 = qco.alloc : !qco.qubit
 * %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit   // %q0 consumed, %q1 produced
 * %q2 = qco.x %q1 : !qco.qubit -> !qco.qubit   // %q1 consumed, %q2 produced
 * ```
 *
 * The qubitMap tracks that the QC qubit %q corresponds to:
 * - %q0 after allocation
 * - %q1 after the H gate
 * - %q2 after the X gate
 */
struct LoweringState {
  /// Map from original QC qubit references to their latest QCO SSA values
  /// for each region
  llvm::DenseMap<Region*, llvm::DenseMap<Value, Value>> qubitMap;
  /// Map each operation to its Set of QC qubit references
  llvm::DenseMap<Operation*, llvm::SetVector<Value>> regionMap;

  /// Modifier information
  int64_t inCtrlOp = 0;
  DenseMap<int64_t, SmallVector<Value>> targetsIn;
  DenseMap<int64_t, SmallVector<Value>> targetsOut;
};

/**
 * @brief Base class for conversion patterns that need access to lowering state
 *
 * @details
 * Extends OpConversionPattern to provide access to a shared LoweringState
 * object, which tracks the mapping from reference-semantics QC qubits
 * to value-semantics QCO qubits across multiple pattern applications.
 *
 * This stateful approach is necessary because the conversion needs to:
 * 1. Track which QCO value corresponds to each QC qubit reference
 * 2. Update these mappings as operations transform qubits
 * 3. Share this information across different conversion patterns
 *
 * @tparam OpType The QC operation type to convert
 */
template <typename OpType>
class StatefulOpConversionPattern : public OpConversionPattern<OpType> {

public:
  StatefulOpConversionPattern(TypeConverter& typeConverter,
                              MLIRContext* context, LoweringState* state)
      : OpConversionPattern<OpType>(typeConverter, context), state_(state) {}

  /// Returns the shared lowering state object
  [[nodiscard]] LoweringState& getState() const { return *state_; }

private:
  LoweringState* state_;
};

} // namespace

static bool isQubitType(Type type) {
  if (!llvm::isa<qc::QubitType>(type)) {
    auto memrefType = dyn_cast<MemRefType>(type);
    if (memrefType) {
      return llvm::isa<qc::QubitType>(memrefType.getElementType());
    }
    return false;
  }
  return true;
}

/**
 * @brief Recursively collects all the QC qubit references used by an
 * operation and store them in map
 *
 * @param Operation The operation that is currently traversed
 * @param state The lowering state
 * @param ctx The MLIRContext of the current program
 * @return llvm::Setvector<Value> The set of unique QC qubit references
 */
static llvm::SetVector<Value>
collectUniqueQubits(Operation* op, LoweringState* state, MLIRContext* ctx) {
  // Get the regions of the current operation
  const auto& regions = op->getRegions();
  SetVector<Value> uniqueQubits;
  for (auto& region : regions) {
    // Skip empty regions e.g. empty else region of an If operation
    if (region.empty()) {
      continue;
    }
    // Check that the region has only one block
    assert(region.hasOneBlock() && "Expected single-block region");

    // Collect qubits from the blockarguments
    for (auto arg : region.front().getArguments()) {
      if (isQubitType(arg.getType())) {
        uniqueQubits.insert(arg);
      }
    }

    // iterate over all operations inside the region
    // currently assumes that each region only has one block
    for (auto& operation : region.front().getOperations()) {
      // check if the operation has an region, if yes recursively collect the
      // qubits
      if (operation.getNumRegions() > 0) {
        auto qubits = collectUniqueQubits(&operation, state, ctx);
        // Remove the memref registers and exclude qubits from load operations
        // in the same region in the set of unique qubits
        if (!llvm::isa<func::FuncOp>(operation)) {
          qubits.remove_if([&](Value qubit) {
            return llvm::isa<MemRefType>(qubit.getType()) ||
                   (llvm::isa<memref::LoadOp>(qubit.getDefiningOp()) &&
                    &region == qubit.getParentRegion());
          });
        }
        uniqueQubits.set_union(qubits);
      }
      // Ignore the alloc operations inside scf.for operations
      if (llvm::isa<memref::AllocOp>(operation)) {
        if (llvm::isa<scf::ForOp>(operation.getParentOp())) {
          continue;
        }
      }
      // Only add the memref to the register when the load operation is matched
      if (llvm::isa<memref::LoadOp>(operation)) {
        auto loadOp = dyn_cast<memref::LoadOp>(operation);
        uniqueQubits.insert(loadOp.getMemRef());
        continue;
      }
      // Collect qubits form the operands
      for (const auto& operand : operation.getOperands()) {
        // Ignore the values from memref store and alloc operations
        if ((operand.getDefiningOp<memref::StoreOp>() ||
             operand.getDefiningOp<memref::AllocOp>())) {
          continue;
        }
        // Ignore the qubits that stems from load operations
        if (operand.getDefiningOp<memref::LoadOp>() &&
            llvm::isa<scf::ForOp>(op)) {
          continue;
        }
        if (isQubitType(operand.getType())) {
          uniqueQubits.insert(operand);
        }
      }
      // Collect qubits from the results
      for (const auto& result : operation.getResults()) {
        if (isQubitType(result.getType())) {
          uniqueQubits.insert(result);
        }
      }
      // Mark scf terminator operations if they need to return a value after the
      // conversion
      if ((llvm::isa<scf::YieldOp>(operation) ||
           llvm::isa<scf::ConditionOp>(operation)) &&
          !uniqueQubits.empty()) {
        operation.setAttr("needChange", StringAttr::get(ctx, "yes"));
      }
      // Mark func.return operation for functions that need to return a qubit
      // value
      if (llvm::isa<func::ReturnOp>(operation)) {
        if (auto func = operation.getParentOfType<func::FuncOp>()) {
          if (!func.getArgumentTypes().empty() &&
              isQubitType(func.getArgumentTypes().front())) {
            operation.setAttr("needChange", StringAttr::get(ctx, "yes"));
            // Only add the arguments as qubits for the regionMap of func
            llvm::SetVector<Value> argQubits;
            for (auto arg : func.getArguments()) {
              if (isQubitType(arg.getType())) {
                argQubits.insert(arg);
              }
            }
            state->regionMap[func] = argQubits;
          }
        }
      }
    }
  }
  // Add the operands from the operation itself
  for (const auto& operand : op->getOperands()) {
    if (isQubitType(operand.getType())) {
      uniqueQubits.insert(operand);
    }
  }
  // Mark scf operations that need to be changed afterwards
  if (!uniqueQubits.empty() &&
      (llvm::isa<scf::IfOp>(op) || (llvm::isa<scf::ForOp>(op)) ||
       llvm::isa<scf::WhileOp>(op))) {
    state->regionMap[op] = uniqueQubits;
    op->setAttr("needChange", StringAttr::get(ctx, "yes"));
  }
  return uniqueQubits;
}

/**
 * @brief Converts a zero-target, one-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertZeroTargetOneParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                              LoweringState& state) {
  const auto inCtrlOp = state.inCtrlOp;

  rewriter.create<QCOOpType>(op.getLoc(), op.getParameter(0));

  // Update the state
  if (inCtrlOp != 0) {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut;
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, zero-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertOneTargetZeroParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                              LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubit
  const auto& qcQubit = op.getOperand();
  Value qcoQubit;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    qcoQubit = qubitMap[qcQubit];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 1 &&
           "Invalid number of input targets");
    qcoQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), qcoQubit);

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit] = qcoOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, one-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertOneTargetOneParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                             LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubit
  const auto& qcQubit = op.getOperand(0);
  Value qcoQubit;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    qcoQubit = qubitMap[qcQubit];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 1 &&
           "Invalid number of input targets");
    qcoQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp =
      rewriter.create<QCOOpType>(op.getLoc(), qcoQubit, op.getParameter(0));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit] = qcoOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, two-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertOneTargetTwoParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                             LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubit
  const auto& qcQubit = op.getOperand(0);
  Value qcoQubit;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    qcoQubit = qubitMap[qcQubit];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 1 &&
           "Invalid number of input targets");
    qcoQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp = rewriter.create<QCOOpType>(
      op.getLoc(), qcoQubit, op.getParameter(0), op.getParameter(1));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit] = qcoOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a one-target, three-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult convertOneTargetThreeParameter(
    QCOpType& op, ConversionPatternRewriter& rewriter, LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubit
  const auto& qcQubit = op.getOperand(0);
  Value qcoQubit;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    qcoQubit = qubitMap[qcQubit];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 1 &&
           "Invalid number of input targets");
    qcoQubit = state.targetsIn[inCtrlOp].front();
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp =
      rewriter.create<QCOOpType>(op.getLoc(), qcoQubit, op.getParameter(0),
                                 op.getParameter(1), op.getParameter(2));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit] = qcoOp.getQubitOut();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut({qcoOp.getQubitOut()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a two-target, zero-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertTwoTargetZeroParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                              LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubits
  const auto& qcQubit0 = op.getOperand(0);
  const auto& qcQubit1 = op.getOperand(1);
  Value qcoQubit0;
  Value qcoQubit1;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit0) && "QC qubit not found");
    assert(qubitMap.contains(qcQubit1) && "QC qubit not found");
    qcoQubit0 = qubitMap[qcQubit0];
    qcoQubit1 = qubitMap[qcQubit1];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 2 &&
           "Invalid number of input targets");
    const auto& targetsIn = state.targetsIn[inCtrlOp];
    qcoQubit0 = targetsIn[0];
    qcoQubit1 = targetsIn[1];
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), qcoQubit0, qcoQubit1);

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit0] = qcoOp.getQubit0Out();
    qubitMap[qcQubit1] = qcoOp.getQubit1Out();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut(
        {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a two-target, one-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertTwoTargetOneParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                             LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubits
  const auto& qcQubit0 = op.getOperand(0);
  const auto& qcQubit1 = op.getOperand(1);
  Value qcoQubit0;
  Value qcoQubit1;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit0) && "QC qubit not found");
    assert(qubitMap.contains(qcQubit1) && "QC qubit not found");
    qcoQubit0 = qubitMap[qcQubit0];
    qcoQubit1 = qubitMap[qcQubit1];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 2 &&
           "Invalid number of input targets");
    const auto& targetsIn = state.targetsIn[inCtrlOp];
    qcoQubit0 = targetsIn[0];
    qcoQubit1 = targetsIn[1];
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp = rewriter.create<QCOOpType>(op.getLoc(), qcoQubit0, qcoQubit1,
                                          op.getParameter(0));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit0] = qcoOp.getQubit0Out();
    qubitMap[qcQubit1] = qcoOp.getQubit1Out();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut(
        {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Converts a two-target, two-parameter QC operation to QCO
 *
 * @tparam QCOOpType The operation type of the QCO operation
 * @tparam QCOpType The operation type of the QC operation
 * @param op The QC operation instance to convert
 * @param rewriter The pattern rewriter
 * @param state The lowering state
 * @return LogicalResult Success or failure of the conversion
 */
template <typename QCOOpType, typename QCOpType>
static LogicalResult
convertTwoTargetTwoParameter(QCOpType& op, ConversionPatternRewriter& rewriter,
                             LoweringState& state) {
  auto& qubitMap = state.qubitMap[op->getParentRegion()];
  const auto inCtrlOp = state.inCtrlOp;

  // Get the latest QCO qubits
  const auto& qcQubit0 = op.getOperand(0);
  const auto& qcQubit1 = op.getOperand(1);
  Value qcoQubit0;
  Value qcoQubit1;
  if (inCtrlOp == 0) {
    assert(qubitMap.contains(qcQubit0) && "QC qubit not found");
    assert(qubitMap.contains(qcQubit1) && "QC qubit not found");
    qcoQubit0 = qubitMap[qcQubit0];
    qcoQubit1 = qubitMap[qcQubit1];
  } else {
    assert(state.targetsIn[inCtrlOp].size() == 2 &&
           "Invalid number of input targets");
    const auto& targetsIn = state.targetsIn[inCtrlOp];
    qcoQubit0 = targetsIn[0];
    qcoQubit1 = targetsIn[1];
  }

  // Create the QCO operation (consumes input, produces output)
  auto qcoOp =
      rewriter.create<QCOOpType>(op.getLoc(), qcoQubit0, qcoQubit1,
                                 op.getParameter(0), op.getParameter(1));

  // Update the state map
  if (inCtrlOp == 0) {
    qubitMap[qcQubit0] = qcoOp.getQubit0Out();
    qubitMap[qcQubit1] = qcoOp.getQubit1Out();
  } else {
    state.targetsIn.erase(inCtrlOp);
    const SmallVector<Value> targetsOut(
        {qcoOp.getQubit0Out(), qcoOp.getQubit1Out()});
    state.targetsOut.try_emplace(inCtrlOp, targetsOut);
  }

  rewriter.eraseOp(op);

  return success();
}

/**
 * @brief Type converter for QC-to-QCO conversion
 *
 * @details
 * Handles type conversion between the QC and QCO dialects.
 * The primary conversion is from !qc.qubit to !qco.qubit, which
 * represents the semantic shift from reference types to value types.
 *
 * Other types (integers, booleans, etc.) pass through unchanged via
 * the identity conversion.
 */
class QCToQCOTypeConverter final : public TypeConverter {
public:
  explicit QCToQCOTypeConverter(MLIRContext* ctx) {
    // Identity conversion for all types by default
    addConversion([](Type type) { return type; });

    // Convert QC qubit references to QCO qubit values
    addConversion([ctx](qc::QubitType /*type*/) -> Type {
      return qco::QubitType::get(ctx);
    });
  }
};

/**
 * @brief Converts qc.alloc to qco.alloc
 *
 * @details
 * Allocates a new qubit and establishes the initial mapping in the state.
 * Both dialects initialize qubits to the |0⟩ state.
 *
 * Register metadata (name, size, index) is preserved during conversion,
 * allowing the QCO representation to maintain register information for
 * debugging and visualization.
 *
 * Example transformation:
 * ```mlir
 * %q = qc.alloc("q", 3, 0) : !qc.qubit
 * // becomes:
 * %q0 = qco.alloc("q", 3, 0) : !qco.qubit
 * ```
 */
struct ConvertQCAllocOp final : StatefulOpConversionPattern<qc::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& qcQubit = op.getResult();

    // Create the qco.alloc operation with preserved register metadata
    auto qcoOp = rewriter.replaceOpWithNewOp<qco::AllocOp>(
        op, op.getRegisterNameAttr(), op.getRegisterSizeAttr(),
        op.getRegisterIndexAttr());

    const auto& qcoQubit = qcoOp.getResult();

    // Establish initial mapping: this QC qubit reference now corresponds
    // to this QCO SSA value
    qubitMap.try_emplace(qcQubit, qcoQubit);

    return success();
  }
};

/**
 * @brief Converts qc.dealloc to qco.dealloc
 *
 * @details
 * Deallocates a qubit by looking up its latest QCO value and creating
 * a corresponding qco.dealloc operation. The mapping is removed from
 * the state as the qubit is no longer in use.
 *
 * Example transformation:
 * ```mlir
 * qc.dealloc %q : !qc.qubit
 * // becomes (where %q maps to %q_final):
 * qco.dealloc %q_final : !qco.qubit
 * ```
 */
struct ConvertQCDeallocOp final : StatefulOpConversionPattern<qc::DeallocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::DeallocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& qcQubit = op.getQubit();

    // Look up the latest QCO value for this QC qubit
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    const auto& qcoQubit = qubitMap[qcQubit];

    // Create the dealloc operation
    rewriter.replaceOpWithNewOp<qco::DeallocOp>(op, qcoQubit);

    // Remove from state as qubit is no longer in use
    qubitMap.erase(qcQubit);

    return success();
  }
};

/**
 * @brief Converts qc.static to qco.static
 *
 * @details
 * Static qubits represent references to hardware-mapped or fixed-position
 * qubits identified by an index. This conversion creates the corresponding
 * qco.static operation and establishes the mapping.
 *
 * Example transformation:
 * ```mlir
 * %q = qc.static 0 : !qc.qubit
 * // becomes:
 * %q0 = qco.static 0 : !qco.qubit
 * ```
 */
struct ConvertQCStaticOp final : StatefulOpConversionPattern<qc::StaticOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::StaticOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& qcQubit = op.getQubit();

    // Create new qco.static operation with the same index
    auto qcoOp = rewriter.create<qco::StaticOp>(op.getLoc(), op.getIndex());

    // Collect QCO qubit SSA value
    const auto& qcoQubit = qcoOp.getQubit();

    // Establish mapping from QC reference to QCO value
    qubitMap[qcQubit] = qcoQubit;

    // Replace the old operation result with the new result
    rewriter.replaceOp(op, qcoQubit);

    return success();
  }
};

/**
 * @brief Converts qc.measure to qco.measure
 *
 * @details
 * Measurement is a key operation where the semantic difference is visible:
 * - QC: Measures in-place, returning only the classical bit
 * - QCO: Consumes input qubit, returns both output qubit and classical bit
 *
 * The conversion looks up the latest QCO value for the QC qubit,
 * performs the measurement, updates the mapping with the output qubit,
 * and returns the classical bit result.
 *
 * Register metadata (name, size, index) for output recording is preserved
 * during conversion.
 *
 * Example transformation:
 * ```mlir
 * %c = qc.measure("c", 2, 0) %q : !qc.qubit -> i1
 * // becomes (where %q maps to %q_in):
 * %q_out, %c = qco.measure("c", 2, 0) %q_in : !qco.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQCMeasureOp final : StatefulOpConversionPattern<qc::MeasureOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::MeasureOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& qcQubit = op.getQubit();

    // Get the latest QCO qubit value from the state map
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    const auto& qcoQubit = qubitMap[qcQubit];

    // Create qco.measure (returns both output qubit and bit result)
    auto qcoOp = rewriter.create<qco::MeasureOp>(
        op.getLoc(), qcoQubit, op.getRegisterNameAttr(),
        op.getRegisterSizeAttr(), op.getRegisterIndexAttr());

    const auto& outQCOQubit = qcoOp.getQubitOut();
    const auto& newBit = qcoOp.getResult();

    // Update mapping: the QC qubit now corresponds to the output qubit
    qubitMap[qcQubit] = outQCOQubit;

    // Replace the QC operation's bit result with the QCO bit result
    rewriter.replaceOp(op, newBit);

    return success();
  }
};

/**
 * @brief Converts qc.reset to qco.reset
 *
 * @details
 * Reset operations force a qubit to the |0⟩ state. The semantic difference:
 * - QC: Resets in-place (no result value)
 * - QCO: Consumes input qubit, returns reset output qubit
 *
 * The conversion looks up the latest QCO value, performs the reset,
 * and updates the mapping with the output qubit. The QC operation
 * is erased as it has no results to replace.
 *
 * Example transformation:
 * ```mlir
 * qc.reset %q : !qc.qubit
 * // becomes (where %q maps to %q_in):
 * %q_out = qco.reset %q_in : !qco.qubit -> !qco.qubit
 * // state updated: %q now maps to %q_out
 * ```
 */
struct ConvertQCResetOp final : StatefulOpConversionPattern<qc::ResetOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::ResetOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto& qcQubit = op.getQubit();

    // Get the latest QCO qubit value from the state map
    assert(qubitMap.contains(qcQubit) && "QC qubit not found");
    const auto& qcoQubit = qubitMap[qcQubit];

    // Create qco.reset (consumes input, produces output)
    auto qcoOp = rewriter.create<qco::ResetOp>(op.getLoc(), qcoQubit);

    // Update mapping: the QC qubit now corresponds to the reset output
    qubitMap[qcQubit] = qcoOp.getQubitOut();

    // Erase the old (it has no results to replace)
    rewriter.eraseOp(op);

    return success();
  }
};

// ZeroTargetOneParameter

#define DEFINE_ZERO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)             \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM)                                                        \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * qco.OP_NAME(%PARAM)                                                       \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertZeroTargetOneParameter<qco::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_ZERO_TARGET_ONE_PARAMETER(GPhaseOp, gphase, theta)

#undef DEFINE_ZERO_TARGET_ONE_PARAMETER

// OneTargetZeroParameter

#define DEFINE_ONE_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME %q : !qc.qubit                                                 \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME %q_in : !qco.qubit -> !qco.qubit                     \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetZeroParameter<qco::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ZERO_PARAMETER(IdOp, id)
DEFINE_ONE_TARGET_ZERO_PARAMETER(XOp, x)
DEFINE_ONE_TARGET_ZERO_PARAMETER(YOp, y)
DEFINE_ONE_TARGET_ZERO_PARAMETER(ZOp, z)
DEFINE_ONE_TARGET_ZERO_PARAMETER(HOp, h)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SOp, s)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SdgOp, sdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TOp, t)
DEFINE_ONE_TARGET_ZERO_PARAMETER(TdgOp, tdg)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXOp, sx)
DEFINE_ONE_TARGET_ZERO_PARAMETER(SXdgOp, sxdg)

#undef DEFINE_ONE_TARGET_ZERO_PARAMETER

// OneTargetOneParameter

#define DEFINE_ONE_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM) %q : !qc.qubit                                         \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM) %q_in : !qco.qubit -> !qco.qubit             \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetOneParameter<qco::OP_CLASS>(op, rewriter,         \
                                                         getState());          \
    }                                                                          \
  };

DEFINE_ONE_TARGET_ONE_PARAMETER(RXOp, rx, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RYOp, ry, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(RZOp, rz, theta)
DEFINE_ONE_TARGET_ONE_PARAMETER(POp, p, theta)

#undef DEFINE_ONE_TARGET_ONE_PARAMETER

// OneTargetTwoParameter

#define DEFINE_ONE_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM1, %PARAM2) %q : !qc.qubit                               \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM1, %PARAM2) %q_in : !qco.qubit ->              \
   * !qco.qubit                                                                \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetTwoParameter<qco::OP_CLASS>(op, rewriter,         \
                                                         getState());          \
    }                                                                          \
  };

DEFINE_ONE_TARGET_TWO_PARAMETER(ROp, r, theta, phi)
DEFINE_ONE_TARGET_TWO_PARAMETER(U2Op, u2, phi, lambda)

#undef DEFINE_ONE_TARGET_TWO_PARAMETER

// OneTargetThreeParameter

#define DEFINE_ONE_TARGET_THREE_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2,   \
                                          PARAM3)                              \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q : !qc.qubit                      \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q_out = qco.OP_NAME(%PARAM1, %PARAM2, %PARAM3) %q_in : !qco.qubit        \
   * -> !qco.qubit                                                             \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertOneTargetThreeParameter<qco::OP_CLASS>(op, rewriter,       \
                                                           getState());        \
    }                                                                          \
  };

DEFINE_ONE_TARGET_THREE_PARAMETER(UOp, u, theta, phi, lambda)

#undef DEFINE_ONE_TARGET_THREE_PARAMETER

// TwoTargetZeroParameter

#define DEFINE_TWO_TARGET_ZERO_PARAMETER(OP_CLASS, OP_NAME)                    \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME %q0, %q1 : !qc.qubit, !qc.qubit                                \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME %q0_in, %q1_in : !qco.qubit, !qco.qubit    \
   * -> !qco.qubit, !qco.qubit                                                 \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetZeroParameter<qco::OP_CLASS>(op, rewriter,        \
                                                          getState());         \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ZERO_PARAMETER(SWAPOp, swap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(iSWAPOp, iswap)
DEFINE_TWO_TARGET_ZERO_PARAMETER(DCXOp, dcx)
DEFINE_TWO_TARGET_ZERO_PARAMETER(ECROp, ecr)

#undef DEFINE_TWO_TARGET_ZERO_PARAMETER

// TwoTargetOneParameter

#define DEFINE_TWO_TARGET_ONE_PARAMETER(OP_CLASS, OP_NAME, PARAM)              \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM) %q0, %q1 : !qc.qubit, !qc.qubit                        \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME(%PARAM) %q0_in, %q1_in : !qco.qubit,       \
   * !qco.qubit -> !qco.qubit, !qco.qubit                                      \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetOneParameter<qco::OP_CLASS>(op, rewriter,         \
                                                         getState());          \
    }                                                                          \
  };

DEFINE_TWO_TARGET_ONE_PARAMETER(RXXOp, rxx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RYYOp, ryy, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZXOp, rzx, theta)
DEFINE_TWO_TARGET_ONE_PARAMETER(RZZOp, rzz, theta)

#undef DEFINE_TWO_TARGET_ONE_PARAMETER

// TwoTargetTwoParameter

#define DEFINE_TWO_TARGET_TWO_PARAMETER(OP_CLASS, OP_NAME, PARAM1, PARAM2)     \
  /**                                                                          \
   * @brief Converts qc.OP_NAME to qco.OP_NAME                                 \
   *                                                                           \
   * @par Example:                                                             \
   * ```mlir                                                                   \
   * qc.OP_NAME(%PARAM1, %PARAM2) %q0, %q1 : !qc.qubit, !qc.qubit              \
   * ```                                                                       \
   * is converted to                                                           \
   * ```mlir                                                                   \
   * %q0_out, %q1_out = qco.OP_NAME(%PARAM1, %PARAM2) %q0_in, %q1_in :         \
   * !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit                          \
   * ```                                                                       \
   */                                                                          \
  struct ConvertQC##OP_CLASS final                                             \
      : StatefulOpConversionPattern<qc::OP_CLASS> {                            \
    using StatefulOpConversionPattern::StatefulOpConversionPattern;            \
                                                                               \
    LogicalResult                                                              \
    matchAndRewrite(qc::OP_CLASS op, OpAdaptor /*adaptor*/,                    \
                    ConversionPatternRewriter& rewriter) const override {      \
      return convertTwoTargetTwoParameter<qco::OP_CLASS>(op, rewriter,         \
                                                         getState());          \
    }                                                                          \
  };

DEFINE_TWO_TARGET_TWO_PARAMETER(XXPlusYYOp, xx_plus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER(XXMinusYYOp, xx_minus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER

// BarrierOp

/**
 * @brief Converts qc.barrier to qco.barrier
 *
 * @par Example:
 * ```mlir
 * qc.barrier %q0, %q1 : !qc.qubit, !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %q0_out, %q1_out = qco.barrier %q0_in, %q1_in : !qco.qubit, !qco.qubit ->
 * !qco.qubit, !qco.qubit
 * ```
 */
struct ConvertQCBarrierOp final : StatefulOpConversionPattern<qc::BarrierOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::BarrierOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& qubitMap = state.qubitMap[op->getParentRegion()];

    // Get QCO qubits from state map
    const auto& qcQubits = op.getQubits();
    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(qcQubits.size());
    for (const auto& qcQubit : qcQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap[qcQubit]);
    }

    // Create qco.barrier
    auto qcoOp = rewriter.create<qco::BarrierOp>(op.getLoc(), qcoQubits);

    // Update the state map
    for (const auto& [qcQubit, qcoQubitOut] :
         llvm::zip(qcQubits, qcoOp.getQubitsOut())) {
      qubitMap[qcQubit] = qcoQubitOut;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.ctrl to qco.ctrl
 *
 * @par Example:
 * ```mlir
 * qc.ctrl(%q0) {
 *   qc.x %q1 : !qc.qubit
 *   qc.yield
 * } : !qc.qubit
 * ```
 * is converted to
 * ```mlir
 * %controls_out, %targets_out = qco.ctrl(%q0_in) %q1_in {
 *   %q1_res = qco.x %q1_in : !qco.qubit -> !qco.qubit
 *   qco.yield %q1_res
 * } : ({!qco.qubit}, {!qco.qubit}) -> ({!qco.qubit}, {!qco.qubit})
 * ```
 */
struct ConvertQCCtrlOp final : StatefulOpConversionPattern<qc::CtrlOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::CtrlOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    auto& qubitMap = state.qubitMap[op->getParentRegion()];

    // Get QCO controls from state map
    const auto& qcControls = op.getControls();
    SmallVector<Value> qcoControls;
    qcoControls.reserve(qcControls.size());

    for (const auto& qcControl : qcControls) {
      assert(qubitMap.contains(qcControl) && "QC qubit not found");
      qcoControls.push_back(qubitMap[qcControl]);
    }

    // Get QCO targets from state map
    const auto numTargets = op.getNumTargets();
    SmallVector<Value> qcoTargets;
    qcoTargets.reserve(numTargets);
    for (size_t i = 0; i < numTargets; ++i) {
      const auto& qcTarget = op.getTarget(i);
      assert(qubitMap.contains(qcTarget) && "QC qubit not found");
      const auto& qcoTarget = qubitMap[qcTarget];
      qcoTargets.push_back(qcoTarget);
    }

    // Create qco.ctrl
    auto qcoOp =
        qco::CtrlOp::create(rewriter, op.getLoc(), qcoControls, qcoTargets);

    // Update the state map if this is a top-level CtrlOp
    // Nested CtrlOps are managed via the targetsIn and targetsOut maps
    if (state.inCtrlOp == 0) {
      for (const auto& [qcControl, qcoControl] :
           llvm::zip(qcControls, qcoOp.getControlsOut())) {
        qubitMap[qcControl] = qcoControl;
      }
      const auto& targetsOut = qcoOp.getTargetsOut();
      for (size_t i = 0; i < numTargets; ++i) {
        const auto& qcTarget = op.getTarget(i);
        qubitMap[qcTarget] = targetsOut[i];
      }
    }

    // Update modifier information
    state.inCtrlOp++;

    // Clone body region from QC to QCO
    auto& dstRegion = qcoOp.getRegion();
    rewriter.cloneRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    // Create block arguments for target qubits and store them in
    // `state.targetsIn`.
    auto& entryBlock = dstRegion.front();
    assert(entryBlock.getNumArguments() == 0 &&
           "QC ctrl region unexpectedly has entry block arguments");
    SmallVector<Value> qcoTargetAliases;
    qcoTargetAliases.reserve(numTargets);
    const auto qubitType = qco::QubitType::get(qcoOp.getContext());
    const auto opLoc = op.getLoc();
    rewriter.modifyOpInPlace(qcoOp, [&] {
      for (auto i = 0UL; i < numTargets; i++) {
        qcoTargetAliases.emplace_back(entryBlock.addArgument(qubitType, opLoc));
      }
    });
    state.targetsIn[state.inCtrlOp] = std::move(qcoTargetAliases);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts qc.yield to qco.yield
 *
 * @par Example:
 * ```mlir
 * qc.yield
 * ```
 * is converted to
 * ```mlir
 * qco.yield %targets
 * ```
 */
struct ConvertQCYieldOp final : StatefulOpConversionPattern<qc::YieldOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(qc::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& state = getState();
    const auto& targets = state.targetsOut[state.inCtrlOp];
    rewriter.replaceOpWithNewOp<qco::YieldOp>(op, targets);
    state.targetsOut.erase(state.inCtrlOp);
    state.inCtrlOp--;
    return success();
  }
};

/**
 * @brief Converts memref.alloc to tensor.from_elements for qubits
 *
 * @par Example:
 * ```mlir
 * %alloc = memref.alloc() : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * %tensor = tensor.from_elements %q0, %q1, %q2 : tensore<3x!qco.qubit>
 * ```
 */
struct ConvertQCMemRefAllocOp final
    : StatefulOpConversionPattern<memref::AllocOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];

    // Get the qco qubits from the users
    SmallVector<Value> qcoQubits;
    const auto users = llvm::to_vector(op->getUsers());
    for (auto* user : llvm::reverse(users)) {
      if (llvm::isa<memref::StoreOp>(user)) {
        auto storeOp = dyn_cast<memref::StoreOp>(user);
        qcoQubits.push_back(qubitMap[storeOp.getValue()]);
      }
    }
    auto const qcoType = qco::QubitType::get(rewriter.getContext());
    const auto tensorType = RankedTensorType::get(
        {static_cast<int64_t>(qcoQubits.size())}, qcoType);
    // Create the FromElements operation
    auto fromElements = tensor::FromElementsOp::create(rewriter, op->getLoc(),
                                                       tensorType, qcoQubits);
    // Add them to the qubitMap
    qubitMap.try_emplace(op->getResult(0), fromElements->getResult(0));

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Deletes the memref.store operation for qubits
 *
 * @par Example:
 * ```mlir
 * memref.store %q0, %alloc[%c0] : memref<3x!qc.qubit>
 * ```
 * is converted to
 * ```mlir
 * ```
 */
struct ConvertQCMemRefStoreOp final
    : StatefulOpConversionPattern<memref::StoreOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts memref.load to tensor.extract for qubits
 *
 * @par Example:
 * ```mlir
 * %q0 = memref.load %memref[%c0] : memref<3x!qco.qubit>
 * ```
 * is converted to
 * ```mlir
 * %q0 = tensor.extract %tensor[%c0] : tensor<3x!qco.qubit>
 * ```
 */
struct ConvertQCMemRefLoadOp final
    : StatefulOpConversionPattern<memref::LoadOp> {
  using StatefulOpConversionPattern::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    const auto tensor = qubitMap[op.getMemRef()];
    auto const qcoType = qco::QubitType::get(rewriter.getContext());
    // Create the extract operation
    auto extractOp = tensor::ExtractOp::create(rewriter, op->getLoc(), qcoType,
                                               tensor, op.getIndices());
    // Update the qubitMap
    qubitMap[op.getResult()] = extractOp.getResult();

    // Erase the old operation
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts scf.if with memory semantics to scf.if with value semantics
 * for qubit values
 *
 * @par Example:
 * ```mlir
 * scf.if %cond {
 *   qc.x %q0 : !qc.qubit
 *   scf.yield
 * }
 * ```
 * is converted to
 * ```mlir
 * %targets_out = scf.if %cond -> (!qco.qubit) {
 *   %q1 = qco.h %q0 : !qco.qubit -> !qco.qubit
 *   scf.yield %q1 : !qco.qubit
 * } else {
 *   scf.yield %q0 : !qco.qubit
 * }
 * ```
 */
struct ConvertQCScfIfOp final : StatefulOpConversionPattern<scf::IfOp> {
  using StatefulOpConversionPattern<scf::IfOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    auto& regionMap = getState().regionMap;
    const auto& qcQubits = regionMap[op];
    const SmallVector<Value> qcValues(qcQubits.begin(), qcQubits.end());

    // Create result typerange
    const SmallVector<Type> qcoTypes(
        qcQubits.size(), qco::QubitType::get(rewriter.getContext()));

    // Create new if operation
    auto newIfOp =
        scf::IfOp::create(rewriter, op->getLoc(), TypeRange{qcoTypes},
                          op.getCondition(), op.getElseRegion().empty());
    auto& thenRegion = newIfOp.getThenRegion();
    auto& elseRegion = newIfOp.getElseRegion();

    // Move the regions of the old operations inside the new operation
    rewriter.inlineRegionBefore(op.getThenRegion(), thenRegion,
                                thenRegion.end());
    // Eliminate the empty block that was created during the initialization
    rewriter.eraseBlock(&thenRegion.front());

    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), elseRegion,
                                  elseRegion.end());
    } else {
      // Create the yield operation if it does not exist yet
      rewriter.setInsertionPointToEnd(&elseRegion.front());
      const auto elseYield =
          scf::YieldOp::create(rewriter, op->getLoc(), qcValues);
      // Mark the yield operation for conversion
      elseYield->setAttr("needChange",
                         StringAttr::get(rewriter.getContext(), "yes"));
    }

    auto& thenRegionQubitMap = getState().qubitMap[&thenRegion];
    auto& elseRegionQubitMap = getState().qubitMap[&elseRegion];

    // Create the qubit map for the regions and update the qubit map for the
    // current region
    for (const auto& [qcQubit, qcoQubit] :
         llvm::zip_equal(qcQubits, newIfOp->getResults())) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      thenRegionQubitMap.try_emplace(qcQubit, qubitMap[qcQubit]);
      elseRegionQubitMap.try_emplace(qcQubit, qubitMap[qcQubit]);
      qubitMap[qcQubit] = qcoQubit;
    }

    // Replace the old entry in the regionMap with the new operation
    const auto& it = regionMap.find(op);
    const auto values = std::move(it->second);
    regionMap.erase(op);
    regionMap.try_emplace(newIfOp, values);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts scf.while with memory semantics to scf.while with value
 * semantics for qubit values.
 *
 * @par Example:
 * ```mlir
 * scf.while : () -> () {
 *   qc.x %q0 : !qc.qubit
 *   scf.condition(%cond)
 * } do {
 *   qc.x %q0 : !qc.qubit
 *   scf.yield
 * }
 * ```
 * is converted to
 * ```mlir
 * %targets_out = scf.while (%arg0 = %q0) : (!qco.qubit) -> !qco.qubit {
 *   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
 *   scf.condition(%cond) %q1 : !qco.qubit
 * } do {
 * ^bb0(%arg0: !qco.qubit):
 *   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
 *   scf.yield %q1 : !qco.qubit
 * }
 * ```
 */
struct ConvertQCScfWhileOp final : StatefulOpConversionPattern<scf::WhileOp> {
  using StatefulOpConversionPattern<scf::WhileOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    auto& regionMap = getState().regionMap;
    const auto& qcQubits = regionMap[op];

    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(qcQubits.size());
    for (const auto& qcQubit : qcQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap[qcQubit]);
    }
    // Create the result typerange
    const SmallVector<Type> qcoTypes(
        qcQubits.size(), qco::QubitType::get(rewriter.getContext()));

    // Create the new while operation
    auto newWhileOp = scf::WhileOp::create(
        rewriter, op.getLoc(), TypeRange(qcoTypes), ValueRange(qcoQubits));
    auto& newBeforeRegion = newWhileOp.getBefore();
    auto& newAfterRegion = newWhileOp.getAfter();
    const SmallVector<Location> locs(qcQubits.size(), op->getLoc());
    // Create the new blocks
    auto* newBeforeBlock =
        rewriter.createBlock(&newBeforeRegion, {}, qcoTypes, locs);
    auto* newAfterBlock =
        rewriter.createBlock(&newAfterRegion, {}, qcoTypes, locs);

    // Move the operations to the new blocks
    newBeforeBlock->getOperations().splice(newBeforeBlock->end(),
                                           op.getBeforeBody()->getOperations());
    newAfterBlock->getOperations().splice(newAfterBlock->end(),
                                          op.getAfterBody()->getOperations());

    auto& newBeforeRegionMap = getState().qubitMap[&newWhileOp.getBefore()];
    auto& newAfterRegionMap = getState().qubitMap[&newWhileOp.getAfter()];

    // Create the qubit map for the new regions and  update the qubit map in the
    // current region
    for (const auto& [qcQubit, beforeArg, afterArg, qcoQubit] : llvm::zip_equal(
             qcQubits, newWhileOp.getBeforeArguments(),
             newWhileOp.getAfterArguments(), newWhileOp->getResults())) {
      newBeforeRegionMap.try_emplace(qcQubit, beforeArg);
      newAfterRegionMap.try_emplace(qcQubit, afterArg);
      qubitMap[qcQubit] = qcoQubit;
    }

    // Replace the old entry in the regionMap with the new operation
    const auto& it = regionMap.find(op);
    const auto values = std::move(it->second);
    regionMap.erase(op);
    regionMap.try_emplace(newWhileOp, values);
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Converts scf.for with memory semantics to scf.for with value
 * semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * scf.for %iv = %lb to %ub step %step {
 *   qc.x %q0 : !qc.qubit
 *   scf.yield
 * }
 * ```
 * is converted to
 * ```mlir
 * %targets_out = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %q0) ->
 * (!qco.qubit) {
 *   %q1 = qco.x %arg0 : !qco.qubit -> !qco.qubit
 *   scf.yield %q1 : !qco.qubit
 * }
 * ```
 */
struct ConvertQCScfForOp final : StatefulOpConversionPattern<scf::ForOp> {
  using StatefulOpConversionPattern<scf::ForOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    auto& regionMap = getState().regionMap;
    const auto& qcQubits = regionMap[op];
    const auto qcoType = qco::QubitType::get(rewriter.getContext());

    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(qcQubits.size());
    for (const auto& qcQubit : qcQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap[qcQubit]);
    }

    // Create a new for-loop with qco qubits as iter_args
    auto newFor = scf::ForOp::create(
        rewriter, op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), ValueRange(qcoQubits));

    // Move the operations to the new block
    auto& srcBlock = op.getRegion().front();
    auto& dstBlock = newFor.getRegion().front();

    dstBlock.getOperations().splice(dstBlock.end(), srcBlock.getOperations());
    rewriter.replaceAllUsesWith(op.getInductionVar(), newFor.getInductionVar());

    auto& newRegion = newFor.getRegion();
    auto& regionQubitMap = getState().qubitMap[&newRegion];

    // Copy the qubit Map into the region
    for (const auto& [key, value] : qubitMap) {
      regionQubitMap[key] = value;
    }

    // Create the qubitmap for the new region and update the qubitmap in the
    // current region
    for (const auto& [qcQubit, iterArg, qcoQubit] : llvm::zip_equal(
             qcQubits, newFor.getRegionIterArgs(), newFor->getResults())) {
      regionQubitMap[qcQubit] = iterArg;
      qubitMap[qcQubit] = qcoQubit;

      // If the value of the qc qubit is a memref register, extract each value
      // from the new tensor and update the qubitmap for each value
      if (llvm::isa<MemRefType>(qcQubit.getType())) {
        // Get all the qubits that were stored in the memref register
        const auto qcQubitUsers = llvm::to_vector(qcQubit.getUsers());
        for (const auto* user : llvm::reverse(qcQubitUsers)) {
          if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
            // gGet the qubit
            const auto qubit = storeOp.getValueToStore();

            // Create the extract operation for each qubit from the resulting
            // tensor of the scf.for operation
            auto extractOp =
                tensor::ExtractOp::create(rewriter, op->getLoc(), qcoType,
                                          qcoQubit, {storeOp.getIndices()});
            // Update the qubit map for each of them
            qubitMap[qubit] = extractOp.getResult();
          }
        }
      }
    }

    // Replace the old entry in the regionMap with the new operation
    const auto& it = regionMap.find(op);
    const auto values = std::move(it->second);
    regionMap.erase(op);
    regionMap.try_emplace(newFor, values);

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts scf.yield with memory semantics to scf.yield with value
 * semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * scf.yield
 * ```
 * is converted to
 * ```mlir
 * scf.yield %targets
 * ```
 */
struct ConvertQCScfYieldOp final : StatefulOpConversionPattern<scf::YieldOp> {
  using StatefulOpConversionPattern<scf::YieldOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    assert(llvm::all_of(op.getOperandTypes(),
                        [&](Type type) { return isQubitType(type); }) &&
           "Not all operands are qc qubits");

    const auto& parentRegion = op->getParentRegion();
    auto& qubitMap = getState().qubitMap[parentRegion];
    const auto& orderedQubits =
        getState().regionMap[parentRegion->getParentOp()];

    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(orderedQubits.size());
    // get the latest qco qubit or the latest qco tensor from the qubitMap
    for (const auto& qcQubit : orderedQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");

      // add an insert operation for every qubit that was extract from a
      // register
      if (dyn_cast<MemRefType>(qcQubit.getType())) {
        // find all extracted values of the register
        for (const auto* user : qcQubit.getUsers()) {
          if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
            // get the latest qco qubit and add it back to the tensor
            auto qubit = loadOp.getResult();
            assert(qubitMap.contains(qubit) && "QC qubit not found");

            auto latestQcoQubit = qubitMap.lookup(qubit);
            auto insertOp = tensor::InsertOp::create(
                rewriter, op.getLoc(), latestQcoQubit, qubitMap[qcQubit],
                loadOp.getIndices());
            qubitMap[qcQubit] = insertOp.getResult();
          }
        }
      }
      qcoQubits.push_back(qubitMap[qcQubit]);
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, qcoQubits);

    return success();
  }
};

/**
 * @brief Converts scf.condition with memory semantics to scf.condition with
 * value semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * scf.condition(%cond)
 * ```
 * is converted to
 * ```mlir
 * scf.condition(%cond) %targets
 * ```
 */
struct ConvertQCScfConditionOp final
    : StatefulOpConversionPattern<scf::ConditionOp> {
  using StatefulOpConversionPattern<
      scf::ConditionOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {

    const auto& parentRegion = op->getParentRegion();
    const auto& qubitMap = getState().qubitMap[parentRegion];
    const auto& orderedQubits =
        getState().regionMap[parentRegion->getParentOp()];

    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(orderedQubits.size());
    for (const auto& qcQubit : orderedQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap.lookup(qcQubit));
    }

    rewriter.replaceOpWithNewOp<scf::ConditionOp>(op, op.getCondition(),
                                                  qcoQubits);
    return success();
  }
};

/**
 * @brief Converts func.call with memory semantics to func.call with
 * value semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * call @test(%q0) : (!qc.qubit) -> ()
 * }
 * ```
 * is converted to
 * ```mlir
 * %q1 = call @test(%q0) : (!qco.qubit) -> !qco.qubit
 * ```
 */
struct ConvertQCFuncCallOp final : StatefulOpConversionPattern<func::CallOp> {
  using StatefulOpConversionPattern<func::CallOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto const qcType = qc::QubitType::get(rewriter.getContext());
    assert(llvm::all_of(op.getOperandTypes(),
                        [&](Type type) { return type == qcType; }) &&
           "Not all operands are qc qubits");

    auto& qubitMap = getState().qubitMap[op->getParentRegion()];
    auto qcQubits = op->getOperands();

    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(qcQubits.size());
    for (const auto& qcQubit : qcQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap[qcQubit]);
    }
    // create the result typerange
    const SmallVector<Type> qcoTypes(
        qcQubits.size(), qco::QubitType::get(rewriter.getContext()));

    const auto callOp = func::CallOp::create(
        rewriter, op->getLoc(), adaptor.getCallee(), qcoTypes, qcoQubits);

    for (const auto& [qcQubit, qcoQubit] :
         llvm::zip_equal(qcQubits, callOp->getResults())) {
      qubitMap[qcQubit] = qcoQubit;
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Converts func.func with memory semantics to func.func with
 * value semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * func.func @test(%arg0: !qc.qubit) {
 * ...
 * }
 * ```
 * is converted to
 * ```mlir
 * func.func @test(%arg0: !qco.qubit) -> !qco.qubit {
 * ...
 * }
 * ```
 */
struct ConvertQCFuncFuncOp final : StatefulOpConversionPattern<func::FuncOp> {
  using StatefulOpConversionPattern<func::FuncOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto const qcType = qc::QubitType::get(rewriter.getContext());
    assert(llvm::all_of(op.getArgumentTypes(),
                        [&](Type type) { return type == qcType; }) &&
           "Not all operands are qc qubits");

    rewriter.modifyOpInPlace(op, [&] {
      auto& qubitMap = getState().qubitMap[&op->getRegion(0)];
      const SmallVector<Type> qcoTypes(
          op.front().getNumArguments(),
          qco::QubitType::get(rewriter.getContext()));

      // set the arguments to qco qubit type
      for (auto blockArg : op.front().getArguments()) {
        blockArg.setType(qco::QubitType::get(rewriter.getContext()));
        qubitMap.try_emplace(blockArg, blockArg);
      }

      // change the function signature to return the same number of qco Qubits
      // as it gets as input
      auto newFuncType = rewriter.getFunctionType(qcoTypes, qcoTypes); //
      op.setFunctionType(newFuncType);
    });
    return success();
  }
};

/**
 * @brief Converts func.return with memory semantics to func.return with
 * value semantics for qubit values
 *
 * @par Example:
 * ```mlir
 * func.return
 * ```
 * is converted to
 * ```mlir
 * func.return %targets
 * ```
 */
struct ConvertQCFuncReturnOp final
    : StatefulOpConversionPattern<func::ReturnOp> {
  using StatefulOpConversionPattern<
      func::ReturnOp>::StatefulOpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter& rewriter) const override {
    auto const qcType = qc::QubitType::get(rewriter.getContext());
    assert(llvm::all_of(op.getOperandTypes(),
                        [&](Type type) { return type == qcType; }) &&
           "Not all operands are qc qubits");

    const auto& parentRegion = op->getParentRegion();
    const auto& qubitMap = getState().qubitMap[parentRegion];
    const auto& orderedQubits =
        getState().regionMap[parentRegion->getParentOp()];

    SmallVector<Value> qcoQubits;
    qcoQubits.reserve(orderedQubits.size());
    for (const auto& qcQubit : orderedQubits) {
      assert(qubitMap.contains(qcQubit) && "QC qubit not found");
      qcoQubits.push_back(qubitMap.lookup(qcQubit));
    }
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, qcoQubits);
    return success();
  }
};

/**
 * @brief Pass implementation for QC-to-QCO conversion
 *
 * @details
 * This pass converts QC dialect operations (reference
 * semantics) to QCO dialect operations (value semantics).
 * The conversion is essential for enabling optimization
 * passes that rely on SSA form and explicit dataflow
 * analysis.
 *
 * The pass operates in several phases:
 * 1. Type conversion: !qc.qubit -> !qco.qubit
 * 2. Operation conversion: Each QC op is converted to
 * its QCO equivalent
 * 3. State tracking: A LoweringState maintains qubit value
 * mappings
 * 4. Function/control-flow adaptation: Function signatures
 * and control flow are updated to use QCO types
 *
 * The conversion maintains semantic equivalence while
 * transforming the representation from imperative
 * (mutation-based) to functional (SSA-based).
 */
struct QCToQCO final : impl::QCToQCOBase<QCToQCO> {
  using QCToQCOBase::QCToQCOBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    // Create state object to track qubit value flow
    LoweringState state;

    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    QCToQCOTypeConverter typeConverter(context);

    // Collect the qubits for each region
    collectUniqueQubits(module, &state, context);
    // Configure conversion target: QC illegal, QCO and tensor
    // legal
    target.addIllegalDialect<QCDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalDialect<QCODialect>();
    target.addLegalDialect<arith::ArithDialect>();

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::WhileOp>([&](scf::WhileOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::ConditionOp>([&](scf::ConditionOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return !(op->getAttrOfType<StringAttr>("needChange"));
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !llvm::any_of(op.front().getArgumentTypes(),
                           [&](Type type) { return isQubitType(type); });
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return !llvm::any_of(op->getOperandTypes(),
                           [&](Type type) { return isQubitType(type); });
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return !op->getAttrOfType<StringAttr>("needChange");
    });
    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return !llvm::any_of(op->getResultTypes(),
                           [&](Type type) { return isQubitType(type); });
    });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      return !llvm::any_of(op.getOperandTypes(),
                           [&](Type type) { return isQubitType(type); });
    });
    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      return !llvm::any_of(op->getResultTypes(),
                           [&](Type type) { return isQubitType(type); });
    });

    // Register operation conversion patterns with state
    // tracking
    patterns
        .add<ConvertQCAllocOp, ConvertQCDeallocOp, ConvertQCStaticOp,
             ConvertQCMeasureOp, ConvertQCResetOp, ConvertQCGPhaseOp,
             ConvertQCIdOp, ConvertQCXOp, ConvertQCYOp, ConvertQCZOp,
             ConvertQCHOp, ConvertQCSOp, ConvertQCSdgOp, ConvertQCTOp,
             ConvertQCTdgOp, ConvertQCSXOp, ConvertQCSXdgOp, ConvertQCRXOp,
             ConvertQCRYOp, ConvertQCRZOp, ConvertQCPOp, ConvertQCROp,
             ConvertQCU2Op, ConvertQCUOp, ConvertQCSWAPOp, ConvertQCiSWAPOp,
             ConvertQCDCXOp, ConvertQCECROp, ConvertQCRXXOp, ConvertQCRYYOp,
             ConvertQCRZXOp, ConvertQCRZZOp, ConvertQCXXPlusYYOp,
             ConvertQCXXMinusYYOp, ConvertQCBarrierOp, ConvertQCCtrlOp,
             ConvertQCYieldOp, ConvertQCMemRefAllocOp, ConvertQCMemRefStoreOp,
             ConvertQCMemRefLoadOp, ConvertQCScfIfOp, ConvertQCScfYieldOp,
             ConvertQCScfWhileOp, ConvertQCScfConditionOp, ConvertQCScfForOp,
             ConvertQCFuncCallOp, ConvertQCFuncFuncOp, ConvertQCFuncReturnOp>(
            typeConverter, context, &state);

    // Apply the conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
