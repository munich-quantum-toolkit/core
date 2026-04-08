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

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

#include <llvm/ADT/ADL.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cstddef>

namespace mlir::qco {

/**
 * @brief Specifies the layering direction.
 */
enum class WalkDirection : bool { Forward, Backward };

using ReleasedIterators = SmallVector<WireIterator*, 8>;
using FrontArrayRef = ArrayRef<ArrayRef<WireIterator*>>;
using WalkCircuitGraphFn =
    function_ref<WalkResult(FrontArrayRef, ReleasedIterators&)>;

class Qubits {
public:
  /**
   * @brief Add qubit with automatically assigned index.
   */
  void add(TypedValue<QubitType> q);

  /**
   * @brief Add qubit with index.
   */
  void add(TypedValue<QubitType> q, std::size_t index);

  /**
   * @brief Remap the qubit value from prev to next.
   */
  void remap(TypedValue<QubitType> prev, TypedValue<QubitType> next,
             const WalkDirection& direction);

  /**
   * @brief Remap all input qubits of the unitary to its outputs.
   */
  void remap(UnitaryOpInterface op, const WalkDirection& direction);

  /**
   * @brief Remove the qubit value.
   */
  void remove(TypedValue<QubitType> q);

  /**
   * @returns the qubit value assigned to a index.
   */
  [[nodiscard]] TypedValue<QubitType> getQubit(std::size_t index) const;

  /**
   * @returns the index assigned to the given qubit value.
   */
  [[nodiscard]] std::size_t getIndex(TypedValue<QubitType> q) const;

private:
  DenseMap<std::size_t, TypedValue<QubitType>> indexToValue_;
  DenseMap<TypedValue<QubitType>, std::size_t> valueToIndex_;
};

using WalkUnitFn = function_ref<WalkResult(Operation*, const Qubits&)>;

/**
 * @brief Perform top-down non-recursive walk of all operations within a
 * region and apply callback function.
 * @details
 * The signature of the callback function is:
 *
 *     (Operation*, Qubits& q) -> WalkResult
 *
 * where the Qubits object tracks the front of qubit SSA values.
 *
 * @param region The targeted region.
 * @param fn The callback function.
 */
void walkUnit(Region& region, WalkUnitFn fn);

/**
 * @brief Walk the two-qubit block spanned by the wires @p first and @p second.
 * @details
 * Advances each of the two wire iterators until a two-qubit op is
 * found. If the ops match, repeat this process. Otherwise, stop.
 */
void walkQubitBlock(WireIterator& first, WireIterator& second,
                    WalkDirection direction);

/**
 * @brief Walk the graph-like circuit IR of QCO dialect programs.
 * @details
 * Depending on the template parameter, the function collects the
 * layers in forward or backward direction, respectively. Towards that end,
 * the function traverses the def-use chain of each qubit until a two-qubit
 * gate is found. If a two-qubit gate is visited twice, it is considered ready
 * and inserted into the layer. This process is repeated until no more
 * two-qubit are found anymore.
 *
 * The signature of the callback function is:
 *
 *     (FrontArrayRef, ReleasedIterator&) -> WalkResult
 *
 * The wire iterators inserted into the parameter "released" determine which
 * two-qubit gates are released in next iteration.
 *
 * @param wires A mutable array-ref of circuit wires (wire iterators).
 * @param direction The traversal direction.
 * @param fn The callback function.
 *
 * @returns
 *     failure(), if the callback returns WalkResult::interrupt()
 *     failure(), if the callback returns WalkResult::skipped()
 *     success(), otherwise.
 */
LogicalResult walkCircuitGraph(MutableArrayRef<WireIterator> wires,
                               WalkDirection direction, WalkCircuitGraphFn fn);
} // namespace mlir::qco
