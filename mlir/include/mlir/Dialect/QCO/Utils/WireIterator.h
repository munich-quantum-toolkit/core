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

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstdint>
#include <iterator>

namespace mlir::qco {

/**
 * @brief Resolves how qubits flow across a call boundary.
 *
 * @details
 * Rather than assuming that the i-th qubit operand of a call becomes its i-th
 * qubit result, the mapping is derived from the callee: every qubit argument is
 * threaded through the callee's body and the result it ends up in is recorded.
 * A callee is therefore free to return its qubits in a different order than it
 * takes them, or to keep some of them entirely.
 *
 * The derived mapping is cached per callee, so threading a body costs once
 * rather than once per traversal step. Instances are cheap to create; share one
 * across a pass to get the caching.
 *
 * @par Cases that fall back to position:
 * An external declaration has no body to thread, so its qubits are paired by
 * position — that is the only contract available. The same fallback applies
 * while a callee's mapping is already being computed, which happens for
 * recursive functions.
 */
class CallQubitMapping {
public:
  /**
   * @brief Get the call result that continues the wire of a qubit operand.
   *
   * @param callOp The call the qubit flows into.
   * @param operand The qubit operand of @p callOp.
   * @return The matching qubit result, or a null value when the callee keeps
   * the qubit.
   */
  [[nodiscard]] Value getResultForOperand(func::CallOp callOp, Value operand);

  /**
   * @brief Get the call operand that feeds the wire of a qubit result.
   *
   * @param callOp The call the qubit comes out of.
   * @param result The qubit result of @p callOp.
   * @return The matching qubit operand, or a null value when the callee creates
   * the qubit.
   */
  [[nodiscard]] Value getOperandForResult(func::CallOp callOp, Value result);

private:
  /// Marks a qubit argument that never reaches a result.
  static constexpr int64_t KEPT = -1;

  /// @returns for each qubit argument position of @p callOp's callee, the index
  /// of the call result it flows into, or `KEPT`.
  ArrayRef<int64_t> mappingFor(func::CallOp callOp);

  /// Derive the mapping of @p callee by threading each of its qubit arguments.
  SmallVector<int64_t> computeMapping(func::FuncOp callee);

  /// Pair qubit arguments and results by position, used when no body is
  /// available.
  static SmallVector<int64_t> positionalMapping(func::FuncOp callee);

  DenseMap<Operation*, SmallVector<int64_t>> cache;
  DenseSet<Operation*> inProgress;
};

/**
 * @brief A bidirectional_iterator traversing the def-use chain of a qubit wire.
 *
 * The iterator follows the flow of a qubit through a sequence of quantum
 * operations while respecting the semantics of the respective operation.
 **/
class [[nodiscard]] WireIterator {
public:
  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = Operation*;

  WireIterator()
      : op_(nullptr), qubit_(nullptr), isFinal_(false), isSentinel_(false) {}

  /**
   * @brief Construct an iterator over the wire of @p qubit.
   *
   * @param qubit The qubit value to start at.
   * @param mapping Optional shared cache for resolving calls. Passing one only
   * affects performance, but the difference is not constant: without it every
   * step across a call re-threads the callee, and the nested calls met while
   * doing so are re-threaded in turn. Share one when traversing in a loop.
   */
  explicit WireIterator(Value qubit, CallQubitMapping* mapping = nullptr)
      : op_(qubit.getDefiningOp()), qubit_(qubit), isFinal_(false),
        isSentinel_(false), mapping_(mapping) {}

  /// @returns the operation the iterator points to.
  [[nodiscard]] Operation* operation() const { return op_; }

  /// @returns the operation the iterator points to.
  [[nodiscard]] Operation* operator*() const { return operation(); }

  /// @returns the qubit the iterator points to.
  [[nodiscard]] Value qubit() const;

  WireIterator& operator++() {
    forward();
    return *this;
  }

  WireIterator operator++(int) {
    auto tmp = *this;
    operator++();
    return tmp;
  }

  WireIterator& operator--() {
    backward();
    return *this;
  }

  WireIterator operator--(int) {
    auto tmp = *this;
    operator--();
    return tmp;
  }

  bool operator==(const WireIterator& other) const {
    return other.qubit_ == qubit_ && other.op_ == op_ &&
           other.isSentinel_ == isSentinel_;
  }

  bool operator==([[maybe_unused]] std::default_sentinel_t s) const {
    return isSentinel_;
  }

  /**
   * @brief Check whether the iterator sits at the start of the wire.
   *
   * @details
   * At the start, backward traversal can no longer make progress. Besides
   * allocations and the like, this is also the case for a call that creates the
   * qubit rather than threading one through.
   *
   * @return True if `operator--` would not move the iterator.
   */
  [[nodiscard]] bool atWireStart() const;

private:
  /// Return true, if an op doesn't return, but only consumes, a qubit value.
  static bool isSinkLikeOperation(Operation* op);

  /// Return true, if an op doesn't consume, but only returns, a qubit value.
  static bool isSourceLikeOperation(Operation* op);

  /// Move to the next operation on the qubit wire.
  void forward();

  /// Move to the previous operation on the qubit wire.
  void backward();

  Operation* op_;
  Value qubit_;
  bool isFinal_;
  bool isSentinel_;
  /// @returns the call result continuing the wire of @p operand, resolved
  /// through the shared mapping when one was supplied.
  Value resultForOperand(func::CallOp callOp, Value operand) const;

  /// @returns the call operand feeding the wire of @p result, resolved through
  /// the shared mapping when one was supplied.
  Value operandForResult(func::CallOp callOp, Value result) const;

  /// Shared, cached call mapping. Null means each query threads the callee
  /// afresh, which is still correct but not cached. The iterator holds only
  /// this pointer, so that copying it stays cheap.
  CallQubitMapping* mapping_ = nullptr;
};

/**
 * @brief Categorizes the current traversal direction.
 */
enum class WireDirection : std::uint8_t { Forward, Backward };

template <WireDirection Direction> struct WireTraversalTraits {};

template <> struct WireTraversalTraits<WireDirection::Forward> {
  /// @returns the forward increment stride size.
  static constexpr std::ptrdiff_t stride() { return 1; }

  /// @returns true if the wire iterator can continue forward.
  static bool isActive(const WireIterator& it) {
    return it != std::default_sentinel;
  }
};

template <> struct WireTraversalTraits<WireDirection::Backward> {
  /// @returns the backward increment stride size.
  static constexpr std::ptrdiff_t stride() { return -1; }

  /// @returns true if the wire iterator can continue backward.
  static bool isActive(const WireIterator& it) { return !it.atWireStart(); }
};

/**
 * @brief A range over the def-use chain of a qubit wire, usable in range-based
 * for-loops.
 *
 * Example:
 * @code
 * for (auto* op : WireRange(qubit)) { ... }
 * @endcode
 */
struct WireRange {
  explicit WireRange(Value qubit) : begin_(qubit) {}

  [[nodiscard]] WireIterator begin() const { return begin_; }
  [[nodiscard]] static std::default_sentinel_t end() {
    return std::default_sentinel;
  }

private:
  WireIterator begin_;
};
} // namespace mlir::qco
