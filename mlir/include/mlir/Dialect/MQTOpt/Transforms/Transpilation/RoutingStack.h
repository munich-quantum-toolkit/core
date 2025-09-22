/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include <cstddef>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {
/**
 * @brief Manages the routing state stack with clear semantics for accessing
 * current and parent states.
 */
template <class StackItem> class [[nodiscard]] RoutingStack {
public:
  /**
   * @brief Returns the current (most recent) state.
   */
  [[nodiscard]] StackItem& getState() {
    assert(!stack_.empty() && "getState: empty state stack");
    return stack_.back();
  }

  /**
   * @brief Returns the parent of the current state.
   */
  [[nodiscard]] StackItem& getParentState() {
    assert(stack_.size() >= 2 && "getParentState: no parent state available");
    return stack_[stack_.size() - 2];
  }

  /**
   * @brief Returns the state at the given index.
   */
  [[nodiscard]] StackItem& at(const std::size_t index) {
    assert(index < stack_.size() && "at: index out of bounds");
    return stack_[index];
  }

  /**
   * @brief Pushes a new item on to the stack.
   */
  void push(StackItem item) { stack_.emplace_back(item); }

  /**
   * @brief Duplicates the current state and pushes it on the stack.
   */
  void duplicateCurrentState() {
    assert(!stack_.empty() && "duplicateCurrentState: empty state stack");
    stack_.emplace_back(stack_.back());
  }

  /**
   * @brief Pops the current state off the stack.
   */
  void pop() {
    assert(!stack_.empty() && "popState: empty state stack");
    stack_.pop_back();
  }

  /**
   * @brief Returns the number of states in the stack.
   */
  [[nodiscard]] std::size_t size() const { return stack_.size(); }

  /**
   * @brief Returns whether the stack is empty.
   */
  [[nodiscard]] bool empty() const { return stack_.empty(); }

  /**
   * @brief Remove all items from the stack.
   */
  void clear() { stack_.clear(); }

private:
  mlir::SmallVector<StackItem, 2> stack_;
};
} // namespace mqt::ir::opt
