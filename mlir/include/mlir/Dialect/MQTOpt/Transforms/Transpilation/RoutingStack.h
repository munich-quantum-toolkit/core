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
#include <deque>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace mqt::ir::opt {
/**
 * @brief Manages the routing state stack with clear semantics for accessing
 * current and parent states.
 */
template <class Item> class [[nodiscard]] RoutingStack {
public:
  /**
   * @brief Returns the top of the stack.
   */
  [[nodiscard]] Item& top() {
    assert(!stack_.empty() && "top: empty state stack");
    return stack_.back();
  }

  /**
   * @brief Returns the item at the specified depth from the top of the stack.
   */
  [[nodiscard]] Item& getItemAtDepth(std::size_t depth) {
    assert(depth < stack_.size() && "getItemAtDepth: depth out of bounds");
    return stack_[stack_.size() - 1 - depth];
  }

  /**
   * @brief Pushes a new item on to the stack.
   */
  void push(Item item) { stack_.emplace_back(item); }

  /**
   * @brief Constructs a new item in-place at the top of the stack.
   */
  template <typename... Args> void emplace(Args&&... args) {
    stack_.emplace_back(std::forward<Args>(args)...);
  }

  /**
   * @brief Duplicates the top item.
   */
  void duplicateTop() {
    assert(!stack_.empty() && "duplicateTop: empty state stack");
    stack_.emplace_back(stack_.back());
  }

  /**
   * @brief Pops the top off the stack.
   */
  void pop() {
    assert(!stack_.empty() && "pop: empty item stack");
    stack_.pop_back();
  }

  /**
   * @brief Returns the number of items in the stack.
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
  std::deque<Item> stack_;
};
} // namespace mqt::ir::opt
