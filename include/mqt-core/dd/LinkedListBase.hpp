/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Linked list functionality required for UniqueTable and MemoryManager.
 */

#pragma once

namespace dd {

/**
 * @brief A class to provide a base for linked list objects
 */
struct LLBase {
  /**
   * @brief The pointer to the next object
   * @details The next pointer is used to form linked lists of objects.
   * Classes used in a linked list must solely inherit from this class.
   * Other code in mqt-core relies on the assumption that all objects in a
   * linked list are of the same type.
   */
  LLBase* next_ = nullptr;

  /**
   * @brief Default getter for the next object
   * @details Classes that inherit from LLBase should implement their own next()
   * method to return the next object in the list with a specialized return
   * type.
   * @return LLBase*
   */
  [[nodiscard]] LLBase* next() const noexcept { return next_; }

  /// Setter for the next object
  void setNext(LLBase* n) noexcept { next_ = n; }
};

} // namespace dd
