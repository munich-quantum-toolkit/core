/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

/** @file
 * @brief The generic MQT QDMI device implementation that specific devices
 * inherit from.
 */

#include <atomic>
#include <cassert>

namespace qdmi {
template <class ConcreteType> class Device {
  /// @brief The singleton instance.
  inline static std::atomic<ConcreteType*> instance = nullptr;

protected:
  /// @brief Protected constructor to enforce the singleton pattern.
  Device() = default;

public:
  // Delete move constructor and move assignment operator.
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  // Delete copy constructor and assignment operator to enforce singleton.
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  /// @brief Destructor for the Device class.
  virtual ~Device() = default;

  /**
   * @brief Initializes the singleton instance.
   * @details Must be called before `get()`.
   */
  static void initialize() {
    // NOLINTNEXTLINE(misc-const-correctness)
    ConcreteType* expected = nullptr;
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* newInstance = new ConcreteType();
    if (!instance.compare_exchange_strong(expected, newInstance)) {
      // Another thread won the race, so delete the instance we created.
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      delete newInstance;
    }
  }

  /**
   * @brief Destroys the singleton instance.
   * @details After this call, `get()` must not be called until a new
   * `initialize()` call.
   */
  static void finalize() {
    // Atomically swap the instance pointer with nullptr and get the old value.
    const auto* oldInstance = instance.exchange(nullptr);
    // Delete the old instance if it existed.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete oldInstance;
  }

  /**
   * @brief Get the singleton instance of the device.
   * @return A shared pointer to the device instance.
   */
  [[nodiscard]] static auto get() -> ConcreteType& {
    auto* loadedInstance = instance.load();
    assert(loadedInstance != nullptr &&
           "Device not initialized. Call `initialize()` first.");
    return *loadedInstance;
  }
};
} // namespace qdmi
