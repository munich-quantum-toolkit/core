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

// Define a macro for hidden visibility, which works on GCC and Clang.
#if defined(__GNUC__) || defined(__clang__)
#define MQT_HIDDEN __attribute__((visibility("hidden")))
#else
#define MQT_HIDDEN
#endif

namespace qdmi {
template <class ConcreteType> class SingletonDevice {
  /// @brief The singleton instance.
  // The MQT_HIDDEN attribute ensures that each module (executable, shared
  // library) gets its own separate instance of this static member, preventing
  // them from being merged by the linker.
  MQT_HIDDEN static std::atomic<ConcreteType*> instance;

protected:
  /// @brief Protected constructor to enforce the singleton pattern.
  SingletonDevice() = default;

public:
  // Delete move constructor and move assignment operator.
  SingletonDevice(SingletonDevice&&) = delete;
  SingletonDevice& operator=(SingletonDevice&&) = delete;
  // Delete copy constructor and assignment operator to enforce singleton.
  SingletonDevice(const SingletonDevice&) = delete;
  SingletonDevice& operator=(const SingletonDevice&) = delete;

  /// @brief Destructor for the SingletonDevice class.
  virtual ~SingletonDevice() = default;

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

// The MQT_HIDDEN attribute must also be applied to the definition.
template <class ConcreteType>
MQT_HIDDEN std::atomic<ConcreteType*> SingletonDevice<ConcreteType>::instance{
    nullptr};
} // namespace qdmi
