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

#include <cassert>
#include <memory>
#include <mutex>

// Define a macro for hidden visibility, which works on GCC and Clang.
#if defined(__GNUC__) || defined(__clang__)
#define MQT_HIDDEN __attribute__((visibility("hidden")))
#else
#define MQT_HIDDEN
#endif

namespace qdmi {
template <class ConcreteType> class SingletonDevice {
  /**
   * @brief Provides a pointer to the singleton instance.
   * @return A pointer to the static shared_ptr holding the instance.
   * @details This uses the "Construct on First Use" idiom.
   */
  static auto getInstancePtr() -> std::shared_ptr<ConcreteType>* {
    MQT_HIDDEN static auto* instance =
        new std::shared_ptr<ConcreteType>(nullptr);
    return instance;
  }

  /**
   * @brief Get the mutex for this singleton instance.
   * @return A reference to the mutex.
   */
  static auto getMutexPtr() -> std::mutex* {
    MQT_HIDDEN static auto* instanceMutex = new std::mutex;
    return instanceMutex;
  }

protected:
  /// @brief Protected constructor to enforce the singleton pattern.
  SingletonDevice() = default;
  // Allow std::make_shared to access the protected constructor.
  friend class std::shared_ptr<ConcreteType>;

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
    const std::scoped_lock lock(*getMutexPtr());
    auto* instance = getInstancePtr();
    if (*instance == nullptr) {
      *instance = std::shared_ptr<ConcreteType>(new ConcreteType);
    }
  }

  /**
   * @brief Destroys the singleton instance.
   * @details After this call, `get()` must not be called until a new
   * `initialize()` call. Any existing shared_ptr will keep the object alive
   * until they go out of scope.
   */
  static void finalize() {
    const std::scoped_lock lock(*getMutexPtr());
    *getInstancePtr() = nullptr;
    delete getInstancePtr();
    delete getMutexPtr();
  }

  /**
   * @brief Get the singleton instance of the device.
   * @return A shared pointer to the device instance.
   */
  [[nodiscard]] static auto get() -> std::shared_ptr<ConcreteType> {
    const std::scoped_lock lock(*getMutexPtr());
    auto* instance = getInstancePtr();
    assert(*instance != nullptr &&
           "Device not initialized. Call `initialize()` first.");
    return *instance;
  }
};
} // namespace qdmi
