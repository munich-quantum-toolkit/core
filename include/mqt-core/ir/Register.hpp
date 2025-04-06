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

#include "Definitions.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace qc {

template <typename BitType> class Register {
public:
  Register(const BitType regStartIndex, const std::size_t regSize,
           std::string regName)
      : startIndex(regStartIndex), size(regSize), name(std::move(regName)) {}
  virtual ~Register() = default;

  [[nodiscard]] const std::string& getName() const noexcept { return name; }
  [[nodiscard]] std::size_t getSize() const noexcept { return size; }
  [[nodiscard]] std::size_t& getSize() noexcept { return size; }
  [[nodiscard]] BitType getStartIndex() const noexcept { return startIndex; }
  [[nodiscard]] BitType& getStartIndex() noexcept { return startIndex; }
  [[nodiscard]] BitType getEndIndex() const noexcept {
    return static_cast<BitType>(startIndex + size - 1);
  }

  [[nodiscard]] bool operator==(const Register& other) const {
    return name == other.name && size == other.size;
  }
  [[nodiscard]] bool operator!=(const Register& other) const {
    return !(*this == other);
  }

  [[nodiscard]] bool contains(const BitType index) const {
    return startIndex <= index && index < startIndex + size;
  }

  [[nodiscard]] BitType getLocalIndex(const BitType globalIndex) const {
    if (!contains(globalIndex)) {
      throw std::out_of_range("Index out of range");
    }
    return globalIndex - startIndex;
  }

  [[nodiscard]] BitType getGlobalIndex(const BitType localIndex) const {
    if (localIndex >= size) {
      throw std::out_of_range("Index out of range");
    }
    return startIndex + localIndex;
  }

  [[nodiscard]] std::string toString(const BitType globalIndex) const {
    return name + "[" + std::to_string(getLocalIndex(globalIndex)) + "]";
  }

  [[nodiscard]] BitType operator[](const BitType localIndex) const {
    return getGlobalIndex(localIndex);
  }

private:
  BitType startIndex;
  std::size_t size;
  std::string name;
};

class QuantumRegister final : public Register<Qubit> {
public:
  QuantumRegister(const Qubit regStartIndex, const std::size_t regSize,
                  const std::string& regName = "")
      : Register(regStartIndex, regSize,
                 regName.empty() ? generateName() : regName) {}

protected:
  static std::string generateName() {
    static std::size_t counter = 0;
    return "q" + std::to_string(counter++);
  }
};

using QubitIndexToRegisterMap =
    std::unordered_map<Qubit, std::pair<const QuantumRegister&, std::string>>;

class ClassicalRegister final : public Register<Bit> {
public:
  ClassicalRegister(const Bit regStartIndex, const std::size_t regSize,
                    const std::string& regName = "")
      : Register(regStartIndex, regSize,
                 regName.empty() ? generateName() : regName) {}

protected:
  static std::string generateName() {
    static std::size_t counter = 0;
    return "c" + std::to_string(counter++);
  }
};

using BitIndexToRegisterMap =
    std::unordered_map<Bit, std::pair<const ClassicalRegister&, std::string>>;

} // namespace qc

template <> struct std::hash<qc::QuantumRegister> {
  std::size_t operator()(const qc::QuantumRegister& reg) const noexcept {
    return qc::combineHash(reg.getStartIndex(), reg.getSize());
  }
};

template <> struct std::hash<qc::ClassicalRegister> {
  std::size_t operator()(const qc::ClassicalRegister& reg) const noexcept {
    return qc::combineHash(reg.getStartIndex(), reg.getSize());
  }
};
