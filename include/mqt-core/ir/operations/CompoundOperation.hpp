/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Control.hpp"
#include "Operation.hpp"
#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <set>
#include <vector>

namespace qc {

class CompoundOperation final : public Operation {
public:
  using iterator = std::vector<std::unique_ptr<Operation>>::iterator;
  using const_iterator =
      std::vector<std::unique_ptr<Operation>>::const_iterator;

private:
  std::vector<std::unique_ptr<Operation>> ops;
  bool customGate;

public:
  explicit CompoundOperation(bool isCustom = false);

  explicit CompoundOperation(
      std::vector<std::unique_ptr<Operation>>&& operations,
      bool isCustom = false);

  CompoundOperation(const CompoundOperation& co);

  CompoundOperation& operator=(const CompoundOperation& co);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override;

  [[nodiscard]] bool isCompoundOperation() const noexcept override;

  [[nodiscard]] bool isNonUnitaryOperation() const override;

  [[nodiscard]] bool isSymbolicOperation() const override;

  [[nodiscard]] bool isCustomGate() const noexcept;

  [[nodiscard]] bool isGlobal(size_t nQubits) const noexcept override;

  void addControl(Control c) override;

  void clearControls() override;

  void removeControl(Control c) override;

  Controls::iterator removeControl(Controls::iterator it) override;

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override;

  std::ostream& print(std::ostream& os, const Permutation& permutation,
                      std::size_t prefixWidth,
                      std::size_t nqubits) const override;

  [[nodiscard]] bool actsOn(Qubit i) const override;

  void addDepthContribution(std::vector<std::size_t>& depths) const override;

  void dumpOpenQASM(std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
                    const BitIndexToRegisterMap& bitMap, std::size_t indent,
                    bool openQASM3) const override;

  std::vector<std::unique_ptr<Operation>>& getOps() noexcept { return ops; }

  [[nodiscard]] auto getUsedQubitsPermuted(const Permutation& perm) const
      -> std::set<Qubit> override;

  [[nodiscard]] auto commutesAtQubit(const Operation& other,
                                     const Qubit& qubit) const -> bool override;

  /**
   * This refines the inherited method because the inherited method leads to
   * false negatives
   */
  [[nodiscard]] auto isInverseOf(const Operation& other) const -> bool override;

  void invert() override;

  void apply(const Permutation& permutation) override;

  /**
   * @brief Merge another compound operation into this one.
   * @details This transfers ownership of the operations from the other compound
   * operation to this one. The other compound operation will be empty after
   * this operation.
   * @param op the compound operation to merge into this one
   */
  void merge(CompoundOperation& op);

  /**
   * @brief Check whether this operation can be collapsed into a single
   * operation.
   * @return true if this operation can be collapsed into a single operation,
   * false otherwise
   */
  [[nodiscard]] bool isConvertibleToSingleOperation() const;

  /**
   * @brief Collapse this operation into a single operation.
   * @details This operation must be convertible to a single operation.
   * @return the collapsed operation
   */
  [[nodiscard]] std::unique_ptr<Operation> collapseToSingleOperation();

  /**
   * Pass-Through
   */

  // Iterators (pass-through)
  auto begin() noexcept { return ops.begin(); }
  [[nodiscard]] auto begin() const noexcept { return ops.begin(); }
  [[nodiscard]] auto cbegin() const noexcept { return ops.cbegin(); }
  auto end() noexcept { return ops.end(); }
  [[nodiscard]] auto end() const noexcept { return ops.end(); }
  [[nodiscard]] auto cend() const noexcept { return ops.cend(); }
  auto rbegin() noexcept { return ops.rbegin(); }
  [[nodiscard]] auto rbegin() const noexcept { return ops.rbegin(); }
  [[nodiscard]] auto crbegin() const noexcept { return ops.crbegin(); }
  auto rend() noexcept { return ops.rend(); }
  [[nodiscard]] auto rend() const noexcept { return ops.rend(); }
  [[nodiscard]] auto crend() const noexcept { return ops.crend(); }

  // Capacity (pass-through)
  [[nodiscard]] bool empty() const noexcept { return ops.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] std::size_t max_size() const noexcept { return ops.max_size(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return ops.capacity(); }

  void reserve(const std::size_t newCap) { ops.reserve(newCap); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  void shrink_to_fit() { ops.shrink_to_fit(); }

  // Modifiers (pass-through)
  void clear() noexcept { ops.clear(); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  void pop_back() { ops.pop_back(); }
  void resize(const std::size_t count) { ops.resize(count); }
  iterator erase(const const_iterator pos) { return ops.erase(pos); }
  iterator erase(const const_iterator first, const const_iterator last) {
    return ops.erase(first, last);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T, class... Args> void emplace_back(Args&&... args) {
    ops.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>& op) {
    ops.emplace_back(std::move(op));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>&& op) {
    ops.emplace_back(std::move(op));
  }

  template <class T, class... Args>
  iterator insert(const_iterator iter, Args&&... args) {
    return ops.insert(iter, std::make_unique<T>(std::forward<Args>(args)...));
  }
  template <class T>
  iterator insert(const_iterator iter, std::unique_ptr<T>& op) {
    return ops.insert(iter, std::move(op));
  }
  template <class T> iterator insert(const_iterator iter, T&& op) {
    return ops.insert(iter, std::forward<decltype(op)>(op));
  }

  // Element access (pass-through)
  [[nodiscard]] const auto& at(const std::size_t i) const { return ops.at(i); }
  [[nodiscard]] auto& operator[](const std::size_t i) { return ops[i]; }
  [[nodiscard]] const auto& operator[](const std::size_t i) const {
    return ops[i];
  }
  [[nodiscard]] auto& front() { return ops.front(); }
  [[nodiscard]] const auto& front() const { return ops.front(); }
  [[nodiscard]] auto& back() { return ops.back(); }
  [[nodiscard]] const auto& back() const { return ops.back(); }
};
} // namespace qc

template <> struct std::hash<qc::CompoundOperation> {
  std::size_t operator()(const qc::CompoundOperation& co) const noexcept;
}; // namespace std
