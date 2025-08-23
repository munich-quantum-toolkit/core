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

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Operation.hpp"

#include <memory>
#include <optional>

namespace qc {

enum ComparisonKind : std::uint8_t {
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
};

ComparisonKind getInvertedComparisonKind(ComparisonKind kind);

std::string toString(const ComparisonKind& kind);

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind);

class IfElseOperation final : public Operation {
public:
  IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                  std::unique_ptr<Operation>&& elseOp,
                  const ClassicalRegister& controlRegister,
                  const std::uint64_t expectedValue = 1U,
                  const ComparisonKind kind = Eq);

  IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                  std::unique_ptr<Operation>&& elseOp, const Bit controlBit,
                  const bool expectedValue = true,
                  const ComparisonKind kind = Eq);

  IfElseOperation(const IfElseOperation& op);

  IfElseOperation& operator=(const IfElseOperation& op);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<IfElseOperation>(*this);
  }

  void apply(const Permutation& permutation) override;

  [[nodiscard]] bool isUnitary() const override { return false; }

  [[nodiscard]] bool isNonUnitaryOperation() const override { return true; }

  [[nodiscard]] bool isIfElseOperation() const noexcept override {
    return true;
  }

  [[nodiscard]] bool isControlled() const override { return false; }

  [[nodiscard]] auto getThenOp() const { return thenOp.get(); }

  [[nodiscard]] auto getElseOp() const { return elseOp.get(); }

  [[nodiscard]] const auto& getControlRegister() const noexcept {
    return controlRegister;
  }

  [[nodiscard]] const auto& getControlBit() const noexcept {
    return controlBit;
  }

  [[nodiscard]] auto getExpectedValueRegister() const noexcept {
    return expectedValueRegister;
  }

  [[nodiscard]] bool getExpectedValueBit() const noexcept {
    return expectedValueBit;
  }

  [[nodiscard]] auto getComparisonKind() const noexcept {
    return comparisonKind;
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& op) const override {
    return equals(op, {}, {});
  }

  virtual std::ostream& print(std::ostream& os, const Permutation& permutation,
                              std::size_t prefixWidth,
                              std::size_t nqubits) const override;

  void dumpOpenQASM(std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
                    const BitIndexToRegisterMap& bitMap, std::size_t indent,
                    bool openQASM3) const override;

  void invert() override {
    throw std::runtime_error("An IfElseOperation cannot be inverted.");
  }

  // Override invalid Operation setters
  void setTargets(const Targets&) override {
    throw std::runtime_error("An IfElseOperation does not have a target.");
  }

  void setControls(const Controls&) override {
    throw std::runtime_error("An IfElseOperation cannot be controlled.");
  }
  void addControl(Control) override {
    throw std::runtime_error("An IfElseOperation cannot be controlled.");
  }
  void clearControls() override {
    throw std::runtime_error("An IfElseOperation cannot be controlled.");
  }
  void removeControl(Control) override {
    throw std::runtime_error("An IfElseOperation cannot be controlled.");
  }
  Controls::iterator removeControl(Controls::iterator) override {
    throw std::runtime_error("An IfElseOperation cannot be controlled.");
  }

  void setGate(const OpType) override {
    throw std::runtime_error(
        "Cannot set operation type of an IfElseOperation.");
  }

  void setParameter(const std::vector<fp>&) override {
    throw std::runtime_error("An IfElseOperation cannot be parameterized.");
  }

private:
  std::unique_ptr<Operation> thenOp;
  std::unique_ptr<Operation> elseOp;
  std::optional<ClassicalRegister> controlRegister;
  std::optional<Bit> controlBit;
  std::uint64_t expectedValueRegister = 1U;
  bool expectedValueBit = true;
  ComparisonKind comparisonKind = Eq;
};
} // namespace qc

namespace std {
template <> struct hash<qc::IfElseOperation> {
  std::size_t operator()(qc::IfElseOperation const& op) const noexcept;
};
} // namespace std
