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
#include "ir/operations/Control.hpp"
#include "ir/operations/Operation.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

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

class ClassicControlledOperation final : public Operation {
public:
  ClassicControlledOperation(std::unique_ptr<Operation>&& operation,
                             ClassicalRegister controlReg,
                             std::uint64_t expectedVal = 1U,
                             ComparisonKind kind = Eq);

  ClassicControlledOperation(std::unique_ptr<Operation>&& operation, Bit cBit,
                             std::uint64_t expectedVal = 1U,
                             ComparisonKind kind = Eq);

  ClassicControlledOperation(const ClassicControlledOperation& ccop);

  ClassicControlledOperation& operator=(const ClassicControlledOperation& ccop);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<ClassicControlledOperation>(*this);
  }

  [[nodiscard]] const auto& getControlRegister() const noexcept {
    return controlRegister;
  }
  [[nodiscard]] const auto& getControlBit() const noexcept {
    return controlBit;
  }

  [[nodiscard]] auto getExpectedValue() const noexcept { return expectedValue; }

  [[nodiscard]] auto getOperation() const { return op.get(); }

  [[nodiscard]] auto getComparisonKind() const noexcept {
    return comparisonKind;
  }

  [[nodiscard]] const Targets& getTargets() const override {
    return op->getTargets();
  }

  Targets& getTargets() override { return op->getTargets(); }

  [[nodiscard]] std::size_t getNtargets() const override {
    return op->getNtargets();
  }

  [[nodiscard]] const Controls& getControls() const override {
    return op->getControls();
  }

  Controls& getControls() override { return op->getControls(); }

  [[nodiscard]] std::size_t getNcontrols() const override {
    return op->getNcontrols();
  }

  [[nodiscard]] bool isUnitary() const override { return false; }

  [[nodiscard]] bool actsOn(const Qubit i) const override {
    return op->actsOn(i);
  }

  void addControl(const Control c) override { op->addControl(c); }

  void clearControls() override { op->clearControls(); }

  void removeControl(const Control c) override { op->removeControl(c); }

  Controls::iterator removeControl(const Controls::iterator it) override {
    return op->removeControl(it);
  }

  [[nodiscard]] bool equals(const Operation& operation,
                            const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
                    const BitIndexToRegisterMap& bitMap, std::size_t indent,
                    bool openQASM3) const override;

  void invert() override { op->invert(); }

private:
  std::unique_ptr<Operation> op;
  std::optional<ClassicalRegister> controlRegister;
  std::optional<Bit> controlBit;
  std::uint64_t expectedValue = 1U;
  ComparisonKind comparisonKind = Eq;
};
} // namespace qc

template <> struct std::hash<qc::ClassicControlledOperation> {
  std::size_t
  operator()(qc::ClassicControlledOperation const& ccop) const noexcept {
    auto seed =
        qc::combineHash(std::hash<qc::Operation>{}(*ccop.getOperation()),
                        ccop.getExpectedValue());
    if (const auto& controlRegister = ccop.getControlRegister();
        controlRegister.has_value()) {
      qc::hashCombine(
          seed, std::hash<qc::ClassicalRegister>{}(controlRegister.value()));
    }
    if (const auto& controlBit = ccop.getControlBit(); controlBit.has_value()) {
      qc::hashCombine(seed, controlBit.value());
    }
    return seed;
  }
};
