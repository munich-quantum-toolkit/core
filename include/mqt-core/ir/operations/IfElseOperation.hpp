#pragma once

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/Operation.hpp"

#include <memory>
#include <optional>

namespace qc {
class IfElseOperation final : public Operation {
public:
  IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                  std::unique_ptr<Operation>&& elseOp,
                  ClassicalRegister controlReg, std::uint64_t expectedVal = 1U,
                  ComparisonKind kind = Eq);
  IfElseOperation(std::unique_ptr<Operation>&& thenOp,
                  std::unique_ptr<Operation>&& elseOp, Bit cBit,
                  std::uint64_t expectedVal = 1U, ComparisonKind kind = Eq);
  IfElseOperation(const IfElseOperation& op);
  IfElseOperation& operator=(const IfElseOperation& op);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<IfElseOperation>(*this);
  }

  [[nodiscard]] bool isUnitary() const override { return false; }
  [[nodiscard]] bool isNonUnitaryOperation() const override { return true; }

  [[nodiscard]] const Operation* getThenBranch() const {
    return thenBranch.get();
  }
  [[nodiscard]] const Operation* getElseBranch() const {
    return elseBranch.get();
  }

  [[nodiscard]] const auto& getControlRegister() const noexcept {
    return controlRegister;
  }
  [[nodiscard]] const auto& getControlBit() const noexcept {
    return controlBit;
  }
  [[nodiscard]] auto getExpectedValue() const noexcept { return expectedValue; }
  [[nodiscard]] auto getComparisonKind() const noexcept {
    return comparisonKind;
  }

  void addControl(Control) override {}
  void clearControls() override {}
  void removeControl(Control) override {}
  Controls::iterator removeControl(Controls::iterator it) override {
    return it;
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& op) const override {
    return equals(op, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
                    const BitIndexToRegisterMap& bitMap, std::size_t indent,
                    bool openQASM3) const override;

  void invert() override {
    thenBranch->invert();
    if (elseBranch) {
      elseBranch->invert();
    }
  }

private:
  std::unique_ptr<Operation> thenBranch;
  std::unique_ptr<Operation> elseBranch;
  std::optional<ClassicalRegister> controlRegister;
  std::optional<Bit> controlBit;
  std::uint64_t expectedValue = 1U;
  ComparisonKind comparisonKind = Eq;
};
} // namespace qc

namespace std {
template <> struct hash<qc::IfElseOperation> {
  std::size_t operator()(qc::IfElseOperation const& op) const noexcept;
};
} // namespace std
