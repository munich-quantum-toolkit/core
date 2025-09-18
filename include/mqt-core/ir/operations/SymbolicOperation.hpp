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
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace qc {

class SymbolicOperation final : public StandardOperation {
public:
  SymbolicOperation() = default;

  [[nodiscard]] SymbolOrNumber getParameter(std::size_t i) const;

  [[nodiscard]] std::vector<SymbolOrNumber> getParameters() const;

  void setSymbolicParameter(const Symbolic& par, const std::size_t i) {
    symbolicParameter.at(i) = par;
  }

  // Standard Constructors
  SymbolicOperation(Qubit target, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});
  SymbolicOperation(const Targets& targ, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  SymbolicOperation(Control control, Qubit target, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});
  SymbolicOperation(Control control, const Targets& targ, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  SymbolicOperation(const Controls& c, Qubit target, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});
  SymbolicOperation(const Controls& c, const Targets& targ, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  // MCF (cSWAP), Peres, parameterized two target Constructor
  SymbolicOperation(const Controls& c, Qubit target0, Qubit target1, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  [[nodiscard]] std::unique_ptr<Operation> clone() const override;

  [[nodiscard]] bool isSymbolicOperation() const override;

  [[nodiscard]] bool isStandardOperation() const override;

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& op) const override {
    return equals(op, {}, {});
  }

  [[noreturn]] void dumpOpenQASM(std::ostream& of,
                                 const QubitIndexToRegisterMap& qubitMap,
                                 const BitIndexToRegisterMap& bitMap,
                                 std::size_t indent,
                                 bool openQASM3) const override;

  [[nodiscard]] StandardOperation
  getInstantiatedOperation(const VariableAssignment& assignment) const;

  // Instantiates this Operation
  // Afterwards casting to StandardOperation can be done if assignment is total
  void instantiate(const VariableAssignment& assignment);

  void invert() override;

protected:
  std::vector<std::optional<Symbolic>> symbolicParameter;

  static OpType parseU3(const Symbolic& theta, fp& phi, fp& lambda);
  static OpType parseU3(fp& theta, const Symbolic& phi, fp& lambda);
  static OpType parseU3(fp& theta, fp& phi, const Symbolic& lambda);
  static OpType parseU3(const Symbolic& theta, const Symbolic& phi, fp& lambda);
  static OpType parseU3(const Symbolic& theta, fp& phi, const Symbolic& lambda);
  static OpType parseU3(fp& theta, const Symbolic& phi, const Symbolic& lambda);

  static OpType parseU2(const Symbolic& phi, const Symbolic& lambda);
  static OpType parseU2(const Symbolic& phi, fp& lambda);
  static OpType parseU2(fp& phi, const Symbolic& lambda);

  static OpType parseU1(const Symbolic& lambda);

  void checkSymbolicUgate();

  void storeSymbolOrNumber(const SymbolOrNumber& param, std::size_t i);

  [[nodiscard]] bool isSymbolicParameter(std::size_t i) const;

  static bool isSymbol(const SymbolOrNumber& param);

  static Symbolic& getSymbol(SymbolOrNumber& param);

  static fp& getNumber(SymbolOrNumber& param);

  void setup(const std::vector<SymbolOrNumber>& params);

  [[nodiscard]] static fp
  getInstantiation(const SymbolOrNumber& symOrNum,
                   const VariableAssignment& assignment);

  void negateSymbolicParameter(std::size_t index);

  void addToSymbolicParameter(std::size_t index, fp value);
};
} // namespace qc

template <> struct std::hash<qc::SymbolicOperation> {
  std::size_t operator()(qc::SymbolicOperation const& op) const noexcept {
    std::size_t seed = 0U;
    qc::hashCombine(seed, std::hash<qc::Operation>{}(op));
    for (const auto& param : op.getParameters()) {
      if (std::holds_alternative<qc::fp>(param)) {
        qc::hashCombine(seed, hash<qc::fp>{}(get<qc::fp>(param)));
      } else {
        qc::hashCombine(seed, hash<qc::Symbolic>{}(get<qc::Symbolic>(param)));
      }
    }
    return seed;
  }
};
