/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/Operation.hpp"

#include <cstdint>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <sstream>
#include <utility>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerIfElseOperation(const nb::module_& m) {
  nb::enum_<qc::ComparisonKind>(
      m, "ComparisonKind", "enum.Enum",
      "Enumeration of comparison types for classic-controlled operations.")
      .value("eq", qc::ComparisonKind::Eq)
      .value("neq", qc::ComparisonKind::Neq)
      .value("lt", qc::ComparisonKind::Lt)
      .value("leq", qc::ComparisonKind::Leq)
      .value("gt", qc::ComparisonKind::Gt)
      .value("geq", qc::ComparisonKind::Geq)
      .export_values();

  auto ifElse =
      nb::class_<qc::IfElseOperation, qc::Operation>(m, "IfElseOperation");

  ifElse.def(
      "__init__",
      [](qc::IfElseOperation* self, qc::Operation* thenOp,
         qc::Operation* elseOp, qc::ClassicalRegister& controlReg,
         const std::uint64_t expectedVal, const qc::ComparisonKind kind) {
        std::unique_ptr<qc::Operation> thenPtr =
            thenOp ? thenOp->clone() : nullptr;
        std::unique_ptr<qc::Operation> elsePtr =
            elseOp ? elseOp->clone() : nullptr;
        new (self) qc::IfElseOperation(std::move(thenPtr), std::move(elsePtr),
                                       controlReg, expectedVal, kind);
      },
      "then_operation"_a, "else_operation"_a, "control_register"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq);
  ifElse.def(
      "__init__",
      [](qc::IfElseOperation* self, qc::Operation* thenOp,
         qc::Operation* elseOp, qc::Bit controlBit, std::uint64_t expectedVal,
         qc::ComparisonKind kind) {
        std::unique_ptr<qc::Operation> thenPtr =
            thenOp ? thenOp->clone() : nullptr;
        std::unique_ptr<qc::Operation> elsePtr =
            elseOp ? elseOp->clone() : nullptr;
        new (self) qc::IfElseOperation(std::move(thenPtr), std::move(elsePtr),
                                       controlBit, expectedVal, kind);
      },
      "then_operation"_a, "else_operation"_a, "control_bit"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq);
  ifElse.def_prop_ro("then_operation", &qc::IfElseOperation::getThenOp,
                     nb::rv_policy::reference_internal);
  ifElse.def_prop_ro("else_operation", &qc::IfElseOperation::getElseOp,
                     nb::rv_policy::reference_internal);
  ifElse.def_prop_ro("control_register",
                     &qc::IfElseOperation::getControlRegister);
  ifElse.def_prop_ro("control_bit", &qc::IfElseOperation::getControlBit);
  ifElse.def_prop_ro("expected_value_register",
                     &qc::IfElseOperation::getExpectedValueRegister);
  ifElse.def_prop_ro("expected_value_bit",
                     &qc::IfElseOperation::getExpectedValueBit);
  ifElse.def_prop_ro("comparison_kind",
                     &qc::IfElseOperation::getComparisonKind);
  ifElse.def("__repr__", [](const qc::IfElseOperation& op) {
    std::stringstream ss;
    ss << "IfElseOperation(<...then-op...>, <...else-op...>, ";
    if (const auto& controlReg = op.getControlRegister();
        controlReg.has_value()) {
      ss << "control_register=ClassicalRegister(" << controlReg->getSize()
         << ", " << controlReg->getStartIndex() << ", " << controlReg->getName()
         << "), "
         << "expected_value=" << op.getExpectedValueRegister() << ", ";
    }
    if (const auto& controlBit = op.getControlBit(); controlBit.has_value()) {
      ss << "control_bit=" << controlBit.value() << ", "
         << "expected_value=" << op.getExpectedValueBit() << ", ";
    }
    ss << "comparison_kind='" << op.getComparisonKind() << "')";
    return ss.str();
  });
}

} // namespace mqt
