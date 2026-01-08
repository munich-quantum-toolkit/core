/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
#include <nanobind/stl/optional.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>   // NOLINT(misc-include-cleaner)
#include <sstream>
#include <utility>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerIfElseOperation(const nb::module_& m) {
  nb::enum_<qc::ComparisonKind>(
      m, "ComparisonKind",
      "Enumeration of comparison types for classic-controlled operations.")
      .value("eq", qc::ComparisonKind::Eq, "Equality comparison.")
      .value("neq", qc::ComparisonKind::Neq, "Inequality comparison.")
      .value("lt", qc::ComparisonKind::Lt, "Less-than comparison.")
      .value("leq", qc::ComparisonKind::Leq, "Less-than-or-equal comparison.")
      .value("gt", qc::ComparisonKind::Gt, "Greater-than comparison.")
      .value("geq", qc::ComparisonKind::Geq,
             "Greater-than-or-equal comparison.");

  auto ifElse = nb::class_<qc::IfElseOperation, qc::Operation>(
      m, "IfElseOperation", R"pb(If-else quantum operation.

This class is used to represent an if-else operation.
The then operation is executed if the value of the classical register matches the expected value.
Otherwise, the else operation is executed.

Args:
    then_operation: The operation that is executed if the condition is met.
    else_operation: The operation that is executed if the condition is not met.
    control_register: The classical register that controls the operation.
    expected_value: The expected value of the classical register.
    comparison_kind: The kind of comparison (default is equality).)pb");

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
      "then_operation"_a, nb::arg("else_operation").none(true),
      "control_register"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq);
  ifElse.def(
      "__init__",
      [](qc::IfElseOperation* self, qc::Operation* thenOp,
         qc::Operation* elseOp, qc::Bit controlBit, bool expectedVal,
         qc::ComparisonKind kind) {
        std::unique_ptr<qc::Operation> thenPtr =
            thenOp ? thenOp->clone() : nullptr;
        std::unique_ptr<qc::Operation> elsePtr =
            elseOp ? elseOp->clone() : nullptr;
        new (self) qc::IfElseOperation(std::move(thenPtr), std::move(elsePtr),
                                       controlBit, expectedVal, kind);
      },
      "then_operation"_a, nb::arg("else_operation").none(true), "control_bit"_a,
      "expected_value"_a = true, "comparison_kind"_a = qc::ComparisonKind::Eq);

  ifElse.def_prop_ro("then_operation", &qc::IfElseOperation::getThenOp,
                     nb::rv_policy::reference_internal,
                     "The operation that is executed if the condition is met.");

  ifElse.def_prop_ro(
      "else_operation", &qc::IfElseOperation::getElseOp,
      nb::rv_policy::reference_internal,
      nb::sig("def else_operation(self) -> "
              "mqt.core.ir.operations.Operation | None"),
      "The operation that is executed if the condition is not met.");

  ifElse.def_prop_ro("control_register",
                     &qc::IfElseOperation::getControlRegister,
                     "The classical register that controls the operation.");

  ifElse.def_prop_ro("control_bit", &qc::IfElseOperation::getControlBit,
                     "The classical bit that controls the operation.");

  ifElse.def_prop_ro("expected_value_register",
                     &qc::IfElseOperation::getExpectedValueRegister,
                     R"pb(The expected value of the classical register.

The then-operation is executed if the value of the classical register matches the expected value based on the kind of comparison.
The expected value is an integer that is interpreted as a binary number, where the least significant bit is at the start index of the classical register.)pb");

  ifElse.def_prop_ro("expected_value_bit",
                     &qc::IfElseOperation::getExpectedValueBit,
                     R"pb(The expected value of the classical bit.

The then-operation is executed if the value of the classical bit matches the expected value based on the kind of comparison.)pb");

  ifElse.def_prop_ro("comparison_kind", &qc::IfElseOperation::getComparisonKind,
                     R"pb(The kind of comparison.

The then-operation is executed if the value of the control matches the expected value based on the kind of comparison.)pb");

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
