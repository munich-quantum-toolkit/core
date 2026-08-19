/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QiskitTranslation.h"
#include "mlir/Dialect/QC/Translation/StandardGate.h"

// Qiskit requires its umbrella header before the extension function table.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h> // NOLINT(misc-include-cleaner): enables the std::complex caster.
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner): enables the std::string caster.
#include <qiskit.h>
#include <qiskit/complex.h>
#include <qiskit/funcs_py.h>
#include <qiskit/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef MQT_QISKIT_VERSION_FACTORY
#define MQT_QISKIT_VERSION_FACTORY createQiskit2_5
#endif

#ifndef MQT_QISKIT_VERSION_EXPECTED_MAJOR
#define MQT_QISKIT_VERSION_EXPECTED_MAJOR 2U
#endif

#ifndef MQT_QISKIT_VERSION_EXPECTED_MINOR
#define MQT_QISKIT_VERSION_EXPECTED_MINOR 5U
#endif

#ifndef MQT_QISKIT_VERSION_EXACT_API
#define MQT_QISKIT_VERSION_EXACT_API 0
#endif

#ifndef MQT_QISKIT_VERSION_LABEL
#define MQT_QISKIT_VERSION_LABEL "2.5"
#endif

// Qiskit's generated extension-table macros expand to C function-pointer casts
// and indexed table access at every call site. The headers are vendored
// byte-for-byte, so the diagnostics cannot be fixed there. This source contains
// the complete version-specific native API surface; translation-unit scope is
// the smallest containment that does not duplicate every generated signature.
// NOLINTBEGIN(cppcoreguidelines-pro-type-cstyle-cast)
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

namespace mqt::bindings::qiskit {
namespace nb = nanobind;
namespace {

constexpr size_t MAX_EXPRESSION_DEPTH = 64U;
constexpr size_t MAX_ANNOTATED_OPERATION_DEPTH = 64U;

[[nodiscard]] nb::object pythonAttribute(const nb::handle object,
                                         const char* name,
                                         const std::string_view error) {
  try {
    return nb::borrow<nb::object>(object).attr(name);
  } catch (const nb::python_error&) {
    throw std::runtime_error(std::string(error));
  }
}

[[nodiscard]] std::string pythonText(const nb::handle object,
                                     const std::string_view error) {
  try {
    return nb::cast<std::string>(nb::str(object));
  } catch (const nb::python_error&) {
    throw std::runtime_error(std::string(error));
  }
}

[[nodiscard]] std::string pythonStringAttribute(const nb::handle object,
                                                const char* name,
                                                const std::string_view error) {
  return pythonText(pythonAttribute(object, name, error), error);
}

[[nodiscard]] uint64_t pythonUnsignedAttribute(const nb::handle object,
                                               const char* name,
                                               const std::string_view error) {
  const auto attribute = pythonAttribute(object, name, error);
  uint64_t result = 0;
  if (!nb::try_cast(attribute, result)) {
    throw std::runtime_error(std::string(error));
  }
  return result;
}

[[noreturn]] void throwPythonError(const std::string_view message) {
  const nb::python_error error;
  throw std::runtime_error(std::string(message) + ": " + error.what());
}

[[noreturn]] void throwPythonError(const std::string_view message,
                                   const nb::python_error& error) {
  throw std::runtime_error(std::string(message) + ": " + error.what());
}

void checkExitCode(const QkExitCode code, const std::string_view operation) {
  if (code != QkExitCode_Success) {
    throw std::runtime_error(std::string(operation) +
                             " failed with Qiskit C API exit code " +
                             std::to_string(static_cast<unsigned int>(code)));
  }
}

using ParameterizedGateFunction = QkExitCode (*)(QkCircuit*, QkGate,
                                                 const uint32_t*,
                                                 const QkParam* const*);

QkExitCode addParameterizedGate(QkCircuit* circuit, const QkGate gate,
                                const uint32_t* qubits,
                                const QkParam* const* parameters) {
  // Qiskit 2.5.0's generated macro contains a duplicate `const` that GCC
  // rejects. Keep the vendored snapshot exact and call the same capsule slot
  // through its intended signature instead.
  const auto function =
      reinterpret_cast<ParameterizedGateFunction>(_Qk_API_Circuit[38]);
  return function(circuit, gate, qubits, parameters);
}

[[nodiscard]] OperationKind normalizeKind(const QkOperationKind kind) {
  switch (kind) {
  case QkOperationKind_Gate:
    return OperationKind::Gate;
  case QkOperationKind_Barrier:
    return OperationKind::Barrier;
  case QkOperationKind_Delay:
    return OperationKind::Delay;
  case QkOperationKind_Measure:
    return OperationKind::Measure;
  case QkOperationKind_Reset:
    return OperationKind::Reset;
  case QkOperationKind_Unitary:
    return OperationKind::Unitary;
  case QkOperationKind_ControlFlow:
    return OperationKind::ControlFlow;
  case QkOperationKind_PauliProductMeasurement:
  case QkOperationKind_PauliProductRotation:
  case QkOperationKind_Unknown:
    return OperationKind::Unknown;
  }
  return OperationKind::Unknown;
}

[[nodiscard]] Parameter normalizeParameter(const QkParam* parameter) {
  const auto number = qk_param_as_real(parameter);
  if (std::isfinite(number)) {
    auto* numeric = qk_param_from_double(number);
    if (numeric == nullptr) {
      throwPythonError("Qiskit failed to inspect a numeric parameter");
    }
    const auto isNumber = qk_param_equal(parameter, numeric);
    qk_param_free(numeric);
    if (isNumber) {
      return {.kind = ParameterKind::Number, .number = number};
    }
  }
  throw std::runtime_error(
      "Qiskit's native API does not expose symbolic parameter-expression "
      "structure");
}

[[nodiscard]] Parameter
normalizePythonParameterLeaf(const nb::handle parameter) {
  double number = 0.0;
  if (nb::try_cast(parameter, number)) {
    if (!std::isfinite(number)) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    return {.kind = ParameterKind::Number, .number = number};
  }

  std::complex<double> complexNumber;
  if (nb::try_cast(parameter, complexNumber)) {
    if (!std::isfinite(complexNumber.real()) ||
        !std::isfinite(complexNumber.imag())) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    if (complexNumber.imag() != 0.0) {
      throw std::runtime_error(
          "Qiskit parameter expressions with complex values are not "
          "supported");
    }
    return {.kind = ParameterKind::Number, .number = complexNumber.real()};
  }

  if (!nb::hasattr(parameter, "name") || !nb::hasattr(parameter, "uuid")) {
    throw std::runtime_error(
        "Qiskit parameter expression contains an unsupported operand");
  }
  auto name = pythonStringAttribute(
      parameter, "name", "Qiskit parameter has an invalid symbol name");
  auto identity =
      pythonText(pythonAttribute(parameter, "uuid",
                                 "Qiskit parameter has no stable identity"),
                 "Qiskit parameter has an invalid stable identity");
  if (name.empty()) {
    throw std::runtime_error("Qiskit parameter has an empty symbol name");
  }
  if (name.find('\0') != std::string::npos) {
    throw std::runtime_error(
        "Qiskit parameter names cannot contain null characters");
  }
  if (identity.empty()) {
    throw std::runtime_error("Qiskit parameter has an empty stable identity");
  }
  if (identity.find('\0') != std::string::npos) {
    throw std::runtime_error(
        "Qiskit parameter identities cannot contain null characters");
  }
  Parameter result{.kind = ParameterKind::Symbol,
                   .text = std::move(name),
                   .identity = std::move(identity)};
  const auto vectorElement =
      nb::module_::import_("qiskit.circuit").attr("ParameterVectorElement");
  if (nb::isinstance(parameter, vectorElement)) {
    throw std::runtime_error(
        "Qiskit parameter-vector elements are not supported");
  }
  return result;
}

struct ParsedParameter {
  Parameter value;
  size_t depth = 1U;
};

[[noreturn]] void throwParameterExpressionSizeError() {
  throw std::runtime_error(
      "Qiskit parameter expression exceeds the supported " +
      std::to_string(MAX_PARAMETER_EXPRESSION_NODES) + "-node size");
}

[[noreturn]] void throwParameterExpressionDepthError() {
  throw std::runtime_error(
      "Qiskit parameter expression exceeds the supported " +
      std::to_string(MAX_PARAMETER_EXPRESSION_DEPTH) + "-level nesting depth");
}

void countParameterExpressionNode(size_t& nodeCount) {
  if (nodeCount >= MAX_PARAMETER_EXPRESSION_NODES) {
    throwParameterExpressionSizeError();
  }
  ++nodeCount;
}

[[nodiscard]] ParsedParameter
takeParameterExpressionOperand(const nb::handle operand,
                               std::vector<ParsedParameter>& stack,
                               size_t& nodeCount) {
  if (operand.is_none()) {
    if (stack.empty()) {
      throw std::runtime_error(
          "Qiskit parameter expression replay has too few operands");
    }
    auto result = std::move(stack.back());
    stack.pop_back();
    return result;
  }
  countParameterExpressionNode(nodeCount);
  return {.value = normalizePythonParameterLeaf(operand)};
}

[[nodiscard]] Parameter makeUnaryParameter(const ParameterKind kind,
                                           Parameter operand) {
  return {.kind = kind,
          .left = std::make_shared<const Parameter>(std::move(operand))};
}

[[nodiscard]] Parameter makeBinaryParameter(const ParameterKind kind,
                                            Parameter lhs, Parameter rhs) {
  return {.kind = kind,
          .left = std::make_shared<const Parameter>(std::move(lhs)),
          .right = std::make_shared<const Parameter>(std::move(rhs))};
}

[[nodiscard]] std::string parameterOpcode(const nb::handle replayEntry) {
  auto opcode = pythonText(
      pythonAttribute(replayEntry, "op",
                      "Qiskit parameter replay entry has no operation"),
      "Qiskit parameter replay entry has an invalid operation");
  constexpr std::string_view prefix = "OpCode.";
  if (opcode.starts_with(prefix)) {
    opcode.erase(0U, prefix.size());
  }
  return opcode;
}

[[nodiscard]] bool isUnaryParameterOpcode(const std::string_view opcode) {
  return opcode == "NEG" || opcode == "SIN" || opcode == "COS" ||
         opcode == "TAN" || opcode == "ASIN" || opcode == "ACOS" ||
         opcode == "ATAN" || opcode == "EXP" || opcode == "LOG" ||
         opcode == "ABS" || opcode == "CONJ" || opcode == "CONJUGATE";
}

[[nodiscard]] ParameterKind unaryParameterKind(const std::string_view opcode) {
  if (opcode == "NEG") {
    return ParameterKind::Negate;
  }
  if (opcode == "SIN") {
    return ParameterKind::Sin;
  }
  if (opcode == "COS") {
    return ParameterKind::Cos;
  }
  if (opcode == "TAN") {
    return ParameterKind::Tan;
  }
  if (opcode == "ASIN") {
    return ParameterKind::ArcSin;
  }
  if (opcode == "ACOS") {
    return ParameterKind::ArcCos;
  }
  if (opcode == "ATAN") {
    return ParameterKind::ArcTan;
  }
  if (opcode == "EXP") {
    return ParameterKind::Exp;
  }
  if (opcode == "LOG") {
    return ParameterKind::Log;
  }
  if (opcode == "ABS") {
    return ParameterKind::Abs;
  }
  return ParameterKind::Conjugate;
}

[[nodiscard]] bool isBinaryParameterOpcode(const std::string_view opcode) {
  return opcode == "ADD" || opcode == "SUB" || opcode == "MUL" ||
         opcode == "DIV" || opcode == "POW" || opcode == "RSUB" ||
         opcode == "RDIV" || opcode == "RPOW";
}

[[nodiscard]] ParameterKind binaryParameterKind(const std::string_view opcode) {
  if (opcode == "ADD") {
    return ParameterKind::Add;
  }
  if (opcode == "SUB" || opcode == "RSUB") {
    return ParameterKind::Subtract;
  }
  if (opcode == "MUL") {
    return ParameterKind::Multiply;
  }
  if (opcode == "DIV" || opcode == "RDIV") {
    return ParameterKind::Divide;
  }
  return ParameterKind::Power;
}

[[nodiscard]] Parameter normalizePythonParameter(const nb::handle parameter) {
  if (nb::hasattr(parameter, "name") && nb::hasattr(parameter, "uuid")) {
    return normalizePythonParameterLeaf(parameter);
  }

  bool hasTrackedSymbols = false;
  if (nb::hasattr(parameter, "parameters")) {
    const auto parameters = pythonAttribute(
        parameter, "parameters",
        "Qiskit parameter expression has no tracked-symbol set");
    try {
      hasTrackedSymbols = nb::len(parameters) != 0U;
    } catch (const nb::python_error& error) {
      throwPythonError(
          "Qiskit parameter expression tracked-symbol set is not sized", error);
    }
  }
  if (!hasTrackedSymbols) {
    return normalizePythonParameterLeaf(parameter);
  }

  const auto replay = pythonAttribute(
      parameter, "_qpy_replay",
      "Qiskit parameter expression does not expose its operation replay");
  size_t replaySize = 0U;
  try {
    replaySize = nb::len(replay);
  } catch (const nb::python_error& error) {
    throwPythonError("Qiskit parameter expression replay is not sized", error);
  }
  if (replaySize == 0U) {
    throw std::runtime_error("Qiskit parameter expression replay is empty");
  }
  if (replaySize > MAX_PARAMETER_EXPRESSION_NODES) {
    throwParameterExpressionSizeError();
  }

  size_t nodeCount = 0U;
  std::vector<ParsedParameter> stack;
  stack.reserve(replaySize);
  try {
    for (const nb::handle replayEntry : nb::iter(replay)) {
      const auto opcode = parameterOpcode(replayEntry);
      if (opcode == "SIGN" || opcode == "GRAD" || opcode == "SUBSTITUTE") {
        throw std::runtime_error("Qiskit parameter expression operation '" +
                                 opcode + "' is not supported");
      }
      const auto lhs =
          pythonAttribute(replayEntry, "lhs",
                          "Qiskit parameter replay entry has no left operand");
      const auto rhs =
          pythonAttribute(replayEntry, "rhs",
                          "Qiskit parameter replay entry has no right operand");
      if (isUnaryParameterOpcode(opcode)) {
        if (!rhs.is_none()) {
          throw std::runtime_error(
              "Qiskit unary parameter replay entry has a right operand");
        }
        auto operand = takeParameterExpressionOperand(lhs, stack, nodeCount);
        countParameterExpressionNode(nodeCount);
        ++operand.depth;
        if (operand.depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
          throwParameterExpressionDepthError();
        }
        operand.value = makeUnaryParameter(unaryParameterKind(opcode),
                                           std::move(operand.value));
        stack.push_back(std::move(operand));
        continue;
      }
      if (!isBinaryParameterOpcode(opcode)) {
        throw std::runtime_error("Qiskit parameter expression operation '" +
                                 opcode + "' is not supported");
      }
      auto right = takeParameterExpressionOperand(rhs, stack, nodeCount);
      auto left = takeParameterExpressionOperand(lhs, stack, nodeCount);
      if (opcode == "RSUB" || opcode == "RDIV" || opcode == "RPOW") {
        std::swap(left, right);
      }
      countParameterExpressionNode(nodeCount);
      const auto depth = std::max(left.depth, right.depth) + 1U;
      if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
        throwParameterExpressionDepthError();
      }
      stack.push_back({.value = makeBinaryParameter(binaryParameterKind(opcode),
                                                    std::move(left.value),
                                                    std::move(right.value)),
                       .depth = depth});
    }
  } catch (const nb::python_error& error) {
    throwPythonError("Qiskit parameter expression replay is not iterable",
                     error);
  }
  if (stack.size() != 1U) {
    throw std::runtime_error(
        "Qiskit parameter expression replay leaves multiple results");
  }
  return std::move(stack.back().value);
}

void appendControlModifier(const nb::handle object,
                           std::vector<GateModifier>& modifiers) {
  const auto controls = pythonUnsignedAttribute(
      object, "num_ctrl_qubits",
      "Qiskit control modifier has an invalid control count");
  if (controls == 0U ||
      controls > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) ||
      controls > std::numeric_limits<uint64_t>::digits) {
    throw std::runtime_error(
        "Qiskit control modifiers require between 1 and 64 controls");
  }
  const auto state = pythonUnsignedAttribute(
      object, "ctrl_state", "Qiskit control modifier has an invalid state");
  const auto closedState = controls == std::numeric_limits<uint64_t>::digits
                               ? std::numeric_limits<uint64_t>::max()
                               : (uint64_t{1} << controls) - 1U;
  if (state != closedState) {
    throw std::runtime_error(
        "Qiskit circuit import does not support open-control modifiers");
  }
  modifiers.push_back({.kind = GateModifierKind::Control,
                       .numControls = static_cast<uint32_t>(controls)});
}

[[nodiscard]] nb::object terminalPythonGate(const nb::handle operation,
                                            const size_t depth = 0U) {
  if (depth >= MAX_ANNOTATED_OPERATION_DEPTH) {
    throw std::runtime_error(
        "Qiskit annotated operations exceed the nesting limit of 64");
  }
  if (nb::hasattr(operation, "base_op")) {
    return terminalPythonGate(
        pythonAttribute(operation, "base_op",
                        "Qiskit annotated operation has no base"),
        depth + 1U);
  }
  if (nb::hasattr(operation, "base_gate")) {
    return terminalPythonGate(
        pythonAttribute(operation, "base_gate",
                        "Qiskit controlled gate has no base"),
        depth + 1U);
  }
  return nb::borrow<nb::object>(operation);
}

[[nodiscard]] bool isPythonUnitaryGate(const nb::handle operation) {
  const auto terminal = terminalPythonGate(operation);
  const auto unitaryGate =
      nb::module_::import_("qiskit.circuit.library").attr("UnitaryGate");
  return nb::isinstance(terminal, unitaryGate);
}

void normalizePythonModifier(const nb::handle modifier,
                             std::vector<GateModifier>& modifiers) {
  const auto type = pythonAttribute(modifier, "__class__",
                                    "Qiskit modifier does not expose its type");
  const auto name = pythonStringAttribute(
      type, "__name__", "Qiskit modifier has an invalid type name");
  if (name == "InverseModifier") {
    modifiers.push_back({.kind = GateModifierKind::Inverse});
    return;
  }
  if (name == "ControlModifier") {
    appendControlModifier(modifier, modifiers);
    return;
  }
  if (name == "PowerModifier") {
    auto power = pythonAttribute(modifier, "power",
                                 "Qiskit power modifier has no exponent");
    modifiers.push_back({.kind = GateModifierKind::Power,
                         .exponent = normalizePythonParameter(power)});
    return;
  }
  throw std::runtime_error("unsupported Qiskit operation modifier '" + name +
                           "'");
}

void normalizePythonGate(const nb::handle operation, Instruction& result,
                         const size_t depth = 0U) {
  if (depth >= MAX_ANNOTATED_OPERATION_DEPTH) {
    throw std::runtime_error(
        "Qiskit annotated operations exceed the nesting limit of 64");
  }
  if (nb::hasattr(operation, "base_op")) {
    const auto base = pythonAttribute(operation, "base_op",
                                      "Qiskit annotated operation has no base");
    normalizePythonGate(base, result, depth + 1U);
    const auto modifiers = pythonAttribute(
        operation, "modifiers", "Qiskit annotated operation has no modifiers");
    try {
      for (const nb::handle modifier : nb::iter(modifiers)) {
        normalizePythonModifier(modifier, result.modifiers);
      }
    } catch (const nb::python_error& error) {
      throwPythonError("Qiskit operation modifiers are not iterable", error);
    }
    return;
  }

  if (nb::hasattr(operation, "base_gate")) {
    const auto name = pythonStringAttribute(
        operation, "name", "Qiskit controlled gate has an invalid name");
    if (name == "cu") {
      // CU's fourth parameter is a phase on its controlled U decomposition;
      // flattening it to U plus a generic control would lose that parameter.
      result.name = name;
      return;
    }
    const auto base = pythonAttribute(operation, "base_gate",
                                      "Qiskit controlled gate has no base");
    normalizePythonGate(base, result, depth + 1U);
    appendControlModifier(operation, result.modifiers);
    return;
  }

  result.name = pythonStringAttribute(operation, "name",
                                      "Qiskit operation has an invalid name");
}

[[nodiscard]] ClassicalType normalizeType(const QkExprTypeInfo type) {
  switch (type.ty) {
  case QkExprType_Bool:
    return ClassicalType::Bool;
  case QkExprType_Uint:
    if (type.width == 0U || type.width > 64U) {
      throw std::runtime_error("Qiskit unsigned classical values wider than 64 "
                               "bits are not supported");
    }
    return ClassicalType::Uint;
  case QkExprType_Float:
    return ClassicalType::Float;
  case QkExprType_Duration:
    throw std::runtime_error(
        "Qiskit circuit import does not support duration expressions");
  }
  throw std::runtime_error(
      "Qiskit returned an unknown classical expression type");
}

void setType(Expression& result, const QkExprTypeInfo type) {
  result.type = normalizeType(type);
  if (result.type == ClassicalType::Bool) {
    result.width = 1U;
  } else if (result.type == ClassicalType::Float) {
    result.width = 64U;
  } else {
    result.width = static_cast<uint32_t>(type.width);
  }
}

[[nodiscard]] BinaryOperation
normalizeBinaryOperation(const QkBinaryOpType op) {
  switch (op) {
  case QkBinaryOpType_BitAnd:
    return BinaryOperation::BitAnd;
  case QkBinaryOpType_BitOr:
    return BinaryOperation::BitOr;
  case QkBinaryOpType_BitXor:
    return BinaryOperation::BitXor;
  case QkBinaryOpType_LogicAnd:
    return BinaryOperation::LogicAnd;
  case QkBinaryOpType_LogicOr:
    return BinaryOperation::LogicOr;
  case QkBinaryOpType_Equal:
    return BinaryOperation::Equal;
  case QkBinaryOpType_NotEqual:
    return BinaryOperation::NotEqual;
  case QkBinaryOpType_Less:
    return BinaryOperation::Less;
  case QkBinaryOpType_LessEqual:
    return BinaryOperation::LessEqual;
  case QkBinaryOpType_Greater:
    return BinaryOperation::Greater;
  case QkBinaryOpType_GreaterEqual:
    return BinaryOperation::GreaterEqual;
  case QkBinaryOpType_ShiftLeft:
    return BinaryOperation::ShiftLeft;
  case QkBinaryOpType_ShiftRight:
    return BinaryOperation::ShiftRight;
  case QkBinaryOpType_Add:
    return BinaryOperation::Add;
  case QkBinaryOpType_Sub:
    return BinaryOperation::Subtract;
  case QkBinaryOpType_Mul:
    return BinaryOperation::Multiply;
  case QkBinaryOpType_Div:
    return BinaryOperation::Divide;
  }
  throw std::runtime_error(
      "Qiskit returned an unknown binary expression operation");
}

[[nodiscard]] UnaryOperation normalizeUnaryOperation(const QkUnaryOpType op) {
  switch (op) {
  case QkUnaryOpType_BitNot:
    return UnaryOperation::BitNot;
  case QkUnaryOpType_LogicNot:
    return UnaryOperation::LogicNot;
  case QkUnaryOpType_Negate:
    return UnaryOperation::Negate;
  }
  throw std::runtime_error(
      "Qiskit returned an unknown unary expression operation");
}

[[nodiscard]] std::unique_ptr<Expression>
normalizeExpression(const QkExprNode* expression, const size_t depth = 0U) {
  if (expression == nullptr) {
    throw std::runtime_error("Qiskit returned a null classical expression");
  }
  if (depth >= MAX_EXPRESSION_DEPTH) {
    throw std::runtime_error(
        "Qiskit classical expressions exceed the nesting limit of 64");
  }
  auto result = std::make_unique<Expression>();
  switch (qk_expr_kind(expression)) {
  case QkExprNodeKind_Binary: {
    const auto info = qk_expr_binary_info(expression);
    result->kind = ExpressionKind::Binary;
    result->binaryOperation = normalizeBinaryOperation(info.op);
    setType(*result, info.ty);
    result->left = normalizeExpression(info.left, depth + 1U);
    result->right = normalizeExpression(info.right, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Unary: {
    const auto info = qk_expr_unary_info(expression);
    result->kind = ExpressionKind::Unary;
    result->unaryOperation = normalizeUnaryOperation(info.op);
    setType(*result, info.ty);
    result->left = normalizeExpression(info.operand, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Cast: {
    const auto info = qk_expr_cast_info(expression);
    result->kind = ExpressionKind::Cast;
    setType(*result, info.ty);
    result->left = normalizeExpression(info.operand, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Index: {
    const auto info = qk_expr_index_info(expression);
    result->kind = ExpressionKind::Index;
    setType(*result, info.ty);
    result->left = normalizeExpression(info.target, depth + 1U);
    result->right = normalizeExpression(info.index, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Value: {
    const auto* value = qk_expr_as_value(expression);
    const auto type = qk_value_type_info(value);
    result->kind = ExpressionKind::Value;
    setType(*result, type);
    switch (result->type) {
    case ClassicalType::Bool:
      result->boolValue = qk_value_bool(value);
      break;
    case ClassicalType::Uint:
      result->uintValue = qk_value_uint(value);
      break;
    case ClassicalType::Float:
      result->floatValue = qk_value_float(value);
      if (!std::isfinite(result->floatValue)) {
        throw std::runtime_error(
            "Qiskit classical floating-point literals must be finite");
      }
      break;
    }
    return result;
  }
  case QkExprNodeKind_Var:
    throw std::runtime_error(
        "Qiskit circuit import does not support variables in classical "
        "expressions");
  case QkExprNodeKind_Stretch:
    throw std::runtime_error(
        "Qiskit circuit import does not support stretch expressions");
  }
  throw std::runtime_error(
      "Qiskit returned an unknown classical expression node");
}

[[nodiscard]] Register normalizeRegister(const QkClassicalRegister* reg,
                                         const QkCircuit* rootCircuit) {
  // qk_str_free requires the mutable allocation returned by Qiskit.
  // NOLINTNEXTLINE(misc-const-correctness)
  char* const name = qk_classical_register_name(reg);
  if (name == nullptr) {
    throwPythonError("Qiskit failed to read a classical-register name");
  }
  Register result{.name = name};
  qk_str_free(name);
  result.bits.resize(qk_classical_register_num_bits(reg));
  if (!result.bits.empty()) {
    qk_classical_register_circuit_bits(reg, rootCircuit, result.bits.data());
  }
  return result;
}

class OwnedParameter final {
public:
  OwnedParameter() : value_(qk_param_zero()) {
    if (value_ == nullptr) {
      throwPythonError("Qiskit failed to allocate a parameter expression");
    }
  }

  explicit OwnedParameter(const double value) {
    if (!std::isfinite(value)) {
      throw std::runtime_error(
          "cannot construct a non-finite Qiskit parameter");
    }
    value_ = qk_param_from_double(value);
    if (value_ == nullptr) {
      throwPythonError("Qiskit failed to construct a circuit parameter");
    }
  }

  explicit OwnedParameter(const std::string_view name) {
    if (name.empty()) {
      throw std::runtime_error(
          "cannot construct a Qiskit parameter with an empty name");
    }
    value_ = qk_param_new_symbol(std::string(name).c_str());
    if (value_ == nullptr) {
      throwPythonError("Qiskit failed to construct a symbolic parameter");
    }
  }

  OwnedParameter(const OwnedParameter&) = delete;
  OwnedParameter& operator=(const OwnedParameter&) = delete;
  OwnedParameter(OwnedParameter&&) = delete;
  OwnedParameter& operator=(OwnedParameter&&) = delete;
  ~OwnedParameter() { qk_param_free(value_); }

  [[nodiscard]] const QkParam* get() const { return value_; }
  [[nodiscard]] QkParam* getMutable() { return value_; }

private:
  QkParam* value_ = nullptr;
};

struct VersionGate {
  constexpr VersionGate(const std::string_view name, const QkGate native,
                        const StandardGateMapping translation)
      : name(name), native(native), translation(translation) {}

  std::string_view name;
  QkGate native;
  StandardGateMapping translation;
};

[[nodiscard]] const auto& gateMap() {
  using Gate = mlir::qc::StandardGate;
  static const std::array GATES{
      VersionGate{"h", QkGate_H, {Gate::H, 0}},
      VersionGate{"id", QkGate_I, {Gate::Id, 0}},
      VersionGate{"x", QkGate_X, {Gate::X, 0}},
      VersionGate{"y", QkGate_Y, {Gate::Y, 0}},
      VersionGate{"z", QkGate_Z, {Gate::Z, 0}},
      VersionGate{"p", QkGate_Phase, {Gate::P, 0}},
      VersionGate{"r", QkGate_R, {Gate::R, 0}},
      VersionGate{"rx", QkGate_RX, {Gate::RX, 0}},
      VersionGate{"ry", QkGate_RY, {Gate::RY, 0}},
      VersionGate{"rz", QkGate_RZ, {Gate::RZ, 0}},
      VersionGate{"s", QkGate_S, {Gate::S, 0}},
      VersionGate{"sdg", QkGate_Sdg, {Gate::Sdg, 0}},
      VersionGate{"sx", QkGate_SX, {Gate::SX, 0}},
      VersionGate{"sxdg", QkGate_SXdg, {Gate::SXdg, 0}},
      VersionGate{"t", QkGate_T, {Gate::T, 0}},
      VersionGate{"tdg", QkGate_Tdg, {Gate::Tdg, 0}},
      VersionGate{"u", QkGate_U, {Gate::U3, 0}},
      VersionGate{"u1", QkGate_U1, {Gate::P, 0}},
      VersionGate{"u2", QkGate_U2, {Gate::U2, 0}},
      VersionGate{"u3", QkGate_U3, {Gate::U3, 0}},
      VersionGate{"ch", QkGate_CH, {Gate::H, 1}},
      VersionGate{"cx", QkGate_CX, {Gate::X, 1}},
      VersionGate{"cy", QkGate_CY, {Gate::Y, 1}},
      VersionGate{"cz", QkGate_CZ, {Gate::Z, 1}},
      VersionGate{"dcx", QkGate_DCX, {Gate::DCX, 0}},
      VersionGate{"ecr", QkGate_ECR, {Gate::ECR, 0}},
      VersionGate{"swap", QkGate_Swap, {Gate::SWAP, 0}},
      VersionGate{"iswap", QkGate_ISwap, {Gate::ISWAP, 0}},
      VersionGate{"cp", QkGate_CPhase, {Gate::P, 1}},
      VersionGate{"crx", QkGate_CRX, {Gate::RX, 1}},
      VersionGate{"cry", QkGate_CRY, {Gate::RY, 1}},
      VersionGate{"crz", QkGate_CRZ, {Gate::RZ, 1}},
      VersionGate{"cs", QkGate_CS, {Gate::S, 1}},
      VersionGate{"csdg", QkGate_CSdg, {Gate::Sdg, 1}},
      VersionGate{"csx", QkGate_CSX, {Gate::SX, 1}},
      VersionGate{"cu", QkGate_CU, {Gate::CU, 0}},
      VersionGate{"cu1", QkGate_CU1, {Gate::P, 1}},
      VersionGate{"cu3", QkGate_CU3, {Gate::U3, 1}},
      VersionGate{"rxx", QkGate_RXX, {Gate::RXX, 0}},
      VersionGate{"ryy", QkGate_RYY, {Gate::RYY, 0}},
      VersionGate{"rzz", QkGate_RZZ, {Gate::RZZ, 0}},
      VersionGate{"rzx", QkGate_RZX, {Gate::RZX, 0}},
      VersionGate{"xx_minus_yy", QkGate_XXMinusYY, {Gate::XXMinusYY, 0}},
      VersionGate{"xx_plus_yy", QkGate_XXPlusYY, {Gate::XXPlusYY, 0}},
      VersionGate{"ccx", QkGate_CCX, {Gate::X, 2}},
      VersionGate{"ccz", QkGate_CCZ, {Gate::Z, 2}},
      VersionGate{"cswap", QkGate_CSwap, {Gate::SWAP, 1}},
      VersionGate{"rccx", QkGate_RCCX, {Gate::RCCX, 0}},
      VersionGate{"mcx", QkGate_C3X, {Gate::X, 3}},
      VersionGate{"c3sx", QkGate_C3SX, {Gate::SX, 3}},
  };
  return GATES;
}

[[nodiscard]] const VersionGate* versionGate(const std::string_view name) {
  for (const auto& gate : gateMap()) {
    if (gate.name == name) {
      return &gate;
    }
  }
  return nullptr;
}

[[nodiscard]] const VersionGate*
versionGate(const StandardGateMapping mapping) {
  for (const auto& gate : gateMap()) {
    if (gate.translation == mapping) {
      return &gate;
    }
  }
  return nullptr;
}

[[nodiscard]] std::optional<StandardGateMapping>
standardGateMapping(const std::string_view name) {
  const auto* gate = versionGate(name);
  return gate == nullptr ? std::nullopt : std::optional{gate->translation};
}

class NativeControlFlowReader;

class NativeCircuitReader final : public CircuitReader {
public:
  explicit NativeCircuitReader(const nb::handle circuit)
      : pythonCircuit_(nb::borrow<nb::object>(circuit)),
        data_(pythonAttribute(
            circuit, "_data",
            "expected a Qiskit QuantumCircuit with native CircuitData")),
        circuit_(qk_circuit_borrow_from_python(data_.ptr())) {
    if (circuit_ == nullptr) {
      throwPythonError("Qiskit rejected QuantumCircuit._data");
    }
    rootCircuit_ = circuit_;
  }

  NativeCircuitReader(nb::object pythonCircuit, const QkCircuit* circuit,
                      const QkCircuit* rootCircuit,
                      const QkControlFlowInstruction* parent)
      : pythonCircuit_(std::move(pythonCircuit)),
        data_(pythonAttribute(
            pythonCircuit_, "_data",
            "Qiskit control-flow block has no native CircuitData")),
        circuit_(circuit), rootCircuit_(rootCircuit), parent_(parent) {}

  [[nodiscard]] uint32_t numQubits() const override {
    return qk_circuit_num_qubits(circuit_);
  }
  [[nodiscard]] uint32_t numClbits() const override {
    return qk_circuit_num_clbits(circuit_);
  }
  [[nodiscard]] size_t numInstructions() const override {
    return qk_circuit_num_instructions(circuit_);
  }
  [[nodiscard]] size_t numQuantumRegisters() const override {
    return qk_circuit_num_quantum_registers(circuit_);
  }
  [[nodiscard]] size_t numClassicalRegisters() const override {
    return qk_circuit_num_classical_registers(circuit_);
  }
  [[nodiscard]] bool hasClassicalVariables() const override {
    return pythonUnsignedAttribute(
               pythonCircuit_, "num_vars",
               "Qiskit circuit has an invalid classical-variable count") != 0U;
  }

  [[nodiscard]] Register quantumRegister(const size_t index) const override {
    const auto* reg = qk_circuit_get_quantum_register(circuit_, index);
    // qk_str_free requires the mutable allocation returned by Qiskit.
    // NOLINTNEXTLINE(misc-const-correctness)
    char* const name = qk_quantum_register_name(reg);
    if (name == nullptr) {
      throwPythonError("Qiskit failed to read a quantum-register name");
    }
    Register result{.name = name};
    qk_str_free(name);
    result.bits.resize(qk_quantum_register_num_bits(reg));
    if (!result.bits.empty()) {
      qk_quantum_register_circuit_bits(reg, circuit_, result.bits.data());
    }
    return result;
  }

  [[nodiscard]] Register classicalRegister(const size_t index) const override {
    const auto* reg = qk_circuit_get_classical_register(circuit_, index);
    // qk_str_free requires the mutable allocation returned by Qiskit.
    // NOLINTNEXTLINE(misc-const-correctness)
    char* const name = qk_classical_register_name(reg);
    if (name == nullptr) {
      throwPythonError("Qiskit failed to read a classical-register name");
    }
    Register result{.name = name};
    qk_str_free(name);
    result.bits.resize(qk_classical_register_num_bits(reg));
    if (!result.bits.empty()) {
      qk_classical_register_circuit_bits(reg, circuit_, result.bits.data());
    }
    return result;
  }

  [[nodiscard]] std::vector<Parameter> parameters() const override {
    std::vector<Parameter> result;
    const auto parameters =
        pythonAttribute(pythonCircuit_, "parameters",
                        "Qiskit circuit does not expose its free parameters");
    try {
      result.reserve(nb::len(parameters));
      for (const nb::handle parameter : nb::iter(parameters)) {
        result.push_back(normalizePythonParameter(parameter));
      }
    } catch (const nb::python_error& error) {
      throwPythonError("Qiskit circuit parameters are not iterable", error);
    }
    return result;
  }

  [[nodiscard]] Parameter globalPhase() const override {
    return normalizePythonParameter(
        pythonAttribute(pythonCircuit_, "global_phase",
                        "Qiskit circuit does not expose its global phase"));
  }

  [[nodiscard]] Instruction instruction(const size_t index) const override {
    const auto kind =
        normalizeKind(qk_circuit_instruction_kind(circuit_, index));
    if (kind == OperationKind::Delay) {
      return {.kind = kind, .name = "delay"};
    }
    if (kind == OperationKind::ControlFlow) {
      return {.kind = kind, .name = "control_flow"};
    }
    std::optional<Instruction> normalizedUnknown;
    if (kind == OperationKind::Unknown) {
      const auto operation = pythonOperation(index);
      if (isPythonUnitaryGate(operation)) {
        Instruction result{.kind = OperationKind::Unitary, .name = "unitary"};
        normalizePythonGate(operation, result);
        result.name = "unitary";
        result.qubits = pythonInstructionQubits(index);
        return result;
      }
      normalizedUnknown.emplace();
      normalizePythonGate(operation, *normalizedUnknown);
    }
    QkCircuitInstruction native{};
    qk_circuit_get_instruction(circuit_, index, &native);
    struct InstructionGuard {
      QkCircuitInstruction* instruction;
      ~InstructionGuard() { qk_circuit_instruction_clear(instruction); }
    };
    const InstructionGuard guard{&native};
    Instruction result;
    result.kind = kind;
    result.name = native.name == nullptr ? "" : native.name;
    if (native.num_qubits != 0U) {
      result.qubits.resize(native.num_qubits);
      std::copy_n(native.qubits, native.num_qubits, result.qubits.begin());
    }
    if (native.num_clbits != 0U) {
      result.clbits.resize(native.num_clbits);
      std::copy_n(native.clbits, native.num_clbits, result.clbits.begin());
    }
    result.parameters.reserve(native.num_params);
    if (result.kind == OperationKind::Gate ||
        result.kind == OperationKind::Unknown) {
      const auto parameters =
          pythonAttribute(pythonOperation(index), "params",
                          "Qiskit operation does not expose its parameters");
      try {
        for (const nb::handle parameter : nb::iter(parameters)) {
          result.parameters.push_back(normalizePythonParameter(parameter));
        }
      } catch (const nb::python_error& error) {
        throwPythonError("Qiskit operation parameters are not iterable", error);
      }
      if (result.parameters.size() != native.num_params) {
        throw std::runtime_error(
            "Qiskit Python and native parameter counts do not match");
      }
    } else {
      for (const auto* parameter :
           std::span(native.params, static_cast<size_t>(native.num_params))) {
        result.parameters.emplace_back(normalizeParameter(parameter));
      }
    }
    if (result.kind == OperationKind::Unknown) {
      result.name = std::move(normalizedUnknown->name);
      result.modifiers = std::move(normalizedUnknown->modifiers);
      if (!result.modifiers.empty()) {
        result.kind = OperationKind::Gate;
      }
    }
    result.standardGate = standardGateMapping(result.name);
    return result;
  }

  [[nodiscard]] std::vector<std::complex<double>>
  unitary(const size_t index) const override {
    const auto instructionData = instruction(index);
    if (instructionData.kind != OperationKind::Unitary) {
      throw std::runtime_error(
          "requested unitary data for a non-unitary instruction");
    }
    const auto nativeKind =
        normalizeKind(qk_circuit_instruction_kind(circuit_, index));
    if (nativeKind == OperationKind::Unknown) {
      size_t numControls = 0U;
      for (const auto& modifier : instructionData.modifiers) {
        if (modifier.kind == GateModifierKind::Control) {
          if (modifier.numControls >
              std::numeric_limits<size_t>::max() - numControls) {
            throw std::runtime_error("Qiskit control count is too large");
          }
          numControls += modifier.numControls;
        }
      }
      if (numControls >= instructionData.qubits.size()) {
        throw std::runtime_error(
            "Qiskit unitary instruction has an unsupported operand arity");
      }
      const auto numTargets = instructionData.qubits.size() - numControls;
      if (numTargets >= std::numeric_limits<size_t>::digits / 2U) {
        throw std::runtime_error(
            "Qiskit unitary is too large to represent safely");
      }
      const auto expectedDimension = size_t{1} << numTargets;
      using Matrix =
          nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<2>>;
      try {
        const auto terminal = terminalPythonGate(pythonOperation(index));
        const auto matrixObject =
            pythonAttribute(terminal, "to_matrix",
                            "Qiskit unitary does not expose its matrix")();
        const auto matrix = nb::cast<Matrix>(matrixObject);
        if (matrix.shape(0) != expectedDimension ||
            matrix.shape(1) != expectedDimension) {
          throw std::runtime_error(
              "Qiskit unitary matrix has an invalid dimension");
        }
        std::vector<std::complex<double>> result;
        result.reserve(expectedDimension * expectedDimension);
        for (size_t row = 0U; row < expectedDimension; ++row) {
          for (size_t column = 0U; column < expectedDimension; ++column) {
            result.push_back(matrix(row, column));
          }
        }
        return result;
      } catch (const nb::python_error& error) {
        throwPythonError("Qiskit failed to read a wrapped unitary matrix",
                         error);
      }
    }
    if (instructionData.qubits.size() >=
        std::numeric_limits<size_t>::digits / 2U) {
      throw std::runtime_error(
          "Qiskit unitary is too large to represent safely");
    }
    const auto entries = size_t{1} << (2U * instructionData.qubits.size());
    std::vector<QkComplex64> native(entries);
    qk_circuit_inst_unitary(
        // Qiskit's read-only accessor is not const-correct in version 2.5.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<QkCircuit*>(circuit_), index, native.data());
    std::vector<std::complex<double>> result;
    result.reserve(entries);
    for (const auto value : native) {
      result.emplace_back(value.re, value.im);
    }
    return result;
  }

  [[nodiscard]] std::unique_ptr<ControlFlowReader>
  controlFlow(size_t index) const override;

  [[nodiscard]] std::unique_ptr<CircuitReader>
  definition(const size_t index) const override {
    const auto operation = pythonOperation(index);
    const auto definition = pythonAttribute(
        operation, "definition",
        "Qiskit instruction does not expose a circuit definition");
    if (definition.is_none()) {
      throw std::runtime_error("Qiskit instruction '" +
                               instruction(index).name +
                               "' has no circuit definition");
    }
    return std::make_unique<NativeCircuitReader>(definition);
  }

  [[nodiscard]] uintptr_t
  definitionIdentity(const size_t index) const override {
    const auto definition = pythonAttribute(
        pythonOperation(index), "definition",
        "Qiskit instruction does not expose a circuit definition");
    if (definition.is_none()) {
      return 0U;
    }
    return reinterpret_cast<uintptr_t>(definition.ptr());
  }

private:
  [[nodiscard]] std::vector<uint32_t>
  pythonInstructionQubits(const size_t index) const {
    std::vector<uint32_t> result;
    try {
      const auto qubits =
          pythonAttribute(data_[index], "qubits",
                          "Qiskit circuit instruction has no qubit operands");
      result.reserve(nb::len(qubits));
      const auto findBit =
          pythonAttribute(pythonCircuit_, "find_bit",
                          "Qiskit circuit cannot resolve instruction qubits");
      for (const nb::handle qubit : nb::iter(qubits)) {
        const auto location = findBit(qubit);
        const auto position = pythonUnsignedAttribute(
            location, "index", "Qiskit qubit has an invalid circuit index");
        if (position > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error("Qiskit qubit index cannot be represented");
        }
        result.push_back(static_cast<uint32_t>(position));
      }
    } catch (const nb::python_error& error) {
      throwPythonError("Qiskit failed to resolve unitary qubits", error);
    }
    return result;
  }

  [[nodiscard]] nb::object pythonOperation(const size_t index) const {
    if (index >= nb::len(data_)) {
      throw std::runtime_error("Qiskit instruction index is out of bounds");
    }
    return pythonAttribute(data_[index], "operation",
                           "Qiskit circuit instruction has no operation");
  }

  nb::object pythonCircuit_;
  nb::object data_;
  const QkCircuit* circuit_ = nullptr;
  const QkCircuit* rootCircuit_ = circuit_;
  const QkControlFlowInstruction* parent_ = nullptr;
};

class NativeControlFlowReader final : public ControlFlowReader {
public:
  NativeControlFlowReader(const QkCircuit* rootCircuit,
                          const QkCircuit* circuit, const size_t index,
                          const QkControlFlowInstruction* parent,
                          nb::object operation)
      : rootCircuit_(rootCircuit),
        controlFlow_(
            qk_circuit_get_control_flow_instruction(circuit, index, parent)),
        operation_(std::move(operation)) {
    if (controlFlow_ == nullptr) {
      throwPythonError("Qiskit failed to inspect a control-flow instruction");
    }
  }

  ~NativeControlFlowReader() override {
    qk_control_flow_instruction_free(controlFlow_);
  }

  [[nodiscard]] ControlFlowKind kind() const override {
    switch (qk_control_flow_kind(controlFlow_)) {
    case QkControlFlowKind_Box:
      return ControlFlowKind::Box;
    case QkControlFlowKind_BreakLoop:
      return ControlFlowKind::Break;
    case QkControlFlowKind_ContinueLoop:
      return ControlFlowKind::Continue;
    case QkControlFlowKind_ForLoop:
      return ControlFlowKind::For;
    case QkControlFlowKind_IfElse:
      return ControlFlowKind::IfElse;
    case QkControlFlowKind_Switch:
      return ControlFlowKind::Switch;
    case QkControlFlowKind_While:
      return ControlFlowKind::While;
    }
    throw std::runtime_error("Qiskit returned an unknown control-flow kind");
  }

  [[nodiscard]] size_t numBlocks() const override {
    return qk_control_flow_num_blocks(controlFlow_);
  }

  [[nodiscard]] std::unique_ptr<CircuitReader>
  block(const size_t index) const override {
    if (index >= numBlocks()) {
      throw std::runtime_error(
          "Qiskit control-flow block index is out of bounds");
    }
    const auto blocks = pythonAttribute(operation_, "blocks",
                                        "Qiskit control flow has no blocks");
    const auto block = nb::borrow<nb::object>(blocks[index]);
    return std::make_unique<NativeCircuitReader>(
        block, qk_control_flow_block_circuit(controlFlow_, index), rootCircuit_,
        controlFlow_);
  }

  [[nodiscard]] std::vector<uint32_t> qubitMap() const override {
    if (numBlocks() == 0U) {
      return {};
    }
    const auto size =
        qk_circuit_num_qubits(qk_control_flow_block_circuit(controlFlow_, 0));
    std::vector<uint32_t> result(size);
    if (!result.empty()) {
      std::copy_n(qk_control_flow_qubit_map(controlFlow_), size,
                  result.begin());
    }
    return result;
  }

  [[nodiscard]] std::vector<uint32_t> clbitMap() const override {
    if (numBlocks() == 0U) {
      return {};
    }
    const auto size =
        qk_circuit_num_clbits(qk_control_flow_block_circuit(controlFlow_, 0));
    std::vector<uint32_t> result(size);
    if (!result.empty()) {
      std::copy_n(qk_control_flow_clbit_map(controlFlow_), size,
                  result.begin());
    }
    return result;
  }

  [[nodiscard]] ClassicalTarget condition() const override {
    ClassicalTarget result;
    switch (qk_control_flow_condition_type(controlFlow_)) {
    case QkConditionType_ClBit: {
      const auto bit = qk_control_flow_condition_bit_info(controlFlow_);
      result.kind = ClassicalTargetKind::ClassicalBit;
      result.bit = static_cast<uint32_t>(bit.clbit);
      result.expectedBit = bit.condition;
      return result;
    }
    case QkConditionType_ClReg: {
      const auto conditionWidth =
          qk_control_flow_condition_reg_cond_bit_width(controlFlow_);
      if (conditionWidth > 64U) {
        throw std::runtime_error(
            "Qiskit register conditions wider than 64 bits are not supported");
      }
      result.kind = ClassicalTargetKind::ClassicalRegister;
      result.reg = normalizeRegister(
          qk_control_flow_condition_reg(controlFlow_), rootCircuit_);
      if (result.reg.bits.empty() || result.reg.bits.size() > 64U) {
        throw std::runtime_error(
            "Qiskit register conditions require between 1 and 64 bits");
      }
      result.width = static_cast<uint32_t>(
          std::max<uint64_t>(conditionWidth, result.reg.bits.size()));
      result.expectedRegister =
          qk_control_flow_condition_reg_cond_uint(controlFlow_);
      return result;
    }
    case QkConditionType_Expr:
      result.kind = ClassicalTargetKind::Expression;
      result.expression =
          normalizeExpression(qk_control_flow_condition_expr(controlFlow_));
      return result;
    }
    throw std::runtime_error("Qiskit returned an unknown condition type");
  }

  [[nodiscard]] Loop loop() const override {
    Loop result;
    switch (qk_control_flow_loop_collection_type(controlFlow_)) {
    case QkLoopCollectionType_Range:
      result.isRange = true;
      qk_control_flow_loop_range(controlFlow_, &result.start, &result.stop,
                                 &result.step);
      break;
    case QkLoopCollectionType_List: {
      result.isRange = false;
      const auto elements = qk_control_flow_loop_elements(controlFlow_);
      if (elements.len != 0U) {
        result.values.resize(elements.len);
        std::copy_n(elements.elements, elements.len, result.values.begin());
      }
      break;
    }
    }
    switch (qk_control_flow_loop_param_kind(controlFlow_)) {
    case QkLoopParamKind_NoLoopParam:
      break;
    case QkLoopParamKind_Parameter: {
      auto symbol = qk_control_flow_loop_symbol_info(controlFlow_);
      if (symbol.ty != QkSymbolType_Standalone) {
        if (symbol.name != nullptr) {
          qk_str_free(symbol.name);
        }
        throw std::runtime_error(
            "Qiskit indexed parameter-vector loop variables are not "
            "supported");
      }
      if (symbol.name == nullptr) {
        throwPythonError("Qiskit failed to read a loop-parameter name");
      }
      const std::string nativeName = symbol.name;
      qk_str_free(symbol.name);
      const auto parameters = pythonAttribute(
          operation_, "params",
          "Qiskit for-loop operation does not expose its parameters");
      try {
        if (nb::len(parameters) < 2U) {
          throw std::runtime_error(
              "Qiskit for-loop operation has no loop parameter");
        }
        auto parameter = normalizePythonParameter(parameters[1]);
        if (parameter.kind != ParameterKind::Symbol) {
          throw std::runtime_error("Qiskit for-loop parameter is not a symbol");
        }
        if (parameter.text != nativeName) {
          throw std::runtime_error(
              "Qiskit Python and native loop-parameter names do not match");
        }
        result.parameter = std::move(parameter);
      } catch (const nb::python_error& error) {
        throwPythonError("Qiskit failed to inspect a loop parameter", error);
      }
      break;
    }
    case QkLoopParamKind_Variable:
      throw std::runtime_error(
          "Qiskit classical-variable loop parameters are not supported");
    }
    return result;
  }

  [[nodiscard]] ClassicalTarget switchTarget() const override {
    ClassicalTarget result;
    switch (qk_control_flow_switch_target_type(controlFlow_)) {
    case QkConditionType_ClBit:
      result.kind = ClassicalTargetKind::ClassicalBit;
      result.bit = qk_control_flow_switch_target_bit(controlFlow_);
      return result;
    case QkConditionType_ClReg:
      result.kind = ClassicalTargetKind::ClassicalRegister;
      result.reg = normalizeRegister(
          qk_control_flow_switch_target_register(controlFlow_), rootCircuit_);
      if (result.reg.bits.empty() || result.reg.bits.size() > 64U) {
        throw std::runtime_error(
            "Qiskit switch registers must contain between 1 and 64 bits");
      }
      result.width = static_cast<uint32_t>(result.reg.bits.size());
      return result;
    case QkConditionType_Expr:
      result.kind = ClassicalTargetKind::Expression;
      result.expression =
          normalizeExpression(qk_control_flow_switch_target_expr(controlFlow_));
      return result;
    }
    throw std::runtime_error("Qiskit returned an unknown switch-target type");
  }

  [[nodiscard]] std::vector<SwitchCase> switchCases() const override {
    std::vector<SwitchCase> result;
    result.reserve(qk_control_flow_switch_num_cases(controlFlow_));
    for (size_t index = 0;
         index < qk_control_flow_switch_num_cases(controlFlow_); ++index) {
      if (qk_control_flow_switch_case_labels_bit_width(controlFlow_, index) >
          64U) {
        throw std::runtime_error(
            "Qiskit switch labels wider than 64 bits are not supported");
      }
      auto native =
          qk_control_flow_switch_case_labels_uint(controlFlow_, index);
      SwitchCase entry{.isDefault = qk_control_flow_switch_is_case_default(
                           controlFlow_, index)};
      if (native.num_labels != 0U) {
        entry.labels.resize(native.num_labels);
        std::copy_n(native.labels, native.num_labels, entry.labels.begin());
      }
      qk_control_flow_switch_case_labels_clear(&native);
      result.push_back(std::move(entry));
    }
    return result;
  }

private:
  const QkCircuit* rootCircuit_ = nullptr;
  QkControlFlowInstruction* controlFlow_ = nullptr;
  nb::object operation_;
};

std::unique_ptr<ControlFlowReader>
NativeCircuitReader::controlFlow(const size_t index) const {
  return std::make_unique<NativeControlFlowReader>(
      rootCircuit_, circuit_, index, parent_, pythonOperation(index));
}

class NativeCircuitWriter final : public CircuitWriter {
public:
  NativeCircuitWriter(const uint32_t looseQubits, const uint32_t looseClbits)
      : circuit_(qk_circuit_new(looseQubits, looseClbits)) {
    if (circuit_ == nullptr) {
      throwPythonError("Qiskit failed to allocate a circuit");
    }
  }

  ~NativeCircuitWriter() override {
    if (circuit_ != nullptr) {
      qk_circuit_free(circuit_);
    }
  }

  void addQuantumRegister(const std::string_view name,
                          const uint32_t size) override {
    auto* reg = qk_quantum_register_new(size, std::string(name).c_str());
    if (reg == nullptr) {
      throwPythonError("Qiskit failed to allocate a quantum register");
    }
    qk_circuit_add_quantum_register(circuit_, reg);
    qk_quantum_register_free(reg);
  }

  void addClassicalRegister(const std::string_view name,
                            const uint32_t size) override {
    auto* reg = qk_classical_register_new(size, std::string(name).c_str());
    if (reg == nullptr) {
      throwPythonError("Qiskit failed to allocate a classical register");
    }
    qk_circuit_add_classical_register(circuit_, reg);
    qk_classical_register_free(reg);
  }

  void setGlobalPhase(const Parameter& phase) override {
    std::vector<std::unique_ptr<OwnedParameter>> ownedParameters;
    const auto* parameter = nativeParameter(phase, ownedParameters);
    checkExitCode(qk_circuit_set_global_phase(circuit_, parameter),
                  "setting global phase");
  }

  void addGate(const StandardGateMapping mapping,
               const std::vector<uint32_t>& qubits,
               const std::vector<Parameter>& parameters) override {
    const auto* gate = versionGate(mapping);
    if (gate == nullptr) {
      const auto& descriptor =
          mlir::qc::getStandardGateDescriptor(mapping.gate);
      throw std::runtime_error("Qiskit " MQT_QISKIT_VERSION_LABEL
                               " output cannot construct standard gate '" +
                               descriptor.operationSymbol.str() + "' with " +
                               std::to_string(mapping.controls) + " controls");
    }
    if (qk_gate_num_qubits(gate->native) != qubits.size() ||
        qk_gate_num_params(gate->native) != parameters.size()) {
      throw std::runtime_error("Qiskit gate '" + std::string(gate->name) +
                               "' has incompatible arity");
    }
    if (parameters.empty()) {
      checkExitCode(
          qk_circuit_gate(circuit_, gate->native, qubits.data(), nullptr),
          "adding gate");
      return;
    }
    std::vector<std::unique_ptr<OwnedParameter>> ownedParameters;
    std::vector<const QkParam*> nativeParameters;
    ownedParameters.reserve(parameters.size());
    nativeParameters.reserve(parameters.size());
    for (const auto& parameter : parameters) {
      nativeParameters.emplace_back(
          nativeParameter(parameter, ownedParameters));
    }
    checkExitCode(addParameterizedGate(circuit_, gate->native, qubits.data(),
                                       nativeParameters.data()),
                  "adding parameterized gate");
  }

  void addMeasure(const uint32_t qubit, const uint32_t clbit) override {
    checkExitCode(qk_circuit_measure(circuit_, qubit, clbit),
                  "adding measurement");
  }

  void addReset(const uint32_t qubit) override {
    checkExitCode(qk_circuit_reset(circuit_, qubit), "adding reset");
  }

  void addBarrier(const std::vector<uint32_t>& qubits) override {
    checkExitCode(qk_circuit_barrier(circuit_, qubits.data(),
                                     static_cast<uint32_t>(qubits.size())),
                  "adding barrier");
  }

  void addUnitary(const std::vector<std::complex<double>>& matrix,
                  const std::vector<uint32_t>& qubits,
                  const uint32_t numControls) override {
    if (numControls >= qubits.size()) {
      throw std::runtime_error("Qiskit unitary has an invalid control count");
    }
    const std::vector targets(qubits.begin() + numControls, qubits.end());
    std::vector<QkComplex64> native;
    native.reserve(matrix.size());
    for (const auto value : matrix) {
      native.push_back({.re = value.real(), .im = value.imag()});
    }
    const auto instructionIndex = qk_circuit_num_instructions(circuit_);
    checkExitCode(qk_circuit_unitary(circuit_, native.data(), targets.data(),
                                     static_cast<uint32_t>(targets.size()),
                                     true),
                  "adding unitary");
    if (numControls != 0U) {
      // The Qiskit C API can append only a bare unitary. Defer its control
      // wrapper until finish() exposes the Python operation.
      pendingControlledUnitaries_.push_back(
          {.instructionIndex = instructionIndex,
           .numControls = numControls,
           .qubits = qubits});
    }
  }

  [[nodiscard]] nb::object finish() override {
    if (circuit_ == nullptr) {
      throw std::runtime_error(
          "Qiskit circuit writer has already been finalized");
    }
    auto* result = qk_circuit_to_python_full(circuit_);
    circuit_ = nullptr;
    if (result == nullptr) {
      throwPythonError("Qiskit failed to create a QuantumCircuit");
    }
    auto pythonCircuit = nb::steal<nb::object>(result);
    try {
      replacePendingControlledUnitaries(pythonCircuit);
    } catch (const nb::python_error& error) {
      throwPythonError("Qiskit failed to construct a controlled unitary",
                       error);
    }
    return pythonCircuit;
  }

private:
  struct PendingControlledUnitary {
    size_t instructionIndex = 0U;
    uint32_t numControls = 0U;
    std::vector<uint32_t> qubits;
  };

  void replacePendingControlledUnitaries(const nb::handle pythonCircuit) const {
    auto data = pythonAttribute(pythonCircuit, "data",
                                "Qiskit circuit has no instruction data");
    const auto circuitQubits = pythonAttribute(pythonCircuit, "qubits",
                                               "Qiskit circuit has no qubits");
    for (const auto& pending : pendingControlledUnitaries_) {
      if (pending.instructionIndex >= nb::len(data)) {
        throw std::runtime_error(
            "Qiskit controlled-unitary placeholder is missing");
      }
      const auto placeholder =
          nb::borrow<nb::object>(data[pending.instructionIndex]);
      const auto operation =
          pythonAttribute(placeholder, "operation",
                          "Qiskit unitary placeholder has no operation");
      const auto controlled =
          pythonAttribute(operation, "control",
                          "Qiskit unitary operation cannot be controlled")(
              pending.numControls, nb::arg("annotated") = true);
      nb::list qargs;
      for (const auto qubit : pending.qubits) {
        if (qubit >= nb::len(circuitQubits)) {
          throw std::runtime_error(
              "Qiskit controlled unitary references an invalid qubit");
        }
        qargs.append(circuitQubits[qubit]);
      }
      const auto replacement =
          pythonAttribute(placeholder, "replace",
                          "Qiskit unitary placeholder cannot be replaced")(
              nb::arg("operation") = controlled, nb::arg("qubits") = qargs);
      data[pending.instructionIndex] = replacement;
    }
  }

  [[nodiscard]] const QkParam* nativeParameter(
      const Parameter& parameter,
      std::vector<std::unique_ptr<OwnedParameter>>& ownedParameters) {
    size_t nodeCount = 0U;
    return nativeParameter(parameter, ownedParameters, nodeCount, 1U);
  }

  [[nodiscard]] const QkParam*
  nativeParameter(const Parameter& parameter,
                  std::vector<std::unique_ptr<OwnedParameter>>& ownedParameters,
                  size_t& nodeCount, const size_t depth) {
    countParameterExpressionNode(nodeCount);
    if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
      throwParameterExpressionDepthError();
    }
    if (parameter.kind == ParameterKind::Number) {
      if (parameter.left != nullptr || parameter.right != nullptr) {
        throw std::runtime_error(
            "numeric parameter expression node has operands");
      }
      ownedParameters.emplace_back(
          std::make_unique<OwnedParameter>(parameter.number));
      return ownedParameters.back()->get();
    }
    if (parameter.kind == ParameterKind::Symbol) {
      if (parameter.left != nullptr || parameter.right != nullptr) {
        throw std::runtime_error(
            "symbolic parameter expression node has operands");
      }
      if (parameter.identity.empty()) {
        throw std::runtime_error(
            "cannot export a symbolic parameter without a stable identity");
      }
      if (parameter.text.empty()) {
        throw std::runtime_error(
            "cannot export a symbolic parameter without a name");
      }
      const auto found = symbols_.find(parameter.identity);
      if (found != symbols_.end()) {
        if (found->second.name != parameter.text) {
          throw std::runtime_error(
              "one symbolic parameter identity has conflicting metadata");
        }
        return found->second.parameter->get();
      }
      auto [inserted, success] =
          symbols_.emplace(parameter.identity,
                           Symbol{.name = parameter.text,
                                  .parameter = std::make_unique<OwnedParameter>(
                                      parameter.text)});
      static_cast<void>(success);
      return inserted->second.parameter->get();
    }

    const auto unary = parameter.kind == ParameterKind::Negate ||
                       parameter.kind == ParameterKind::Sin ||
                       parameter.kind == ParameterKind::Cos ||
                       parameter.kind == ParameterKind::Tan ||
                       parameter.kind == ParameterKind::ArcSin ||
                       parameter.kind == ParameterKind::ArcCos ||
                       parameter.kind == ParameterKind::ArcTan ||
                       parameter.kind == ParameterKind::Exp ||
                       parameter.kind == ParameterKind::Log ||
                       parameter.kind == ParameterKind::Abs ||
                       parameter.kind == ParameterKind::Conjugate;
    if (parameter.left == nullptr || (unary && parameter.right != nullptr) ||
        (!unary && parameter.right == nullptr)) {
      throw std::runtime_error("parameter expression has invalid operands");
    }
    const auto* left = nativeParameter(*parameter.left, ownedParameters,
                                       nodeCount, depth + 1U);
    const QkParam* right = nullptr;
    if (!unary) {
      right = nativeParameter(*parameter.right, ownedParameters, nodeCount,
                              depth + 1U);
    }
    auto output = std::make_unique<OwnedParameter>();
    QkExitCode result = QkExitCode_Success;
    switch (parameter.kind) {
    case ParameterKind::Number:
    case ParameterKind::Symbol:
      throw std::runtime_error("invalid parameter expression node");
    case ParameterKind::Add:
      result = qk_param_add(output->getMutable(), left, right);
      break;
    case ParameterKind::Subtract:
      result = qk_param_sub(output->getMutable(), left, right);
      break;
    case ParameterKind::Multiply:
      result = qk_param_mul(output->getMutable(), left, right);
      break;
    case ParameterKind::Divide:
      result = qk_param_div(output->getMutable(), left, right);
      break;
    case ParameterKind::Power:
      result = qk_param_pow(output->getMutable(), left, right);
      break;
    case ParameterKind::Negate:
      result = qk_param_neg(output->getMutable(), left);
      break;
    case ParameterKind::Sin:
      result = qk_param_sin(output->getMutable(), left);
      break;
    case ParameterKind::Cos:
      result = qk_param_cos(output->getMutable(), left);
      break;
    case ParameterKind::Tan:
      result = qk_param_tan(output->getMutable(), left);
      break;
    case ParameterKind::ArcSin:
      result = qk_param_asin(output->getMutable(), left);
      break;
    case ParameterKind::ArcCos:
      result = qk_param_acos(output->getMutable(), left);
      break;
    case ParameterKind::ArcTan:
      result = qk_param_atan(output->getMutable(), left);
      break;
    case ParameterKind::Exp:
      result = qk_param_exp(output->getMutable(), left);
      break;
    case ParameterKind::Log:
      result = qk_param_log(output->getMutable(), left);
      break;
    case ParameterKind::Abs:
      result = qk_param_abs(output->getMutable(), left);
      break;
    case ParameterKind::Conjugate:
      result = qk_param_conjugate(output->getMutable(), left);
      break;
    }
    checkExitCode(result, "constructing a parameter expression");
    const auto* value = output->get();
    ownedParameters.push_back(std::move(output));
    return value;
  }

  struct Symbol {
    std::string name;
    std::unique_ptr<OwnedParameter> parameter;
  };

  QkCircuit* circuit_ = nullptr;
  std::vector<PendingControlledUnitary> pendingControlledUnitaries_;
  std::unordered_map<std::string, Symbol> symbols_;
};

class NativeTranslation final : public VersionedTranslation {
public:
  [[nodiscard]] std::unique_ptr<CircuitReader>
  openCircuit(const nb::handle circuit) const override {
    return std::make_unique<NativeCircuitReader>(circuit);
  }
  [[nodiscard]] bool
  supportsGate(const StandardGateMapping gate) const override {
    return versionGate(gate) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<CircuitWriter>
  createCircuit(const uint32_t looseQubits,
                const uint32_t looseClbits) const override {
    return std::make_unique<NativeCircuitWriter>(looseQubits, looseClbits);
  }
};

} // namespace

std::unique_ptr<VersionedTranslation>
MQT_QISKIT_VERSION_FACTORY() { // NOLINT(misc-use-internal-linkage): declared in
                               // the version registry.
  if (qk_import() < 0) {
    throwPythonError("failed to initialize the Qiskit " MQT_QISKIT_VERSION_LABEL
                     " C API");
  }
  const auto version = qk_api_version();
  const auto major = (version >> 24U) & 0xffU;
  const auto minor = (version >> 16U) & 0xffU;
  if (major != MQT_QISKIT_VERSION_EXPECTED_MAJOR ||
      minor != MQT_QISKIT_VERSION_EXPECTED_MINOR ||
      (MQT_QISKIT_VERSION_EXACT_API != 0 && version != QISKIT_VERSION_HEX)) {
    throw std::runtime_error("Qiskit C API capsule version does not match the "
                             "selected " MQT_QISKIT_VERSION_LABEL
                             " translation");
  }
  return std::make_unique<NativeTranslation>();
}

} // namespace mqt::bindings::qiskit

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
// NOLINTEND(cppcoreguidelines-pro-type-cstyle-cast)
