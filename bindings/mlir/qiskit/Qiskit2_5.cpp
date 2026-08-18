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
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
      return {.number = number};
    }
  }
  // qk_str_free requires the mutable allocation returned by Qiskit.
  // NOLINTNEXTLINE(misc-const-correctness)
  char* const text = qk_param_str(parameter);
  if (text == nullptr) {
    throwPythonError("Qiskit failed to format an instruction parameter");
  }
  Parameter result{.number = std::nullopt, .text = text};
  qk_str_free(text);
  return result;
}

[[nodiscard]] Parameter normalizePythonParameter(const nb::handle parameter) {
  double number = 0.0;
  if (nb::try_cast(parameter, number)) {
    return {.number = number};
  }
  if (nb::hasattr(parameter, "name")) {
    return {.number = std::nullopt,
            .text = pythonStringAttribute(
                parameter, "name",
                "Qiskit modifier exponent has an invalid symbol name")};
  }
  auto text =
      pythonText(parameter, "Qiskit modifier exponent has a non-text value");
  return {.number = std::nullopt, .text = std::move(text)};
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

template <class NormalizeVariable>
[[nodiscard]] std::unique_ptr<Expression> normalizeExpression(
    const QkExprNode* expression, const nb::handle pythonExpression,
    NormalizeVariable& normalizeVariable, const size_t depth = 0U) {
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
    result->left = normalizeExpression(
        info.left,
        pythonAttribute(pythonExpression, "left",
                        "Qiskit binary expression has no left operand"),
        normalizeVariable, depth + 1U);
    result->right = normalizeExpression(
        info.right,
        pythonAttribute(pythonExpression, "right",
                        "Qiskit binary expression has no right operand"),
        normalizeVariable, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Unary: {
    const auto info = qk_expr_unary_info(expression);
    result->kind = ExpressionKind::Unary;
    result->unaryOperation = normalizeUnaryOperation(info.op);
    setType(*result, info.ty);
    result->left = normalizeExpression(
        info.operand,
        pythonAttribute(pythonExpression, "operand",
                        "Qiskit unary expression has no operand"),
        normalizeVariable, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Cast: {
    const auto info = qk_expr_cast_info(expression);
    result->kind = ExpressionKind::Cast;
    setType(*result, info.ty);
    result->left = normalizeExpression(
        info.operand,
        pythonAttribute(pythonExpression, "operand",
                        "Qiskit cast expression has no operand"),
        normalizeVariable, depth + 1U);
    return result;
  }
  case QkExprNodeKind_Index: {
    const auto info = qk_expr_index_info(expression);
    result->kind = ExpressionKind::Index;
    setType(*result, info.ty);
    result->left = normalizeExpression(
        info.target,
        pythonAttribute(pythonExpression, "target",
                        "Qiskit index expression has no target"),
        normalizeVariable, depth + 1U);
    result->right = normalizeExpression(
        info.index,
        pythonAttribute(pythonExpression, "index",
                        "Qiskit index expression has no index"),
        normalizeVariable, depth + 1U);
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
    setType(*result, qk_var_type_info(qk_expr_as_var(expression)));
    normalizeVariable(*result, pythonExpression);
    return result;
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
  explicit OwnedParameter(const Parameter& parameter) {
    if (!parameter.number) {
      if (parameter.text.empty()) {
        throw std::runtime_error(
            "Qiskit parameter symbol name cannot be empty");
      }
      value_ = qk_param_new_symbol(parameter.text.c_str());
    } else if (!std::isfinite(*parameter.number)) {
      throw std::runtime_error(
          "cannot construct a non-finite Qiskit parameter");
    } else {
      value_ = qk_param_from_double(*parameter.number);
    }
    if (value_ == nullptr) {
      throwPythonError("Qiskit failed to construct a circuit parameter");
    }
  }

  OwnedParameter(const OwnedParameter&) = delete;
  OwnedParameter& operator=(const OwnedParameter&) = delete;
  OwnedParameter(OwnedParameter&&) = delete;
  OwnedParameter& operator=(OwnedParameter&&) = delete;
  ~OwnedParameter() { qk_param_free(value_); }

  [[nodiscard]] const QkParam* get() const { return value_; }

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

  [[nodiscard]] Parameter globalPhase() const override {
    // qk_param_free requires the mutable allocation returned by Qiskit.
    // NOLINTNEXTLINE(misc-const-correctness)
    QkParam* const phase = qk_circuit_global_phase(circuit_);
    if (phase == nullptr) {
      throwPythonError("Qiskit failed to read the circuit global phase");
    }
    const auto result = normalizeParameter(phase);
    qk_param_free(phase);
    return result;
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
    for (const auto* parameter :
         std::span(native.params, static_cast<size_t>(native.num_params))) {
      result.parameters.emplace_back(normalizeParameter(parameter));
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

    const auto definitionParameters =
        nb::cast<nb::list>(nb::module_::import_("builtins")
                               .attr("list")(pythonAttribute(
                                   definition, "parameters",
                                   "Qiskit definition has no parameter list")));
    if (definitionParameters.empty()) {
      return std::make_unique<NativeCircuitReader>(definition);
    }
    throw std::runtime_error(
        "Qiskit custom instruction definitions must be numerically bound "
        "before import");
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
                          nb::object instruction)
      : rootCircuit_(rootCircuit),
        controlFlow_(
            qk_circuit_get_control_flow_instruction(circuit, index, parent)),
        instruction_(std::move(instruction)),
        operation_(pythonAttribute(
            instruction_, "operation",
            "Qiskit circuit instruction has no control-flow operation")) {
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
      result.expression = normalizePythonExpression(
          qk_control_flow_condition_expr(controlFlow_),
          pythonAttribute(operation_, "condition",
                          "Qiskit control flow has no condition"));
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
        qk_str_free(symbol.name);
        throw std::runtime_error(
            "Qiskit indexed parameter-vector loop variables are not supported");
      }
      result.parameter = symbol.name;
      qk_str_free(symbol.name);
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
    const auto target =
        pythonAttribute(operation_, "target", "Qiskit switch has no target");
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    if (nb::isinstance(target, circuitModule.attr("Clbit"))) {
      result.kind = ClassicalTargetKind::ClassicalBit;
      result.bit = rootClbitIndex(target);
      return result;
    }
    if (nb::isinstance(target, circuitModule.attr("ClassicalRegister"))) {
      result.kind = ClassicalTargetKind::ClassicalRegister;
      result.reg.name = pythonStringAttribute(
          target, "name", "Qiskit switch register has no name");
      if (nb::len(target) == 0U || nb::len(target) > 64U) {
        throw std::runtime_error(
            "Qiskit switch registers must contain between 1 and 64 bits");
      }
      result.reg.bits.reserve(nb::len(target));
      for (const nb::handle bit : nb::iter(target)) {
        result.reg.bits.push_back(rootClbitIndex(bit));
      }
      result.width = static_cast<uint32_t>(result.reg.bits.size());
      return result;
    }
    const auto expressionModule =
        nb::module_::import_("qiskit.circuit.classical.expr");
    if (nb::isinstance(target, expressionModule.attr("Expr"))) {
      result.kind = ClassicalTargetKind::Expression;
      // Qiskit 2.5's C switch-target accessors abort on expression targets.
      result.expression = normalizePythonExpressionOnly(target);
      return result;
    }
    throw std::runtime_error("Qiskit switch has an unknown target type");
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
  [[nodiscard]] uint32_t rootClbitIndex(const nb::handle bit) const {
    const auto clbits = pythonAttribute(
        instruction_, "clbits",
        "Qiskit control-flow instruction has no classical-bit operands");
    if (numBlocks() == 0U ||
        nb::len(clbits) != qk_circuit_num_clbits(qk_control_flow_block_circuit(
                               controlFlow_, 0U))) {
      throw std::runtime_error(
          "Qiskit control flow has incompatible classical-bit captures");
    }
    const auto* const map = qk_control_flow_clbit_map(controlFlow_);
    if (map == nullptr && nb::len(clbits) != 0U) {
      throw std::runtime_error(
          "Qiskit control flow has no classical-bit capture map");
    }
    for (size_t index = 0U; index < nb::len(clbits); ++index) {
      if (clbits[index].equal(bit)) {
        return map[index];
      }
    }
    throw std::runtime_error(
        "Qiskit expression variable is absent from its control-flow captures");
  }

  static void setPythonExpressionType(Expression& result,
                                      const nb::handle pythonExpression) {
    const auto type = pythonAttribute(pythonExpression, "type",
                                      "Qiskit expression has no type");
    const auto typeName = pythonStringAttribute(
        pythonAttribute(type, "__class__",
                        "Qiskit expression type has no Python class"),
        "__name__", "Qiskit expression type has no class name");
    if (typeName == "Bool") {
      result.type = ClassicalType::Bool;
      result.width = 1U;
      return;
    }
    if (typeName == "Uint") {
      const auto width = pythonUnsignedAttribute(
          type, "width", "Qiskit Uint expression has no width");
      if (width == 0U || width > 64U) {
        throw std::runtime_error(
            "Qiskit unsigned classical values must be between 1 and 64 bits");
      }
      result.type = ClassicalType::Uint;
      result.width = static_cast<uint32_t>(width);
      return;
    }
    if (typeName == "Float") {
      result.type = ClassicalType::Float;
      result.width = 64U;
      return;
    }
    if (typeName == "Duration") {
      throw std::runtime_error(
          "Qiskit circuit import does not support duration expressions");
    }
    throw std::runtime_error("Qiskit expression has an unknown Python type");
  }

  [[nodiscard]] static BinaryOperation
  pythonBinaryOperation(const std::string_view name) {
    if (name == "BIT_AND") {
      return BinaryOperation::BitAnd;
    }
    if (name == "BIT_OR") {
      return BinaryOperation::BitOr;
    }
    if (name == "BIT_XOR") {
      return BinaryOperation::BitXor;
    }
    if (name == "LOGIC_AND") {
      return BinaryOperation::LogicAnd;
    }
    if (name == "LOGIC_OR") {
      return BinaryOperation::LogicOr;
    }
    if (name == "EQUAL") {
      return BinaryOperation::Equal;
    }
    if (name == "NOT_EQUAL") {
      return BinaryOperation::NotEqual;
    }
    if (name == "LESS") {
      return BinaryOperation::Less;
    }
    if (name == "LESS_EQUAL") {
      return BinaryOperation::LessEqual;
    }
    if (name == "GREATER") {
      return BinaryOperation::Greater;
    }
    if (name == "GREATER_EQUAL") {
      return BinaryOperation::GreaterEqual;
    }
    if (name == "SHIFT_LEFT") {
      return BinaryOperation::ShiftLeft;
    }
    if (name == "SHIFT_RIGHT") {
      return BinaryOperation::ShiftRight;
    }
    if (name == "ADD") {
      return BinaryOperation::Add;
    }
    if (name == "SUB") {
      return BinaryOperation::Subtract;
    }
    if (name == "MUL") {
      return BinaryOperation::Multiply;
    }
    if (name == "DIV") {
      return BinaryOperation::Divide;
    }
    throw std::runtime_error(
        "Qiskit expression has an unknown Python binary operation");
  }

  [[nodiscard]] static UnaryOperation
  pythonUnaryOperation(const std::string_view name) {
    if (name == "BIT_NOT") {
      return UnaryOperation::BitNot;
    }
    if (name == "LOGIC_NOT") {
      return UnaryOperation::LogicNot;
    }
    if (name == "NEGATE") {
      return UnaryOperation::Negate;
    }
    throw std::runtime_error(
        "Qiskit expression has an unknown Python unary operation");
  }

  [[nodiscard]] std::unique_ptr<Expression>
  normalizePythonExpressionOnly(const nb::handle pythonExpression,
                                const size_t depth = 0U) const {
    if (depth >= MAX_EXPRESSION_DEPTH) {
      throw std::runtime_error(
          "Qiskit classical expressions exceed the nesting limit of 64");
    }
    auto result = std::make_unique<Expression>();
    setPythonExpressionType(*result, pythonExpression);
    const auto className = pythonStringAttribute(
        pythonAttribute(pythonExpression, "__class__",
                        "Qiskit expression has no Python class"),
        "__name__", "Qiskit expression has no class name");
    if (className == "Var") {
      normalizePythonVariable(*result, pythonExpression);
      return result;
    }
    if (className == "Value") {
      result->kind = ExpressionKind::Value;
      const auto value = pythonAttribute(
          pythonExpression, "value", "Qiskit literal expression has no value");
      switch (result->type) {
      case ClassicalType::Bool:
        if (!nb::try_cast(value, result->boolValue)) {
          throw std::runtime_error(
              "Qiskit Boolean expression has an invalid value");
        }
        break;
      case ClassicalType::Uint:
        if (!nb::try_cast(value, result->uintValue)) {
          throw std::runtime_error(
              "Qiskit Uint expression has an invalid value");
        }
        break;
      case ClassicalType::Float:
        if (!nb::try_cast(value, result->floatValue) ||
            !std::isfinite(result->floatValue)) {
          throw std::runtime_error(
              "Qiskit Float expression has an invalid value");
        }
        break;
      }
      return result;
    }
    if (className == "Unary") {
      result->kind = ExpressionKind::Unary;
      result->unaryOperation = pythonUnaryOperation(pythonStringAttribute(
          pythonAttribute(pythonExpression, "op",
                          "Qiskit unary expression has no operation"),
          "name", "Qiskit unary expression operation has no name"));
      result->left = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "operand",
                          "Qiskit unary expression has no operand"),
          depth + 1U);
      return result;
    }
    if (className == "Binary") {
      result->kind = ExpressionKind::Binary;
      result->binaryOperation = pythonBinaryOperation(pythonStringAttribute(
          pythonAttribute(pythonExpression, "op",
                          "Qiskit binary expression has no operation"),
          "name", "Qiskit binary expression operation has no name"));
      result->left = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "left",
                          "Qiskit binary expression has no left operand"),
          depth + 1U);
      result->right = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "right",
                          "Qiskit binary expression has no right operand"),
          depth + 1U);
      return result;
    }
    if (className == "Cast") {
      result->kind = ExpressionKind::Cast;
      result->left = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "operand",
                          "Qiskit cast expression has no operand"),
          depth + 1U);
      return result;
    }
    if (className == "Index") {
      result->kind = ExpressionKind::Index;
      result->left = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "target",
                          "Qiskit index expression has no target"),
          depth + 1U);
      result->right = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "index",
                          "Qiskit index expression has no index"),
          depth + 1U);
      return result;
    }
    if (className == "Stretch") {
      throw std::runtime_error(
          "Qiskit circuit import does not support stretch expressions");
    }
    throw std::runtime_error("Qiskit expression has an unknown Python node");
  }

  void normalizePythonVariable(Expression& result,
                               const nb::handle pythonExpression) const {
    const auto variable = pythonAttribute(
        pythonExpression, "var", "Qiskit variable expression has no value");
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    if (nb::isinstance(variable, circuitModule.attr("Clbit"))) {
      if (result.type != ClassicalType::Bool || result.width != 1U) {
        throw std::runtime_error(
            "Qiskit classical-bit variable must have Boolean type");
      }
      result.kind = ExpressionKind::ClassicalBit;
      result.bit = rootClbitIndex(variable);
      return;
    }
    if (nb::isinstance(variable, circuitModule.attr("ClassicalRegister"))) {
      if (result.type != ClassicalType::Uint || nb::len(variable) == 0U ||
          nb::len(variable) > 64U || result.width < nb::len(variable)) {
        throw std::runtime_error(
            "Qiskit classical-register variable has an invalid type");
      }
      result.kind = ExpressionKind::ClassicalRegister;
      result.reg.name = pythonStringAttribute(
          variable, "name", "Qiskit classical register has no name");
      result.reg.bits.reserve(nb::len(variable));
      for (const nb::handle bit : nb::iter(variable)) {
        result.reg.bits.push_back(rootClbitIndex(bit));
      }
      return;
    }
    throw std::runtime_error(
        "Qiskit circuit import does not support standalone variables in "
        "classical expressions");
  }

  [[nodiscard]] std::unique_ptr<Expression>
  normalizePythonExpression(const QkExprNode* expression,
                            const nb::handle pythonExpression) const {
    auto normalizeVariable = [this](Expression& result,
                                    const nb::handle pythonVariable) {
      normalizePythonVariable(result, pythonVariable);
    };
    return normalizeExpression(expression, pythonExpression, normalizeVariable);
  }

  const QkCircuit* rootCircuit_ = nullptr;
  QkControlFlowInstruction* controlFlow_ = nullptr;
  nb::object instruction_;
  nb::object operation_;
};

std::unique_ptr<ControlFlowReader>
NativeCircuitReader::controlFlow(const size_t index) const {
  return std::make_unique<NativeControlFlowReader>(
      rootCircuit_, circuit_, index, parent_,
      nb::borrow<nb::object>(data_[index]));
}

class PythonClassicalBuilder final {
public:
  explicit PythonClassicalBuilder(const nb::handle circuit)
      : circuit_(nb::borrow<nb::object>(circuit)),
        clbits_(pythonAttribute(circuit, "clbits",
                                "Qiskit circuit has no classical bits")),
        expressionModule_(
            nb::module_::import_("qiskit.circuit.classical.expr")),
        typesModule_(nb::module_::import_("qiskit.circuit.classical.types")) {}

  [[nodiscard]] nb::object expression(const Expression& value) const {
    return expression(value, 0U);
  }

  [[nodiscard]] nb::object condition(const ClassicalTarget& target) const {
    switch (target.kind) {
    case ClassicalTargetKind::ClassicalBit:
      return nb::make_tuple(classicalBit(target.bit),
                            nb::bool_(target.expectedBit));
    case ClassicalTargetKind::ClassicalRegister: {
      validateRegisterValue(target.reg, target.expectedRegister);
      if (const auto reg = registeredClassicalRegister(target.reg)) {
        return nb::make_tuple(*reg, nb::int_(target.expectedRegister));
      }
      const auto packed = packedRegister(target.reg);
      const auto expected = expressionModule_.attr("lift")(
          nb::int_(target.expectedRegister),
          classicalType(ClassicalType::Uint,
                        static_cast<uint32_t>(target.reg.bits.size())));
      return expressionModule_.attr("equal")(packed, expected);
    }
    case ClassicalTargetKind::Expression:
      if (!target.expression) {
        throw std::runtime_error(
            "Qiskit control-flow condition has no expression");
      }
      if (target.expression->type != ClassicalType::Bool) {
        throw std::runtime_error(
            "Qiskit control-flow condition expression must be Boolean");
      }
      return expression(*target.expression);
    }
    throw std::runtime_error("Qiskit control flow has an unknown condition");
  }

  [[nodiscard]] nb::object switchTarget(const ClassicalTarget& target) const {
    switch (target.kind) {
    case ClassicalTargetKind::ClassicalBit:
      return classicalBit(target.bit);
    case ClassicalTargetKind::ClassicalRegister:
      if (target.reg.bits.empty() || target.reg.bits.size() > 64U) {
        throw std::runtime_error(
            "Qiskit switch registers must contain between 1 and 64 bits");
      }
      if (const auto reg = registeredClassicalRegister(target.reg)) {
        return *reg;
      }
      return packedRegister(target.reg);
    case ClassicalTargetKind::Expression:
      if (!target.expression) {
        throw std::runtime_error("Qiskit switch target has no expression");
      }
      if (target.expression->type == ClassicalType::Float) {
        throw std::runtime_error(
            "Qiskit switch target expression cannot be floating-point");
      }
      return expression(*target.expression);
    }
    throw std::runtime_error(
        "Qiskit control flow has an unknown switch target");
  }

private:
  [[nodiscard]] nb::object classicalType(const ClassicalType type,
                                         const uint32_t width) const {
    switch (type) {
    case ClassicalType::Bool:
      if (width != 1U) {
        throw std::runtime_error("Qiskit Boolean expressions require width 1");
      }
      return typesModule_.attr("Bool")();
    case ClassicalType::Uint:
      if (width == 0U || width > 64U) {
        throw std::runtime_error(
            "Qiskit unsigned expressions require a width from 1 to 64");
      }
      return typesModule_.attr("Uint")(width);
    case ClassicalType::Float:
      if (width != 64U) {
        throw std::runtime_error(
            "Qiskit floating-point expressions require width 64");
      }
      return typesModule_.attr("Float")();
    }
    throw std::runtime_error("Qiskit expression has an unknown type");
  }

  [[nodiscard]] nb::object classicalBit(const uint32_t bit) const {
    if (bit >= nb::len(clbits_)) {
      throw std::runtime_error(
          "Qiskit classical expression references an invalid bit");
    }
    return nb::borrow<nb::object>(clbits_[bit]);
  }

  [[nodiscard]] std::optional<nb::object>
  registeredClassicalRegister(const Register& reg) const {
    const auto registers = pythonAttribute(
        circuit_, "cregs", "Qiskit circuit has no classical registers");
    std::optional<nb::object> matchingBits;
    for (const nb::handle candidateHandle : nb::iter(registers)) {
      if (nb::len(candidateHandle) != reg.bits.size()) {
        continue;
      }
      auto candidate = nb::borrow<nb::object>(candidateHandle);
      bool matches = true;
      for (size_t index = 0U; index < reg.bits.size(); ++index) {
        if (!candidate[index].equal(classicalBit(reg.bits[index]))) {
          matches = false;
          break;
        }
      }
      if (!matches) {
        continue;
      }
      if (pythonStringAttribute(candidate, "name",
                                "Qiskit classical register has no name") ==
          reg.name) {
        return candidate;
      }
      matchingBits = std::move(candidate);
    }
    return matchingBits;
  }

  static void validateRegisterValue(const Register& reg, const uint64_t value) {
    if (reg.bits.empty() || reg.bits.size() > 64U) {
      throw std::runtime_error(
          "Qiskit condition registers must contain between 1 and 64 bits");
    }
    if (reg.bits.size() < std::numeric_limits<uint64_t>::digits &&
        value >= (uint64_t{1} << reg.bits.size())) {
      throw std::runtime_error(
          "Qiskit register condition value exceeds its register width");
    }
  }

  [[nodiscard]] nb::object
  packedRegister(const Register& reg,
                 const uint32_t expressionWidth = 0U) const {
    const auto width = expressionWidth == 0U
                           ? static_cast<uint32_t>(reg.bits.size())
                           : expressionWidth;
    if (reg.bits.empty() || reg.bits.size() > 64U || width < reg.bits.size() ||
        width > 64U) {
      throw std::runtime_error(
          "Qiskit expression register has an invalid width");
    }
    std::unordered_set<uint32_t> seen;
    std::vector<nb::object> terms;
    terms.reserve(reg.bits.size());
    const auto type = classicalType(ClassicalType::Uint, width);
    for (size_t index = 0U; index < reg.bits.size(); ++index) {
      if (!seen.insert(reg.bits[index]).second) {
        throw std::runtime_error(
            "Qiskit expression register contains a repeated bit");
      }
      auto term =
          expressionModule_.attr("cast")(classicalBit(reg.bits[index]), type);
      if (index != 0U) {
        term = expressionModule_.attr("shift_left")(term, nb::int_(index));
      }
      terms.emplace_back(std::move(term));
    }
    while (terms.size() > 1U) {
      std::vector<nb::object> reduced;
      reduced.reserve((terms.size() + 1U) / 2U);
      for (size_t index = 0U; index < terms.size(); index += 2U) {
        if (index + 1U == terms.size()) {
          reduced.emplace_back(std::move(terms[index]));
          continue;
        }
        reduced.emplace_back(
            expressionModule_.attr("bit_or")(terms[index], terms[index + 1U]));
      }
      terms = std::move(reduced);
    }
    return std::move(terms.front());
  }

  [[nodiscard]] static const char* binaryFunction(const BinaryOperation op) {
    switch (op) {
    case BinaryOperation::BitAnd:
      return "bit_and";
    case BinaryOperation::BitOr:
      return "bit_or";
    case BinaryOperation::BitXor:
      return "bit_xor";
    case BinaryOperation::LogicAnd:
      return "logic_and";
    case BinaryOperation::LogicOr:
      return "logic_or";
    case BinaryOperation::Equal:
      return "equal";
    case BinaryOperation::NotEqual:
      return "not_equal";
    case BinaryOperation::Less:
      return "less";
    case BinaryOperation::LessEqual:
      return "less_equal";
    case BinaryOperation::Greater:
      return "greater";
    case BinaryOperation::GreaterEqual:
      return "greater_equal";
    case BinaryOperation::ShiftLeft:
      return "shift_left";
    case BinaryOperation::ShiftRight:
      return "shift_right";
    case BinaryOperation::Add:
      return "add";
    case BinaryOperation::Subtract:
      return "sub";
    case BinaryOperation::Multiply:
      return "mul";
    case BinaryOperation::Divide:
      return "div";
    }
    throw std::runtime_error(
        "Qiskit expression has an unknown binary operation");
  }

  [[nodiscard]] static const char* unaryFunction(const UnaryOperation op) {
    switch (op) {
    case UnaryOperation::BitNot:
      return "bit_not";
    case UnaryOperation::LogicNot:
      return "logic_not";
    case UnaryOperation::Negate:
      return "negate";
    }
    throw std::runtime_error(
        "Qiskit expression has an unknown unary operation");
  }

  [[nodiscard]] nb::object expression(const Expression& value,
                                      const size_t depth) const {
    if (depth >= MAX_EXPRESSION_DEPTH) {
      throw std::runtime_error(
          "Qiskit classical expressions exceed the nesting limit of 64");
    }
    const auto requireOperand = [](const std::unique_ptr<Expression>& operand) {
      if (!operand) {
        throw std::runtime_error(
            "Qiskit classical expression has a missing operand");
      }
      return operand.get();
    };
    switch (value.kind) {
    case ExpressionKind::Value: {
      const auto type = classicalType(value.type, value.width);
      switch (value.type) {
      case ClassicalType::Bool:
        return expressionModule_.attr("lift")(nb::bool_(value.boolValue), type);
      case ClassicalType::Uint:
        if (value.width < std::numeric_limits<uint64_t>::digits &&
            value.uintValue >= (uint64_t{1} << value.width)) {
          throw std::runtime_error(
              "Qiskit unsigned expression value exceeds its width");
        }
        return expressionModule_.attr("lift")(nb::int_(value.uintValue), type);
      case ClassicalType::Float:
        if (!std::isfinite(value.floatValue)) {
          throw std::runtime_error(
              "Qiskit floating-point expression value must be finite");
        }
        return expressionModule_.attr("lift")(nb::float_(value.floatValue),
                                              type);
      }
      break;
    }
    case ExpressionKind::ClassicalBit:
      if (value.type != ClassicalType::Bool || value.width != 1U) {
        throw std::runtime_error(
            "Qiskit classical-bit expression must have Boolean type");
      }
      return expressionModule_.attr("lift")(classicalBit(value.bit));
    case ExpressionKind::ClassicalRegister:
      if (value.type != ClassicalType::Uint || value.width == 0U ||
          value.width < value.reg.bits.size() || value.width > 64U) {
        throw std::runtime_error(
            "Qiskit classical-register expression has an invalid type");
      }
      if (const auto reg = registeredClassicalRegister(value.reg)) {
        return expressionModule_.attr("lift")(
            *reg, classicalType(ClassicalType::Uint, value.width));
      }
      return packedRegister(value.reg, value.width);
    case ExpressionKind::Unary:
      return expressionModule_.attr(unaryFunction(value.unaryOperation))(
          expression(*requireOperand(value.left), depth + 1U));
    case ExpressionKind::Binary:
      return expressionModule_.attr(binaryFunction(value.binaryOperation))(
          expression(*requireOperand(value.left), depth + 1U),
          expression(*requireOperand(value.right), depth + 1U));
    case ExpressionKind::Cast:
      return expressionModule_.attr("cast")(
          expression(*requireOperand(value.left), depth + 1U),
          classicalType(value.type, value.width));
    case ExpressionKind::Index:
      return expressionModule_.attr("index")(
          expression(*requireOperand(value.left), depth + 1U),
          expression(*requireOperand(value.right), depth + 1U));
    }
    throw std::runtime_error("Qiskit classical expression has an unknown kind");
  }

  nb::object circuit_;
  nb::object clbits_;
  nb::object expressionModule_;
  nb::object typesModule_;
};

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
    if (!phase.number) {
      checkExitCode(
          qk_circuit_set_global_phase(circuit_, symbolicParameter(phase.text)),
          "setting global phase");
      return;
    }
    const OwnedParameter parameter(phase);
    checkExitCode(qk_circuit_set_global_phase(circuit_, parameter.get()),
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
      if (!parameter.number) {
        nativeParameters.emplace_back(symbolicParameter(parameter.text));
        continue;
      }
      ownedParameters.emplace_back(std::make_unique<OwnedParameter>(parameter));
      nativeParameters.emplace_back(ownedParameters.back()->get());
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

  void addControlFlow(const ControlFlowKind kind, ClassicalTarget target,
                      Loop loop, std::vector<SwitchCase> switchCases,
                      std::vector<std::unique_ptr<CircuitWriter>> blocks,
                      const std::vector<uint32_t>& qubits,
                      const std::vector<uint32_t>& clbits) override {
    validateControlFlowShape(kind, target, loop, switchCases, blocks, qubits,
                             clbits);
    for (const auto& block : blocks) {
      const auto* const native =
          dynamic_cast<const NativeCircuitWriter*>(block.get());
      if (native == nullptr) {
        throw std::runtime_error(
            "Qiskit control-flow blocks use an incompatible writer");
      }
      if (native->circuit_ == nullptr ||
          qk_circuit_num_qubits(native->circuit_) != qubits.size() ||
          qk_circuit_num_clbits(native->circuit_) != clbits.size()) {
        throw std::runtime_error(
            "Qiskit control-flow block has incompatible bit counts");
      }
    }
    pendingControlFlow_.push_back(
        {.instructionIndex = qk_circuit_num_instructions(circuit_),
         .kind = kind,
         .target = std::move(target),
         .loop = std::move(loop),
         .switchCases = std::move(switchCases),
         .blockWriters = std::move(blocks),
         .qubits = qubits,
         .clbits = clbits});
  }

  [[nodiscard]] nb::object finish() override {
    return finishImpl(false, nb::none(), nb::none());
  }

private:
  [[nodiscard]] nb::object finishImpl(const bool rebase,
                                      const nb::handle exactQubits,
                                      const nb::handle exactClbits) {
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
      if (rebase) {
        pythonCircuit = rebaseCircuit(pythonCircuit, exactQubits, exactClbits);
      }
      const auto unitaryReplacements =
          pendingControlledUnitaryReplacements(pythonCircuit);
      finalizeControlFlowBlocks(pythonCircuit);
      const auto canonicalParameters =
          canonicalizeControlFlowParameters(pythonCircuit);
      const auto controlFlowInstructions =
          pendingControlFlowInstructions(pythonCircuit, canonicalParameters);
      applyPendingInstructions(pythonCircuit, unitaryReplacements,
                               controlFlowInstructions);
    } catch (const nb::python_error& error) {
      throwPythonError("Qiskit failed to construct deferred instructions",
                       error);
    }
    return pythonCircuit;
  }

  struct PendingControlledUnitary {
    size_t instructionIndex = 0U;
    uint32_t numControls = 0U;
    std::vector<uint32_t> qubits;
  };

  struct PendingControlFlow {
    size_t instructionIndex = 0U;
    ControlFlowKind kind = ControlFlowKind::IfElse;
    ClassicalTarget target;
    Loop loop;
    std::vector<SwitchCase> switchCases;
    std::vector<std::unique_ptr<CircuitWriter>> blockWriters;
    std::vector<nb::object> blocks;
    std::vector<uint32_t> qubits;
    std::vector<uint32_t> clbits;
  };

  struct IndexedPythonInstruction {
    size_t instructionIndex = 0U;
    nb::object instruction;
  };

  using PythonParameterMap = std::unordered_map<std::string, nb::object>;

  [[nodiscard]] const QkParam* symbolicParameter(const std::string_view name) {
    if (name.empty()) {
      throw std::runtime_error("Qiskit parameter symbol name cannot be empty");
    }
    if (const auto found = symbolicParameters_.find(std::string(name));
        found != symbolicParameters_.end()) {
      return found->second->get();
    }
    const Parameter parameter{.number = std::nullopt,
                              .text = std::string(name)};
    auto owned = std::make_unique<OwnedParameter>(parameter);
    const auto* result = owned->get();
    symbolicParameters_.emplace(parameter.text, std::move(owned));
    return result;
  }

  static void collectExpressionBits(const Expression& expression,
                                    std::unordered_set<uint32_t>& bits,
                                    const size_t depth = 0U) {
    if (depth >= MAX_EXPRESSION_DEPTH) {
      throw std::runtime_error(
          "Qiskit classical expressions exceed the nesting limit of 64");
    }
    const auto collectOperand =
        [&](const std::unique_ptr<Expression>& operand) {
          if (!operand) {
            throw std::runtime_error(
                "Qiskit classical expression has a missing operand");
          }
          collectExpressionBits(*operand, bits, depth + 1U);
        };
    switch (expression.kind) {
    case ExpressionKind::Value:
      return;
    case ExpressionKind::ClassicalBit:
      bits.insert(expression.bit);
      return;
    case ExpressionKind::ClassicalRegister:
      bits.insert(expression.reg.bits.begin(), expression.reg.bits.end());
      return;
    case ExpressionKind::Unary:
    case ExpressionKind::Cast:
      collectOperand(expression.left);
      return;
    case ExpressionKind::Binary:
    case ExpressionKind::Index:
      collectOperand(expression.left);
      collectOperand(expression.right);
      return;
    }
  }

  static void
  validateTargetCaptures(const ClassicalTarget& target,
                         const std::vector<uint32_t>& capturedClbits) {
    std::unordered_set<uint32_t> referenced;
    switch (target.kind) {
    case ClassicalTargetKind::ClassicalBit:
      referenced.insert(target.bit);
      break;
    case ClassicalTargetKind::ClassicalRegister:
      referenced.insert(target.reg.bits.begin(), target.reg.bits.end());
      break;
    case ClassicalTargetKind::Expression:
      if (!target.expression) {
        throw std::runtime_error(
            "Qiskit control flow contains an empty classical expression");
      }
      collectExpressionBits(*target.expression, referenced);
      break;
    }
    const std::unordered_set<uint32_t> captured(capturedClbits.begin(),
                                                capturedClbits.end());
    for (const auto bit : referenced) {
      if (!captured.contains(bit)) {
        throw std::runtime_error(
            "Qiskit control flow does not capture a referenced classical bit");
      }
    }
  }

  static void validateControlFlowShape(
      const ControlFlowKind kind, const ClassicalTarget& target,
      const Loop& loop, const std::vector<SwitchCase>& switchCases,
      const std::vector<std::unique_ptr<CircuitWriter>>& blocks,
      const std::vector<uint32_t>& qubits,
      const std::vector<uint32_t>& clbits) {
    const auto requireUnique = [](const std::vector<uint32_t>& bits,
                                  const std::string_view kindName) {
      std::unordered_set<uint32_t> seen;
      for (const auto bit : bits) {
        if (!seen.insert(bit).second) {
          throw std::runtime_error("Qiskit control flow repeats a " +
                                   std::string(kindName));
        }
      }
    };
    requireUnique(qubits, "qubit capture");
    requireUnique(clbits, "classical-bit capture");
    for (const auto& block : blocks) {
      if (!block) {
        throw std::runtime_error("Qiskit control flow has an empty block");
      }
    }

    switch (kind) {
    case ControlFlowKind::Box:
    case ControlFlowKind::Break:
    case ControlFlowKind::Continue:
      throw std::runtime_error(
          "Qiskit circuit export does not support this control-flow kind");
    case ControlFlowKind::IfElse:
      if (blocks.empty() || blocks.size() > 2U) {
        throw std::runtime_error("Qiskit if/else requires one or two blocks");
      }
      break;
    case ControlFlowKind::While:
      if (blocks.size() != 1U) {
        throw std::runtime_error("Qiskit while loop requires one block");
      }
      break;
    case ControlFlowKind::For:
      if (blocks.size() != 1U) {
        throw std::runtime_error("Qiskit for loop requires one block");
      }
      if (loop.isRange && loop.step == 0) {
        throw std::runtime_error("Qiskit for-loop range step cannot be zero");
      }
      if (loop.parameter && loop.parameter->empty()) {
        throw std::runtime_error(
            "Qiskit for-loop parameter name cannot be empty");
      }
      break;
    case ControlFlowKind::Switch: {
      if (blocks.empty() || switchCases.size() != blocks.size()) {
        throw std::runtime_error(
            "Qiskit switch metadata must match its non-empty block list");
      }
      bool foundDefault = false;
      std::unordered_set<uint64_t> labels;
      for (size_t index = 0U; index < switchCases.size(); ++index) {
        const auto& switchCase = switchCases[index];
        if (switchCase.isDefault) {
          if (std::exchange(foundDefault, true) ||
              index + 1U != switchCases.size() || !switchCase.labels.empty()) {
            throw std::runtime_error(
                "Qiskit switch requires one final unlabeled default case");
          }
          continue;
        }
        if (switchCase.labels.empty()) {
          throw std::runtime_error(
              "Qiskit switch case requires at least one label");
        }
        for (const auto label : switchCase.labels) {
          if (!labels.insert(label).second) {
            throw std::runtime_error(
                "Qiskit switch contains a repeated case label");
          }
        }
      }
      break;
    }
    }
    if (kind != ControlFlowKind::Switch && !switchCases.empty()) {
      throw std::runtime_error(
          "Qiskit non-switch control flow has switch-case metadata");
    }
    if (kind == ControlFlowKind::IfElse || kind == ControlFlowKind::While ||
        kind == ControlFlowKind::Switch) {
      validateTargetCaptures(target, clbits);
    }
  }

  [[nodiscard]] std::vector<IndexedPythonInstruction>
  pendingControlledUnitaryReplacements(const nb::handle pythonCircuit) const {
    std::vector<IndexedPythonInstruction> result;
    result.reserve(pendingControlledUnitaries_.size());
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
      result.push_back({.instructionIndex = pending.instructionIndex,
                        .instruction = replacement});
    }
    return result;
  }

  [[nodiscard]] static nb::object rebaseCircuit(const nb::handle circuit,
                                                const nb::handle exactQubits,
                                                const nb::handle exactClbits) {
    if (nb::len(pythonAttribute(circuit, "qubits",
                                "Qiskit circuit has no qubits")) !=
            nb::len(exactQubits) ||
        nb::len(pythonAttribute(circuit, "clbits",
                                "Qiskit circuit has no classical bits")) !=
            nb::len(exactClbits)) {
      throw std::runtime_error(
          "Qiskit control-flow block has incompatible bit counts");
    }
    const auto quantumCircuit =
        nb::module_::import_("qiskit.circuit").attr("QuantumCircuit");
    auto rebased = quantumCircuit();
    if (nb::len(exactQubits) != 0U) {
      pythonAttribute(rebased, "add_bits",
                      "Qiskit circuit cannot add captured qubits")(exactQubits);
    }
    if (nb::len(exactClbits) != 0U) {
      pythonAttribute(rebased, "add_bits",
                      "Qiskit circuit cannot add captured classical bits")(
          exactClbits);
    }
    pythonAttribute(rebased, "compose",
                    "Qiskit circuit cannot compose a control-flow block")(
        circuit,
        nb::arg("qubits") = pythonAttribute(
            rebased, "qubits", "Qiskit rebased block has no qubits"),
        nb::arg("clbits") = pythonAttribute(
            rebased, "clbits", "Qiskit rebased block has no classical bits"),
        nb::arg("inplace") = true);
    return rebased;
  }

  void finalizeControlFlowBlocks(const nb::handle pythonCircuit) {
    const auto circuitQubits = pythonAttribute(pythonCircuit, "qubits",
                                               "Qiskit circuit has no qubits");
    const auto circuitClbits = pythonAttribute(
        pythonCircuit, "clbits", "Qiskit circuit has no classical bits");
    for (auto& pending : pendingControlFlow_) {
      auto qargs = mappedBits(circuitQubits, pending.qubits, "qubit");
      auto cargs = mappedBits(circuitClbits, pending.clbits, "classical bit");
      std::vector<nb::object> blocks;
      blocks.reserve(pending.blockWriters.size());
      for (size_t index = 0U; index < pending.blockWriters.size(); ++index) {
        try {
          auto* const writer = dynamic_cast<NativeCircuitWriter*>(
              pending.blockWriters[index].get());
          if (writer == nullptr) {
            throw std::runtime_error(
                "Qiskit control-flow blocks use an incompatible writer");
          }
          blocks.emplace_back(writer->finishImpl(true, qargs, cargs));
        } catch (const std::exception& error) {
          throw std::runtime_error(
              "Qiskit failed to finalize control-flow block " +
              std::to_string(index) + ": " + error.what());
        }
      }
      pending.blocks = std::move(blocks);
      pending.blockWriters.clear();
    }
  }

  static void collectCanonicalParameters(const nb::handle circuit,
                                         PythonParameterMap& canonical,
                                         const bool replace) {
    const auto parameters = pythonAttribute(
        circuit, "parameters", "Qiskit circuit has no parameter collection");
    std::vector<nb::object> values;
    for (const nb::handle parameter : nb::iter(parameters)) {
      values.emplace_back(nb::borrow<nb::object>(parameter));
    }
    nb::dict replacements;
    for (const auto& parameter : values) {
      const auto name = pythonStringAttribute(
          parameter, "name", "Qiskit circuit parameter has no name");
      const auto [found, inserted] = canonical.emplace(name, parameter);
      if (!inserted && !found->second.is(parameter)) {
        if (!replace) {
          throw std::runtime_error(
              "Qiskit native circuit contains distinct parameters named '" +
              name + "'");
        }
        replacements[parameter] = found->second;
      }
    }
    if (replace && nb::len(replacements) != 0U) {
      pythonAttribute(circuit, "assign_parameters",
                      "Qiskit circuit cannot replace parameters")(
          replacements, nb::arg("inplace") = true);
    }
  }

  [[nodiscard]] PythonParameterMap
  canonicalizeControlFlowParameters(const nb::handle pythonCircuit) {
    PythonParameterMap canonical;
    collectCanonicalParameters(pythonCircuit, canonical, false);
    for (auto& pending : pendingControlFlow_) {
      for (auto& block : pending.blocks) {
        collectCanonicalParameters(block, canonical, true);
      }
    }
    return canonical;
  }

  [[nodiscard]] static nb::list mappedBits(const nb::handle bits,
                                           const std::vector<uint32_t>& indices,
                                           const std::string_view kind) {
    nb::list result;
    for (const auto index : indices) {
      if (index >= nb::len(bits)) {
        throw std::runtime_error("Qiskit control flow references an invalid " +
                                 std::string(kind));
      }
      result.append(bits[index]);
    }
    return result;
  }

  [[nodiscard]] static nb::object loopIndexSet(const Loop& loop) {
    if (loop.isRange) {
      return nb::module_::import_("builtins")
          .attr("range")(loop.start, loop.stop, loop.step);
    }
    nb::list values;
    for (const auto value : loop.values) {
      values.append(nb::int_(value));
    }
    return values;
  }

  [[nodiscard]] static nb::object
  constructControlFlowOperation(const PendingControlFlow& pending,
                                const PythonClassicalBuilder& classical,
                                const PythonParameterMap& parameters) {
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    switch (pending.kind) {
    case ControlFlowKind::IfElse:
      return circuitModule.attr("IfElseOp")(
          classical.condition(pending.target), pending.blocks.front(),
          pending.blocks.size() == 2U ? pending.blocks[1]
                                      : nb::borrow<nb::object>(nb::none()));
    case ControlFlowKind::While:
      return circuitModule.attr("WhileLoopOp")(
          classical.condition(pending.target), pending.blocks.front());
    case ControlFlowKind::For: {
      nb::object parameter = nb::none();
      if (pending.loop.parameter) {
        const auto found = parameters.find(*pending.loop.parameter);
        if (found == parameters.end()) {
          throw std::runtime_error(
              "Qiskit for-loop parameter is absent from its body");
        }
        parameter = found->second;
      }
      return circuitModule.attr("ForLoopOp")(loopIndexSet(pending.loop),
                                             parameter, pending.blocks.front());
    }
    case ControlFlowKind::Switch: {
      nb::list cases;
      for (size_t index = 0U; index < pending.switchCases.size(); ++index) {
        const auto& switchCase = pending.switchCases[index];
        nb::object labels;
        if (switchCase.isDefault) {
          labels = nb::borrow<nb::object>(circuitModule.attr("CASE_DEFAULT"));
        } else if (switchCase.labels.size() == 1U) {
          labels = nb::int_(switchCase.labels.front());
        } else {
          nb::list values;
          for (const auto label : switchCase.labels) {
            values.append(nb::int_(label));
          }
          labels = std::move(values);
        }
        cases.append(nb::make_tuple(labels, pending.blocks[index]));
      }
      return circuitModule.attr("SwitchCaseOp")(
          classical.switchTarget(pending.target), cases);
    }
    case ControlFlowKind::Box:
    case ControlFlowKind::Break:
    case ControlFlowKind::Continue:
      break;
    }
    throw std::runtime_error(
        "Qiskit circuit export encountered an unsupported control-flow kind");
  }

  [[nodiscard]] std::vector<IndexedPythonInstruction>
  pendingControlFlowInstructions(const nb::handle pythonCircuit,
                                 const PythonParameterMap& parameters) const {
    std::vector<IndexedPythonInstruction> result;
    result.reserve(pendingControlFlow_.size());
    const auto data = pythonAttribute(pythonCircuit, "data",
                                      "Qiskit circuit has no instruction data");
    const auto circuitQubits = pythonAttribute(pythonCircuit, "qubits",
                                               "Qiskit circuit has no qubits");
    const auto circuitClbits = pythonAttribute(
        pythonCircuit, "clbits", "Qiskit circuit has no classical bits");
    const auto circuitInstruction =
        nb::module_::import_("qiskit.circuit").attr("CircuitInstruction");
    const PythonClassicalBuilder classical(pythonCircuit);
    for (const auto& pending : pendingControlFlow_) {
      if (pending.instructionIndex > nb::len(data)) {
        throw std::runtime_error(
            "Qiskit control-flow insertion point is invalid");
      }
      auto operation =
          constructControlFlowOperation(pending, classical, parameters);
      auto qargs = mappedBits(circuitQubits, pending.qubits, "qubit");
      auto cargs = mappedBits(circuitClbits, pending.clbits, "classical bit");
      if (pythonUnsignedAttribute(operation, "num_qubits",
                                  "Qiskit control flow has no qubit count") !=
              pending.qubits.size() ||
          pythonUnsignedAttribute(
              operation, "num_clbits",
              "Qiskit control flow has no classical-bit count") !=
              pending.clbits.size()) {
        throw std::runtime_error(
            "Qiskit control-flow operation has incompatible bit counts");
      }
      result.push_back(
          {.instructionIndex = pending.instructionIndex,
           .instruction = circuitInstruction(operation, qargs, cargs)});
    }
    return result;
  }

  static void applyPendingInstructions(
      const nb::handle pythonCircuit,
      const std::vector<IndexedPythonInstruction>& unitaryReplacements,
      const std::vector<IndexedPythonInstruction>& controlFlowInstructions) {
    auto data = pythonAttribute(pythonCircuit, "data",
                                "Qiskit circuit has no instruction data");
    for (const auto& replacement : unitaryReplacements) {
      if (replacement.instructionIndex >= nb::len(data)) {
        throw std::runtime_error(
            "Qiskit controlled-unitary replacement point is invalid");
      }
      data[replacement.instructionIndex] = replacement.instruction;
    }
    size_t inserted = 0U;
    size_t previous = 0U;
    bool first = true;
    for (const auto& pending : controlFlowInstructions) {
      if ((!first && pending.instructionIndex < previous) ||
          pending.instructionIndex + inserted > nb::len(data)) {
        throw std::runtime_error(
            "Qiskit control-flow instruction order is invalid");
      }
      pythonAttribute(data, "insert",
                      "Qiskit circuit data does not support insertion")(
          pending.instructionIndex + inserted, pending.instruction);
      previous = pending.instructionIndex;
      first = false;
      ++inserted;
    }
  }

  QkCircuit* circuit_ = nullptr;
  std::unordered_map<std::string, std::unique_ptr<OwnedParameter>>
      symbolicParameters_;
  std::vector<PendingControlledUnitary> pendingControlledUnitaries_;
  std::vector<PendingControlFlow> pendingControlFlow_;
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
