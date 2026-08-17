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
        throw std::runtime_error(
            "Qiskit circuit import supports only native, unwrapped "
            "UnitaryGate instructions");
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

  void setGlobalPhase(const double phase) override {
    const OwnedParameter parameter(phase);
    checkExitCode(qk_circuit_set_global_phase(circuit_, parameter.get()),
                  "setting global phase");
  }

  void addGate(const StandardGateMapping mapping,
               const std::vector<uint32_t>& qubits,
               const std::vector<double>& parameters) override {
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
    for (const auto parameter : parameters) {
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
                  const std::vector<uint32_t>& qubits) override {
    std::vector<QkComplex64> native;
    native.reserve(matrix.size());
    for (const auto value : matrix) {
      native.push_back({.re = value.real(), .im = value.imag()});
    }
    checkExitCode(qk_circuit_unitary(circuit_, native.data(), qubits.data(),
                                     static_cast<uint32_t>(qubits.size()),
                                     true),
                  "adding unitary");
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
    return nb::steal<nb::object>(result);
  }

private:
  QkCircuit* circuit_ = nullptr;
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
