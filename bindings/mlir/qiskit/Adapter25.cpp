/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QiskitAdapter.h"

// Qiskit's supported extension interface is intentionally exposed through the
// qiskit.h umbrella; individual generated implementation headers are private.
// NOLINTBEGIN(misc-include-cleaner)
#include <qiskit.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef MQT_QISKIT_ADAPTER_FACTORY
#define MQT_QISKIT_ADAPTER_FACTORY createAdapter25
#endif

#ifndef MQT_QISKIT_ADAPTER_EXPECTED_MAJOR
#define MQT_QISKIT_ADAPTER_EXPECTED_MAJOR 2U
#endif

#ifndef MQT_QISKIT_ADAPTER_EXPECTED_MINOR
#define MQT_QISKIT_ADAPTER_EXPECTED_MINOR 5U
#endif

#ifndef MQT_QISKIT_ADAPTER_FINAL_ONLY
#define MQT_QISKIT_ADAPTER_FINAL_ONLY 1
#endif

#ifndef MQT_QISKIT_ADAPTER_EXACT_API
#define MQT_QISKIT_ADAPTER_EXACT_API 0
#endif

#ifndef MQT_QISKIT_ADAPTER_LABEL
#define MQT_QISKIT_ADAPTER_LABEL "2.5"
#endif

namespace mqt::bindings::qiskit {
namespace {

struct PythonObjectDeleter {
  void operator()(PyObject* object) const { Py_XDECREF(object); }
};

using PythonObject = std::unique_ptr<PyObject, PythonObjectDeleter>;

[[nodiscard]] PythonObject pythonAttribute(PyObject* object, const char* name,
                                           const std::string_view error) {
  PythonObject result(PyObject_GetAttrString(object, name));
  if (!result) {
    PyErr_Clear();
    throw std::runtime_error(std::string(error));
  }
  return result;
}

[[nodiscard]] std::string pythonText(PyObject* object,
                                     const std::string_view error) {
  const PythonObject encoded(
      PyUnicode_AsEncodedString(object, "utf-8", "strict"));
  if (!encoded) {
    PyErr_Clear();
    throw std::runtime_error(std::string(error));
  }
  const char* text = PyBytes_AsString(encoded.get());
  if (text == nullptr) {
    PyErr_Clear();
    throw std::runtime_error(std::string(error));
  }
  return text;
}

[[nodiscard]] std::string pythonStringAttribute(PyObject* object,
                                                const char* name,
                                                const std::string_view error) {
  return pythonText(pythonAttribute(object, name, error).get(), error);
}

[[noreturn]] void throwPythonError(const std::string_view message) {
  PyErr_Clear();
  throw std::runtime_error(std::string(message));
}

void checkExitCode(const QkExitCode code, const std::string_view operation) {
  if (code != QkExitCode_Success) {
    throw std::runtime_error(std::string(operation) +
                             " failed with Qiskit C-API exit code " +
                             std::to_string(static_cast<unsigned int>(code)));
  }
}

using ParameterizedGateFunction = QkExitCode (*)(QkCircuit*, QkGate,
                                                 const std::uint32_t*,
                                                 const QkParam* const*);

QkExitCode addParameterizedGate(QkCircuit* circuit, const QkGate gate,
                                const std::uint32_t* qubits,
                                const QkParam* const* parameters) {
  // Qiskit 2.5.0's generated macro contains a duplicate `const` that GCC
  // rejects. Keep the vendored snapshot exact and call the same capsule slot
  // through its intended signature instead.
  const auto function = reinterpret_cast<ParameterizedGateFunction>(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      _Qk_API_Circuit[38]);
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

using ParameterNames = std::unordered_set<std::string>;

[[nodiscard]] std::shared_ptr<const ParameterNames>
parameterNames(PyObject* circuit) {
  PyObject* parameters = PyObject_GetAttrString(circuit, "parameters");
  if (parameters == nullptr) {
    throwPythonError("Qiskit circuit does not expose its parameter symbols");
  }
  PyObject* iterator = PyObject_GetIter(parameters);
  Py_DECREF(parameters);
  if (iterator == nullptr) {
    throwPythonError("Qiskit circuit parameters are not iterable");
  }
  auto result = std::make_shared<ParameterNames>();
  while (PyObject* parameter = PyIter_Next(iterator)) {
    PyObject* name = PyObject_GetAttrString(parameter, "name");
    Py_DECREF(parameter);
    if (name == nullptr) {
      Py_DECREF(iterator);
      throwPythonError("Qiskit parameter does not expose its name");
    }
    PyObject* encoded = PyUnicode_AsEncodedString(name, "utf-8", "strict");
    Py_DECREF(name);
    if (encoded == nullptr) {
      Py_DECREF(iterator);
      throwPythonError("Qiskit parameter has a non-text name");
    }
    const char* text = PyBytes_AsString(encoded);
    if (text == nullptr) {
      Py_DECREF(encoded);
      Py_DECREF(iterator);
      throwPythonError("Qiskit parameter has a non-text name");
    }
    result->emplace(text);
    Py_DECREF(encoded);
  }
  Py_DECREF(iterator);
  if (PyErr_Occurred() != nullptr) {
    throwPythonError("failed to iterate Qiskit circuit parameters");
  }
  return result;
}

[[nodiscard]] Parameter normalizeParameter(const QkParam* parameter,
                                           const ParameterNames& symbols) {
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
  // The C API requires a mutable pointer solely so qk_str_free can release it.
  // NOLINTNEXTLINE(misc-const-correctness)
  char* text = qk_param_str(parameter);
  if (text == nullptr) {
    throwPythonError("Qiskit failed to format an instruction parameter");
  }
  const auto kind = symbols.contains(text) ? ParameterKind::Symbol
                                           : ParameterKind::Expression;
  Parameter result{.kind = kind, .text = text};
  qk_str_free(text);
  return result;
}

[[nodiscard]] Parameter
normalizePythonParameter(PyObject* parameter, const ParameterNames& symbols) {
  if (PyFloat_Check(parameter) != 0 || PyLong_Check(parameter) != 0) {
    const auto number = PyFloat_AsDouble(parameter);
    if (PyErr_Occurred() != nullptr) {
      throwPythonError("Qiskit modifier exponent is not a real number");
    }
    return {.kind = ParameterKind::Number, .number = number};
  }
  const auto hasName = PyObject_HasAttrString(parameter, "name");
  if (hasName < 0) {
    throwPythonError("Qiskit failed to inspect a modifier exponent");
  }
  if (hasName != 0) {
    return {.kind = ParameterKind::Symbol,
            .text = pythonStringAttribute(
                parameter, "name",
                "Qiskit modifier exponent has an invalid symbol name")};
  }
  const PythonObject formatted(PyObject_Str(parameter));
  if (!formatted) {
    throwPythonError("Qiskit failed to format a modifier exponent");
  }
  auto text = pythonText(formatted.get(),
                         "Qiskit modifier exponent has a non-text value");
  return {.kind = symbols.contains(text) ? ParameterKind::Symbol
                                         : ParameterKind::Expression,
          .text = std::move(text)};
}

[[nodiscard]] std::uint64_t
pythonUnsignedAttribute(PyObject* object, const char* name,
                        const std::string_view error) {
  const auto attribute = pythonAttribute(object, name, error);
  const auto result = PyLong_AsUnsignedLongLong(attribute.get());
  if (PyErr_Occurred() != nullptr) {
    throwPythonError(error);
  }
  return result;
}

void appendControlModifier(PyObject* object,
                           std::vector<GateModifier>& modifiers) {
  const auto controls = pythonUnsignedAttribute(
      object, "num_ctrl_qubits",
      "Qiskit control modifier has an invalid control count");
  if (controls == 0U ||
      controls > static_cast<std::uint64_t>(
                     std::numeric_limits<std::uint32_t>::max()) ||
      controls > std::numeric_limits<std::uint64_t>::digits) {
    throw std::runtime_error(
        "Qiskit control modifiers require between 1 and 64 controls");
  }
  const auto state = pythonUnsignedAttribute(
      object, "ctrl_state", "Qiskit control modifier has an invalid state");
  const auto closedState =
      controls == std::numeric_limits<std::uint64_t>::digits
          ? std::numeric_limits<std::uint64_t>::max()
          : (std::uint64_t{1} << controls) - 1U;
  if (state != closedState) {
    throw std::runtime_error(
        "Qiskit open-control modifiers are not supported by the compiler "
        "bridge");
  }
  modifiers.push_back({.kind = GateModifierKind::Control,
                       .numControls = static_cast<std::uint32_t>(controls)});
}

void normalizePythonModifier(PyObject* modifier, const ParameterNames& symbols,
                             std::vector<GateModifier>& modifiers) {
  const auto type = pythonAttribute(modifier, "__class__",
                                    "Qiskit modifier does not expose its type");
  const auto name = pythonStringAttribute(
      type.get(), "__name__", "Qiskit modifier has an invalid type name");
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
    modifiers.push_back(
        {.kind = GateModifierKind::Power,
         .exponent = normalizePythonParameter(power.get(), symbols)});
    return;
  }
  throw std::runtime_error("unsupported Qiskit operation modifier '" + name +
                           "'");
}

void normalizePythonGate(PyObject* operation, const ParameterNames& symbols,
                         Instruction& result) {
  const auto hasBaseOperation = PyObject_HasAttrString(operation, "base_op");
  if (hasBaseOperation < 0) {
    throwPythonError("Qiskit failed to inspect an annotated operation");
  }
  if (hasBaseOperation != 0) {
    const auto base = pythonAttribute(operation, "base_op",
                                      "Qiskit annotated operation has no base");
    normalizePythonGate(base.get(), symbols, result);
    const auto modifiers = pythonAttribute(
        operation, "modifiers", "Qiskit annotated operation has no modifiers");
    const auto count = PySequence_Size(modifiers.get());
    if (count < 0) {
      throwPythonError("Qiskit operation modifiers are not a sequence");
    }
    for (Py_ssize_t index = 0; index < count; ++index) {
      const PythonObject modifier(PySequence_GetItem(modifiers.get(), index));
      if (!modifier) {
        throwPythonError("Qiskit failed to inspect an operation modifier");
      }
      normalizePythonModifier(modifier.get(), symbols, result.modifiers);
    }
    return;
  }

  const auto hasBaseGate = PyObject_HasAttrString(operation, "base_gate");
  if (hasBaseGate < 0) {
    throwPythonError("Qiskit failed to inspect a controlled gate");
  }
  if (hasBaseGate != 0) {
    const auto base = pythonAttribute(operation, "base_gate",
                                      "Qiskit controlled gate has no base");
    normalizePythonGate(base.get(), symbols, result);
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
        "Qiskit duration expressions are not supported by the compiler bridge");
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
    result.width = static_cast<std::uint32_t>(type.width);
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
normalizeExpression(const QkExprNode* expression) {
  if (expression == nullptr) {
    throw std::runtime_error("Qiskit returned a null classical expression");
  }
  auto result = std::make_unique<Expression>();
  switch (qk_expr_kind(expression)) {
  case QkExprNodeKind_Binary: {
    const auto info = qk_expr_binary_info(expression);
    result->kind = ExpressionKind::Binary;
    result->binaryOperation = normalizeBinaryOperation(info.op);
    setType(*result, info.ty);
    result->left = normalizeExpression(info.left);
    result->right = normalizeExpression(info.right);
    return result;
  }
  case QkExprNodeKind_Unary: {
    const auto info = qk_expr_unary_info(expression);
    result->kind = ExpressionKind::Unary;
    result->unaryOperation = normalizeUnaryOperation(info.op);
    setType(*result, info.ty);
    result->left = normalizeExpression(info.operand);
    return result;
  }
  case QkExprNodeKind_Cast: {
    const auto info = qk_expr_cast_info(expression);
    result->kind = ExpressionKind::Cast;
    setType(*result, info.ty);
    result->left = normalizeExpression(info.operand);
    return result;
  }
  case QkExprNodeKind_Index: {
    const auto info = qk_expr_index_info(expression);
    result->kind = ExpressionKind::Index;
    setType(*result, info.ty);
    result->left = normalizeExpression(info.target);
    result->right = normalizeExpression(info.index);
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
  case QkExprNodeKind_Var: {
    const auto* variable = qk_expr_as_var(expression);
    // The C API requires a mutable pointer solely so qk_str_free can release
    // it. NOLINTNEXTLINE(misc-const-correctness)
    char* name = qk_var_name(variable);
    if (name == nullptr) {
      throw std::runtime_error(
          "Qiskit 2.5 cannot identify bit- or register-backed variables in "
          "classical-expression trees through its C API");
    }
    result->kind = ExpressionKind::Variable;
    result->variableName = name;
    qk_str_free(name);
    setType(*result, qk_var_type_info(variable));
    return result;
  }
  case QkExprNodeKind_Stretch:
    throw std::runtime_error(
        "Qiskit stretch expressions are not supported by the compiler bridge");
  }
  throw std::runtime_error(
      "Qiskit returned an unknown classical expression node");
}

[[nodiscard]] Register normalizeRegister(const QkClassicalRegister* reg,
                                         const QkCircuit* rootCircuit) {
  // Qiskit transfers a mutable allocation that must be passed to qk_str_free.
  // NOLINTNEXTLINE(misc-const-correctness)
  char* name = qk_classical_register_name(reg);
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
    switch (parameter.kind) {
    case ParameterKind::Number:
      if (!std::isfinite(parameter.number)) {
        throw std::runtime_error(
            "cannot construct a non-finite Qiskit parameter");
      }
      value_ = qk_param_from_double(parameter.number);
      break;
    case ParameterKind::Symbol:
      if (parameter.text.empty()) {
        throw std::runtime_error("cannot construct an empty Qiskit symbol");
      }
      value_ = qk_param_new_symbol(parameter.text.c_str());
      break;
    case ParameterKind::Expression:
      throw std::runtime_error("cannot construct a composite Qiskit parameter");
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

[[nodiscard]] const std::unordered_map<std::string_view, QkGate>& gateMap() {
  static const std::unordered_map<std::string_view, QkGate> GATES = {
      {"h", QkGate_H},
      {"id", QkGate_I},
      {"x", QkGate_X},
      {"y", QkGate_Y},
      {"z", QkGate_Z},
      {"p", QkGate_Phase},
      {"r", QkGate_R},
      {"rx", QkGate_RX},
      {"ry", QkGate_RY},
      {"rz", QkGate_RZ},
      {"s", QkGate_S},
      {"sdg", QkGate_Sdg},
      {"sx", QkGate_SX},
      {"sxdg", QkGate_SXdg},
      {"t", QkGate_T},
      {"tdg", QkGate_Tdg},
      {"u", QkGate_U},
      {"u1", QkGate_U1},
      {"u2", QkGate_U2},
      {"u3", QkGate_U3},
      {"ch", QkGate_CH},
      {"cx", QkGate_CX},
      {"cy", QkGate_CY},
      {"cz", QkGate_CZ},
      {"dcx", QkGate_DCX},
      {"ecr", QkGate_ECR},
      {"swap", QkGate_Swap},
      {"iswap", QkGate_ISwap},
      {"cp", QkGate_CPhase},
      {"crx", QkGate_CRX},
      {"cry", QkGate_CRY},
      {"crz", QkGate_CRZ},
      {"cs", QkGate_CS},
      {"csdg", QkGate_CSdg},
      {"csx", QkGate_CSX},
      {"cu", QkGate_CU},
      {"cu1", QkGate_CU1},
      {"cu3", QkGate_CU3},
      {"rxx", QkGate_RXX},
      {"ryy", QkGate_RYY},
      {"rzz", QkGate_RZZ},
      {"rzx", QkGate_RZX},
      {"xx_minus_yy", QkGate_XXMinusYY},
      {"xx_plus_yy", QkGate_XXPlusYY},
      {"ccx", QkGate_CCX},
      {"ccz", QkGate_CCZ},
      {"cswap", QkGate_CSwap},
      {"rccx", QkGate_RCCX},
      {"mcx", QkGate_C3X},
      {"c3sx", QkGate_C3SX},
      {"rcccx", QkGate_RC3X},
  };
  return GATES;
}

class ControlFlowView25;

class CircuitView25 final : public CircuitView {
public:
  explicit CircuitView25(PyObject* circuit)
      : parameterNames_(parameterNames(circuit)) {
    data_.reset(PyObject_GetAttrString(circuit, "_data"));
    if (!data_) {
      throwPythonError(
          "expected a Qiskit QuantumCircuit with native CircuitData");
    }
    circuit_ = qk_circuit_borrow_from_python(data_.get());
    if (circuit_ == nullptr) {
      throwPythonError("Qiskit rejected QuantumCircuit._data");
    }
    rootCircuit_ = circuit_;
  }

  CircuitView25(PyObject* pythonCircuit, const QkCircuit* circuit,
                const QkCircuit* rootCircuit,
                const QkControlFlowInstruction* parent,
                std::shared_ptr<const ParameterNames> names)
      : circuit_(circuit), rootCircuit_(rootCircuit), parent_(parent),
        parameterNames_(std::move(names)) {
    data_.reset(PyObject_GetAttrString(pythonCircuit, "_data"));
    if (!data_) {
      throwPythonError("Qiskit control-flow block has no native CircuitData");
    }
  }

  [[nodiscard]] std::uint32_t numQubits() const override {
    return qk_circuit_num_qubits(circuit_);
  }
  [[nodiscard]] std::uint32_t numClbits() const override {
    return qk_circuit_num_clbits(circuit_);
  }
  [[nodiscard]] std::size_t numInstructions() const override {
    return qk_circuit_num_instructions(circuit_);
  }
  [[nodiscard]] std::size_t numQuantumRegisters() const override {
    return qk_circuit_num_quantum_registers(circuit_);
  }
  [[nodiscard]] std::size_t numClassicalRegisters() const override {
    return qk_circuit_num_classical_registers(circuit_);
  }

  [[nodiscard]] Register
  quantumRegister(const std::size_t index) const override {
    const auto* reg = qk_circuit_get_quantum_register(circuit_, index);
    // NOLINTNEXTLINE(misc-const-correctness)
    char* name = qk_quantum_register_name(reg);
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

  [[nodiscard]] Register
  classicalRegister(const std::size_t index) const override {
    const auto* reg = qk_circuit_get_classical_register(circuit_, index);
    // NOLINTNEXTLINE(misc-const-correctness)
    char* name = qk_classical_register_name(reg);
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
    QkParam* phase = // NOLINT(misc-const-correctness)
        qk_circuit_global_phase(circuit_);
    if (phase == nullptr) {
      throwPythonError("Qiskit failed to read the circuit global phase");
    }
    const auto result = normalizeParameter(phase, *parameterNames_);
    qk_param_free(phase);
    return result;
  }

  [[nodiscard]] Instruction
  instruction(const std::size_t index) const override {
    const auto kind =
        normalizeKind(qk_circuit_instruction_kind(circuit_, index));
    if (kind == OperationKind::Delay) {
      return {.kind = kind, .name = "delay"};
    }
    if (kind == OperationKind::ControlFlow) {
      return {.kind = kind, .name = "control_flow"};
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
    for (std::size_t parameter = 0;
         parameter < static_cast<std::size_t>(native.num_params); ++parameter) {
      result.parameters.emplace_back(normalizeParameter(
          // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          native.params[parameter], *parameterNames_));
    }
    if (result.kind == OperationKind::Unknown) {
      const auto operation = pythonOperation(index);
      normalizePythonGate(operation.get(), *parameterNames_, result);
      if (!result.modifiers.empty()) {
        result.kind = OperationKind::Gate;
      }
    }
    return result;
  }

  [[nodiscard]] std::vector<std::complex<double>>
  unitary(const std::size_t index) const override {
    const auto instructionData = instruction(index);
    if (instructionData.kind != OperationKind::Unitary) {
      throw std::runtime_error(
          "requested unitary data for a non-unitary instruction");
    }
    if (instructionData.qubits.size() >=
        std::numeric_limits<std::size_t>::digits / 2U) {
      throw std::runtime_error(
          "Qiskit unitary is too large to represent safely");
    }
    const auto entries = std::size_t{1} << (2U * instructionData.qubits.size());
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

  [[nodiscard]] std::unique_ptr<ControlFlowView>
  controlFlow(std::size_t index) const override;

private:
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  [[nodiscard]] PythonObject pythonOperation(const std::size_t index) const {
    const PythonObject instruction(
        PySequence_GetItem(data_.get(), static_cast<Py_ssize_t>(index)));
    if (!instruction) {
      throwPythonError("Qiskit failed to inspect a circuit instruction");
    }
    return pythonAttribute(instruction.get(), "operation",
                           "Qiskit circuit instruction has no operation");
  }

  PythonObject data_;
  const QkCircuit* circuit_ = nullptr;
  const QkCircuit* rootCircuit_ = circuit_;
  const QkControlFlowInstruction* parent_ = nullptr;
  std::shared_ptr<const ParameterNames> parameterNames_;
};

class ControlFlowView25 final : public ControlFlowView {
public:
  ControlFlowView25(const QkCircuit* rootCircuit, const QkCircuit* circuit,
                    const std::size_t index,
                    const QkControlFlowInstruction* parent,
                    std::shared_ptr<const ParameterNames> names,
                    PythonObject operation)
      : rootCircuit_(rootCircuit),
        controlFlow_(
            qk_circuit_get_control_flow_instruction(circuit, index, parent)),
        parameterNames_(std::move(names)), operation_(std::move(operation)) {
    if (controlFlow_ == nullptr) {
      throwPythonError("Qiskit failed to inspect a control-flow instruction");
    }
  }

  ~ControlFlowView25() override {
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

  [[nodiscard]] std::size_t numBlocks() const override {
    return qk_control_flow_num_blocks(controlFlow_);
  }

  [[nodiscard]] std::unique_ptr<CircuitView>
  block(const std::size_t index) const override {
    if (index >= numBlocks()) {
      throw std::runtime_error(
          "Qiskit control-flow block index is out of bounds");
    }
    const auto blocks = pythonAttribute(operation_.get(), "blocks",
                                        "Qiskit control flow has no blocks");
    const PythonObject block(
        PySequence_GetItem(blocks.get(), static_cast<Py_ssize_t>(index)));
    if (!block) {
      throwPythonError("Qiskit failed to inspect a control-flow block");
    }
    return std::make_unique<CircuitView25>(
        block.get(), qk_control_flow_block_circuit(controlFlow_, index),
        rootCircuit_, controlFlow_, parameterNames_);
  }

  [[nodiscard]] std::vector<std::uint32_t> qubitMap() const override {
    if (numBlocks() == 0U) {
      return {};
    }
    const auto size =
        qk_circuit_num_qubits(qk_control_flow_block_circuit(controlFlow_, 0));
    std::vector<std::uint32_t> result(size);
    if (!result.empty()) {
      std::copy_n(qk_control_flow_qubit_map(controlFlow_), size,
                  result.begin());
    }
    return result;
  }

  [[nodiscard]] std::vector<std::uint32_t> clbitMap() const override {
    if (numBlocks() == 0U) {
      return {};
    }
    const auto size =
        qk_circuit_num_clbits(qk_control_flow_block_circuit(controlFlow_, 0));
    std::vector<std::uint32_t> result(size);
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
      result.bit = static_cast<std::uint32_t>(bit.clbit);
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
      result.width = static_cast<std::uint32_t>(
          std::max<std::uint64_t>(conditionWidth, result.reg.bits.size()));
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
      result.width = static_cast<std::uint32_t>(result.reg.bits.size());
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
    for (std::size_t index = 0;
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
  std::shared_ptr<const ParameterNames> parameterNames_;
  PythonObject operation_;
};

std::unique_ptr<ControlFlowView>
CircuitView25::controlFlow(const std::size_t index) const {
  return std::make_unique<ControlFlowView25>(rootCircuit_, circuit_, index,
                                             parent_, parameterNames_,
                                             pythonOperation(index));
}

class CircuitWriter25 final : public CircuitWriter {
public:
  CircuitWriter25(const std::uint32_t looseQubits,
                  const std::uint32_t looseClbits)
      : circuit_(qk_circuit_new(looseQubits, looseClbits)) {
    if (circuit_ == nullptr) {
      throwPythonError("Qiskit failed to allocate a circuit");
    }
  }

  ~CircuitWriter25() override {
    if (circuit_ != nullptr) {
      qk_circuit_free(circuit_);
    }
  }

  void addQuantumRegister(const std::string_view name,
                          const std::uint32_t size) override {
    auto* reg = qk_quantum_register_new(size, std::string(name).c_str());
    if (reg == nullptr) {
      throwPythonError("Qiskit failed to allocate a quantum register");
    }
    qk_circuit_add_quantum_register(circuit_, reg);
    qk_quantum_register_free(reg);
  }

  void addClassicalRegister(const std::string_view name,
                            const std::uint32_t size) override {
    auto* reg = qk_classical_register_new(size, std::string(name).c_str());
    if (reg == nullptr) {
      throwPythonError("Qiskit failed to allocate a classical register");
    }
    qk_circuit_add_classical_register(circuit_, reg);
    qk_classical_register_free(reg);
  }

  void setGlobalPhase(const Parameter& parameter) override {
    std::vector<std::unique_ptr<OwnedParameter>> ownedParameters;
    const auto* value = nativeParameter(parameter, ownedParameters);
    checkExitCode(qk_circuit_set_global_phase(circuit_, value),
                  "setting global phase");
  }

  void addGate(const std::string_view name,
               const std::vector<std::uint32_t>& qubits,
               const std::vector<Parameter>& parameters) override {
    const auto gate = gateMap().find(name);
    if (gate == gateMap().end()) {
      throw std::runtime_error("Qiskit C API cannot construct gate '" +
                               std::string(name) + "'");
    }
    if (qk_gate_num_qubits(gate->second) != qubits.size() ||
        qk_gate_num_params(gate->second) != parameters.size()) {
      throw std::runtime_error("Qiskit gate '" + std::string(name) +
                               "' has incompatible arity");
    }
    if (parameters.empty()) {
      checkExitCode(
          qk_circuit_gate(circuit_, gate->second, qubits.data(), nullptr),
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
    checkExitCode(addParameterizedGate(circuit_, gate->second, qubits.data(),
                                       nativeParameters.data()),
                  "adding parameterized gate");
  }

  void addMeasure(const std::uint32_t qubit,
                  const std::uint32_t clbit) override {
    checkExitCode(qk_circuit_measure(circuit_, qubit, clbit),
                  "adding measurement");
  }

  void addReset(const std::uint32_t qubit) override {
    checkExitCode(qk_circuit_reset(circuit_, qubit), "adding reset");
  }

  void addBarrier(const std::vector<std::uint32_t>& qubits) override {
    checkExitCode(qk_circuit_barrier(circuit_, qubits.data(),
                                     static_cast<std::uint32_t>(qubits.size())),
                  "adding barrier");
  }

  void addUnitary(const std::vector<std::complex<double>>& matrix,
                  const std::vector<std::uint32_t>& qubits) override {
    std::vector<QkComplex64> native;
    native.reserve(matrix.size());
    for (const auto value : matrix) {
      native.push_back({.re = value.real(), .im = value.imag()});
    }
    checkExitCode(qk_circuit_unitary(circuit_, native.data(), qubits.data(),
                                     static_cast<std::uint32_t>(qubits.size()),
                                     true),
                  "adding unitary");
  }

  [[nodiscard]] PythonHandle* finish() override {
    if (circuit_ == nullptr) {
      throw std::runtime_error(
          "Qiskit circuit writer has already been finalized");
    }
    auto* result = qk_circuit_to_python_full(circuit_);
    circuit_ = nullptr;
    if (result == nullptr) {
      throwPythonError("Qiskit failed to create a QuantumCircuit");
    }
    return result;
  }

private:
  [[nodiscard]] const QkParam*
  nativeParameter( // NOLINT(readability-convert-member-functions-to-static)
      const Parameter& parameter,
      std::vector<std::unique_ptr<OwnedParameter>>& temporaries) {
    if (parameter.kind == ParameterKind::Symbol) {
      auto [entry, inserted] = symbols_.try_emplace(parameter.text, nullptr);
      if (inserted) {
        entry->second = std::make_unique<OwnedParameter>(parameter);
      }
      return entry->second->get();
    }
    temporaries.emplace_back(std::make_unique<OwnedParameter>(parameter));
    return temporaries.back()->get();
  }

  QkCircuit* circuit_ = nullptr;
  std::unordered_map<std::string, std::unique_ptr<OwnedParameter>> symbols_;
};

class Adapter25 final : public Adapter {
public:
  [[nodiscard]] std::unique_ptr<CircuitView>
  openCircuit(PythonHandle* circuit) const override {
    return std::make_unique<CircuitView25>(static_cast<PyObject*>(circuit));
  }

  [[nodiscard]] std::unique_ptr<CircuitWriter>
  createCircuit(const std::uint32_t looseQubits,
                const std::uint32_t looseClbits) const override {
    return std::make_unique<CircuitWriter25>(looseQubits, looseClbits);
  }
};

} // namespace

std::unique_ptr<Adapter>
MQT_QISKIT_ADAPTER_FACTORY() { // NOLINT(misc-use-internal-linkage)
  if (qk_import() < 0) {
    throwPythonError("failed to initialize the Qiskit " MQT_QISKIT_ADAPTER_LABEL
                     " C API");
  }
  const auto version = qk_api_version();
  const auto major = (version >> 24U) & 0xffU;
  const auto minor = (version >> 16U) & 0xffU;
  const auto releaseLevel = (version >> 4U) & 0xfU;
  if (major != MQT_QISKIT_ADAPTER_EXPECTED_MAJOR ||
      minor != MQT_QISKIT_ADAPTER_EXPECTED_MINOR ||
      (MQT_QISKIT_ADAPTER_EXACT_API != 0 && version != QISKIT_VERSION_HEX) ||
      (MQT_QISKIT_ADAPTER_FINAL_ONLY != 0 &&
       releaseLevel != QISKIT_RELEASE_LEVEL_FINAL)) {
    throw std::runtime_error("Qiskit C-API capsule version does not match the "
                             "selected " MQT_QISKIT_ADAPTER_LABEL " adapter");
  }
  return std::make_unique<Adapter25>();
}

} // namespace mqt::bindings::qiskit
// NOLINTEND(misc-include-cleaner)
