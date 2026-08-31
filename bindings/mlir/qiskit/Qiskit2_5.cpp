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

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSwitch.h>

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
#include <bit>
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

constexpr size_t MAX_EXPRESSION_DEPTH = 64U;
constexpr size_t MAX_EXPRESSION_NODES = 4096U;
constexpr size_t MAX_ANNOTATED_OPERATION_DEPTH = 64U;

[[nodiscard]] static nb::object pythonAttribute(const nb::handle object,
                                                const char* name,
                                                const std::string_view error) {
  try {
    return nb::borrow<nb::object>(object).attr(name);
  } catch (const nb::python_error&) {
    throw std::runtime_error(std::string(error));
  }
}

[[nodiscard]] static std::string pythonText(const nb::handle object,
                                            const std::string_view error) {
  try {
    return nb::cast<std::string>(nb::str(object));
  } catch (const nb::python_error&) {
    throw std::runtime_error(std::string(error));
  }
}

[[nodiscard]] static std::string
pythonStringAttribute(const nb::handle object, const char* name,
                      const std::string_view error) {
  return pythonText(pythonAttribute(object, name, error), error);
}

[[nodiscard]] static uint64_t
pythonUnsignedAttribute(const nb::handle object, const char* name,
                        const std::string_view error) {
  const auto attribute = pythonAttribute(object, name, error);
  uint64_t result = 0;
  if (!nb::try_cast(attribute, result)) {
    throw std::runtime_error(std::string(error));
  }
  return result;
}

[[noreturn]] static void throwPythonError(const std::string_view message) {
  const nb::python_error error;
  throw std::runtime_error(std::string(message) + ": " + error.what());
}

[[noreturn]] static void throwPythonError(const std::string_view message,
                                          const nb::python_error& error) {
  throw std::runtime_error(std::string(message) + ": " + error.what());
}

static void checkExitCode(const QkExitCode code,
                          const std::string_view operation) {
  if (code != QkExitCode_Success) {
    throw std::runtime_error(std::string(operation) +
                             " failed with Qiskit C API exit code " +
                             std::to_string(static_cast<unsigned int>(code)));
  }
}

using ParameterizedGateFunction = QkExitCode (*)(QkCircuit*, QkGate,
                                                 const uint32_t*,
                                                 const QkParam* const*);

static QkExitCode addParameterizedGate(QkCircuit* circuit, const QkGate gate,
                                       const uint32_t* qubits,
                                       const QkParam* const* parameters) {
  // Qiskit 2.5.0's generated macro contains a duplicate `const` that GCC
  // rejects. Keep the vendored snapshot exact and call the same capsule slot
  // through its intended signature instead.
  const auto function =
      reinterpret_cast<ParameterizedGateFunction>(_Qk_API_Circuit[38]);
  return function(circuit, gate, qubits, parameters);
}

[[nodiscard]] static OperationKind normalizeKind(const QkOperationKind kind) {
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

[[nodiscard]] static Parameter normalizeParameter(const QkParam* parameter) {
  const auto number = qk_param_as_real(parameter);
  if (std::isfinite(number)) {
    auto* numeric = qk_param_from_double(number);
    if (numeric == nullptr) {
      throwPythonError("Qiskit failed to inspect a numeric parameter");
    }
    const auto isNumber = qk_param_equal(parameter, numeric);
    qk_param_free(numeric);
    if (isNumber) {
      return Parameter::number(number);
    }
  }
  throw std::runtime_error(
      "Qiskit's native API does not expose symbolic parameter-expression "
      "structure");
}

[[nodiscard]] static Parameter
normalizePythonParameterLeaf(const nb::handle parameter) {
  double number = 0.0;
  if (nb::try_cast(parameter, number)) {
    if (!std::isfinite(number)) {
      throw std::runtime_error("Qiskit returned a non-finite parameter");
    }
    return Parameter::number(number);
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
    return Parameter::number(complexNumber.real());
  }

  if (!nb::hasattr(parameter, "name")) {
    throw std::runtime_error(
        "Qiskit parameter expression contains an unsupported operand");
  }
  auto name = pythonStringAttribute(
      parameter, "name", "Qiskit parameter has an invalid symbol name");
  if (name.empty()) {
    throw std::runtime_error("Qiskit parameter has an empty symbol name");
  }
  if (name.find('\0') != std::string::npos) {
    throw std::runtime_error(
        "Qiskit parameter names cannot contain null characters");
  }
  const auto vectorElement =
      nb::module_::import_("qiskit.circuit").attr("ParameterVectorElement");
  if (!nb::isinstance(parameter, vectorElement)) {
    return Parameter::symbol(std::move(name));
  }

  const auto vector = pythonAttribute(
      parameter, "vector", "Qiskit parameter-vector element has no vector");
  auto groupName = pythonStringAttribute(
      vector, "name", "Qiskit parameter vector has an invalid name");
  auto groupIdentity =
      pythonText(pythonAttribute(vector, "uuid",
                                 "Qiskit parameter vector has no identity"),
                 "Qiskit parameter vector has an invalid identity");
  const auto groupIndex = pythonUnsignedAttribute(
      parameter, "index",
      "Qiskit parameter-vector element has an invalid index");
  size_t groupSize = 0U;
  try {
    groupSize = nb::len(vector);
  } catch (const nb::python_error& error) {
    throwPythonError("Qiskit parameter vector has an invalid size", error);
  }
  if (groupIdentity.empty() || groupIdentity.find('\0') != std::string::npos ||
      groupName.find('\0') != std::string::npos ||
      name != groupName + "[" + std::to_string(groupIndex) + "]") {
    throw std::runtime_error(
        "Qiskit parameter-vector element has invalid group metadata");
  }
  return Parameter::symbol(std::move(name),
                           ParameterGroup{.identity = std::move(groupIdentity),
                                          .name = std::move(groupName),
                                          .index = groupIndex,
                                          .size = groupSize});
}

namespace {
struct ParsedParameter {
  Parameter value;
  size_t depth = 1U;
};
} // namespace

[[noreturn]] static void throwParameterExpressionSizeError() {
  throw std::runtime_error(
      "Qiskit parameter expression exceeds the supported " +
      std::to_string(MAX_PARAMETER_EXPRESSION_NODES) + "-node size");
}

[[noreturn]] static void throwParameterExpressionDepthError() {
  throw std::runtime_error(
      "Qiskit parameter expression exceeds the supported " +
      std::to_string(MAX_PARAMETER_EXPRESSION_DEPTH) + "-level nesting depth");
}

static void countParameterExpressionNode(size_t& nodeCount) {
  if (nodeCount >= MAX_PARAMETER_EXPRESSION_NODES) {
    throwParameterExpressionSizeError();
  }
  ++nodeCount;
}

[[nodiscard]] static ParsedParameter
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

[[nodiscard]] static Parameter makeUnaryParameter(const UnaryParameterKind kind,
                                                  Parameter operand) {
  return Parameter::unary(kind, std::move(operand));
}

[[nodiscard]] static Parameter
makeBinaryParameter(const BinaryParameterKind kind, Parameter lhs,
                    Parameter rhs) {
  return Parameter::binary(kind, std::move(lhs), std::move(rhs));
}

[[nodiscard]] static std::string parameterOpcode(const nb::handle replayEntry) {
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

[[nodiscard]] static bool
isUnaryParameterOpcode(const std::string_view opcode) {
  return opcode == "NEG" || opcode == "SIN" || opcode == "COS" ||
         opcode == "TAN" || opcode == "ASIN" || opcode == "ACOS" ||
         opcode == "ATAN" || opcode == "EXP" || opcode == "LOG" ||
         opcode == "ABS" || opcode == "CONJ" || opcode == "CONJUGATE";
}

[[nodiscard]] static UnaryParameterKind
unaryParameterKind(const std::string_view opcode) {
  if (opcode == "NEG") {
    return UnaryParameterKind::Negate;
  }
  if (opcode == "SIN") {
    return UnaryParameterKind::Sin;
  }
  if (opcode == "COS") {
    return UnaryParameterKind::Cos;
  }
  if (opcode == "TAN") {
    return UnaryParameterKind::Tan;
  }
  if (opcode == "ASIN") {
    return UnaryParameterKind::ArcSin;
  }
  if (opcode == "ACOS") {
    return UnaryParameterKind::ArcCos;
  }
  if (opcode == "ATAN") {
    return UnaryParameterKind::ArcTan;
  }
  if (opcode == "EXP") {
    return UnaryParameterKind::Exp;
  }
  if (opcode == "LOG") {
    return UnaryParameterKind::Log;
  }
  if (opcode == "ABS") {
    return UnaryParameterKind::Abs;
  }
  return UnaryParameterKind::Conjugate;
}

[[nodiscard]] static bool
isBinaryParameterOpcode(const std::string_view opcode) {
  return opcode == "ADD" || opcode == "SUB" || opcode == "MUL" ||
         opcode == "DIV" || opcode == "POW" || opcode == "RSUB" ||
         opcode == "RDIV" || opcode == "RPOW";
}

[[nodiscard]] static BinaryParameterKind
binaryParameterKind(const std::string_view opcode) {
  if (opcode == "ADD") {
    return BinaryParameterKind::Add;
  }
  if (opcode == "SUB" || opcode == "RSUB") {
    return BinaryParameterKind::Subtract;
  }
  if (opcode == "MUL") {
    return BinaryParameterKind::Multiply;
  }
  if (opcode == "DIV" || opcode == "RDIV") {
    return BinaryParameterKind::Divide;
  }
  return BinaryParameterKind::Power;
}

[[nodiscard]] static Parameter
normalizePythonParameter(const nb::handle parameter) {
  if (nb::hasattr(parameter, "name")) {
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

static void appendControlModifier(const nb::handle object,
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

[[nodiscard]] static nb::object terminalPythonGate(const nb::handle operation,
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

[[nodiscard]] static bool isPythonUnitaryGate(const nb::handle operation) {
  const auto terminal = terminalPythonGate(operation);
  const auto unitaryGate =
      nb::module_::import_("qiskit.circuit.library").attr("UnitaryGate");
  return nb::isinstance(terminal, unitaryGate);
}

static void normalizePythonModifier(const nb::handle modifier,
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

static void normalizePythonGate(const nb::handle operation, Instruction& result,
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

namespace {
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
} // namespace

[[nodiscard]] static const auto& gateMap() {
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

[[nodiscard]] static const VersionGate*
versionGate(const std::string_view name) {
  for (const auto& gate : gateMap()) {
    if (gate.name == name) {
      return &gate;
    }
  }
  return nullptr;
}

[[nodiscard]] static const VersionGate*
versionGate(const StandardGateMapping mapping) {
  for (const auto& gate : gateMap()) {
    if (gate.translation == mapping) {
      return &gate;
    }
  }
  return nullptr;
}

[[nodiscard]] static std::optional<StandardGateMapping>
standardGateMapping(const std::string_view name) {
  const auto* gate = versionGate(name);
  return gate == nullptr ? std::nullopt : std::optional{gate->translation};
}

namespace {
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
                          nb::object instruction,
                          nb::object containingPythonCircuit)
      : rootCircuit_(rootCircuit), circuit_(circuit), parent_(parent),
        instruction_(std::move(instruction)),
        operation_(pythonAttribute(
            instruction_, "operation",
            "Qiskit circuit instruction has no control-flow operation")),
        containingPythonCircuit_(std::move(containingPythonCircuit)),
        controlFlow_(
            qk_circuit_get_control_flow_instruction(circuit, index, parent)) {
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
    const auto condition = pythonAttribute(
        operation_, "condition", "Qiskit control flow has no condition");
    const auto expressionModule =
        nb::module_::import_("qiskit.circuit.classical.expr");
    if (nb::isinstance(condition, expressionModule.attr("Expr"))) {
      return normalizePythonTarget(condition);
    }

    if (!nb::isinstance<nb::tuple>(condition) || nb::len(condition) != 2U) {
      throw std::runtime_error("Qiskit control-flow condition has an invalid "
                               "shape");
    }
    uint64_t expected = 0U;
    if (!nb::try_cast(condition[1], expected)) {
      throw std::runtime_error(
          "Qiskit control-flow condition has an invalid value");
    }

    auto result = normalizePythonTarget(condition[0]);
    if (result.kind == ClassicalTargetKind::ClassicalBit) {
      if (expected > 1U) {
        throw std::runtime_error(
            "Qiskit classical-bit condition must compare against zero or one");
      }
      result.expectedBit = expected != 0U;
      return result;
    }
    if (result.kind == ClassicalTargetKind::ClassicalRegister) {
      result.width = static_cast<uint32_t>(
          std::max<size_t>(result.reg.bits.size(), std::bit_width(expected)));
      result.expectedRegister = expected;
      return result;
    }
    throw std::runtime_error("Qiskit control flow has an unknown condition "
                             "target");
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
      if (symbol.ty != QkSymbolType_Standalone &&
          symbol.ty != QkSymbolType_Element) {
        if (symbol.name != nullptr) {
          qk_str_free(symbol.name);
        }
        throw std::runtime_error(
            "Qiskit for-loop parameter has an unknown symbol type");
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
        const auto* parameterSymbol = parameter.getSymbol();
        if (parameterSymbol == nullptr) {
          throw std::runtime_error("Qiskit for-loop parameter is not a symbol");
        }
        const auto nativeIsElement = symbol.ty == QkSymbolType_Element;
        const auto& group = parameterSymbol->group;
        if (nativeIsElement != group.has_value() ||
            (group ? group->name : parameterSymbol->name) != nativeName ||
            (group && group->index != symbol.index)) {
          throw std::runtime_error(
              "Qiskit Python and native loop-parameter metadata do not match");
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
    // Qiskit 2.5's native switch-target accessors abort for expressions.
    return normalizePythonTarget(
        pythonAttribute(operation_, "target", "Qiskit switch has no target"));
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
  [[nodiscard]] ClassicalTarget
  normalizePythonTarget(const nb::handle target) const {
    ClassicalTarget result;
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    if (nb::isinstance(target, circuitModule.attr("Clbit"))) {
      result.kind = ClassicalTargetKind::ClassicalBit;
      result.bit = rootClbitIndex(target);
      return result;
    }
    if (nb::isinstance(target, circuitModule.attr("ClassicalRegister"))) {
      const auto size = nb::len(target);
      if (size == 0U || size > 64U) {
        throw std::runtime_error(
            "Qiskit classical targets require between 1 and 64 bits");
      }
      result.kind = ClassicalTargetKind::ClassicalRegister;
      result.reg.name = pythonStringAttribute(
          target, "name", "Qiskit classical target register has no name");
      result.reg.bits.reserve(size);
      for (const nb::handle bit : nb::iter(target)) {
        result.reg.bits.push_back(rootClbitIndex(bit));
      }
      result.width = static_cast<uint32_t>(size);
      return result;
    }
    const auto expressionModule =
        nb::module_::import_("qiskit.circuit.classical.expr");
    if (nb::isinstance(target, expressionModule.attr("Expr"))) {
      result.kind = ClassicalTargetKind::Expression;
      size_t nodeCount = 0U;
      result.expression = normalizePythonExpressionOnly(target, nodeCount);
      return result;
    }
    throw std::runtime_error("Qiskit classical target has an unknown type");
  }

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
    // Conditions and switch targets refer to bits in the containing circuit.
    // The current block-operand map is not an identity source: a bit can be
    // absent from all blocks, and a nested map can still use a local index.
    // Resolve the Python bit in the containing circuit, then use the enclosing
    // control flow's native map when that circuit is itself a nested block.
    try {
      const auto findBit = pythonAttribute(
          containingPythonCircuit_, "find_bit",
          "Qiskit containing circuit cannot resolve expression variables");
      const auto location = findBit(bit);
      const auto localIndex = pythonUnsignedAttribute(
          location, "index",
          "Qiskit expression variable has an invalid circuit index");
      if (localIndex >= qk_circuit_num_clbits(circuit_)) {
        throw std::runtime_error(
            "Qiskit expression variable has an invalid circuit index");
      }
      if (parent_ == nullptr) {
        return static_cast<uint32_t>(localIndex);
      }

      const auto* const parentMap = qk_control_flow_clbit_map(parent_);
      if (parentMap == nullptr) {
        throw std::runtime_error(
            "Qiskit enclosing control flow has no classical-bit capture map");
      }
      return parentMap[localIndex];
    } catch (const nb::python_error& error) {
      throwPythonError(
          "Qiskit expression variable is absent from its containing circuit",
          error);
    }
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
    const auto operation =
        llvm::StringSwitch<std::optional<BinaryOperation>>(name)
            .Case("BIT_AND", BinaryOperation::BitAnd)
            .Case("BIT_OR", BinaryOperation::BitOr)
            .Case("BIT_XOR", BinaryOperation::BitXor)
            .Case("LOGIC_AND", BinaryOperation::LogicAnd)
            .Case("LOGIC_OR", BinaryOperation::LogicOr)
            .Case("EQUAL", BinaryOperation::Equal)
            .Case("NOT_EQUAL", BinaryOperation::NotEqual)
            .Case("LESS", BinaryOperation::Less)
            .Case("LESS_EQUAL", BinaryOperation::LessEqual)
            .Case("GREATER", BinaryOperation::Greater)
            .Case("GREATER_EQUAL", BinaryOperation::GreaterEqual)
            .Case("SHIFT_LEFT", BinaryOperation::ShiftLeft)
            .Case("SHIFT_RIGHT", BinaryOperation::ShiftRight)
            .Case("ADD", BinaryOperation::Add)
            .Case("SUB", BinaryOperation::Subtract)
            .Case("MUL", BinaryOperation::Multiply)
            .Case("DIV", BinaryOperation::Divide)
            .Default(std::nullopt);
    if (!operation) {
      throw std::runtime_error(
          "Qiskit expression has an unknown Python binary operation");
    }
    return *operation;
  }

  [[nodiscard]] static UnaryOperation
  pythonUnaryOperation(const std::string_view name) {
    const auto operation =
        llvm::StringSwitch<std::optional<UnaryOperation>>(name)
            .Case("BIT_NOT", UnaryOperation::BitNot)
            .Case("LOGIC_NOT", UnaryOperation::LogicNot)
            .Case("NEGATE", UnaryOperation::Negate)
            .Default(std::nullopt);
    if (!operation) {
      throw std::runtime_error(
          "Qiskit expression has an unknown Python unary operation");
    }
    return *operation;
  }

  [[nodiscard]] std::unique_ptr<Expression>
  normalizePythonExpressionOnly(const nb::handle pythonExpression,
                                size_t& nodeCount,
                                const size_t depth = 0U) const {
    if (depth >= MAX_EXPRESSION_DEPTH) {
      throw std::runtime_error(
          "Qiskit classical expressions exceed the nesting limit of 64");
    }
    if (nodeCount >= MAX_EXPRESSION_NODES) {
      throw std::runtime_error(
          "Qiskit classical expressions exceed the node limit of 4096");
    }
    ++nodeCount;
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
      case ClassicalType::Bool: {
        uint64_t boolValue = 0U;
        if (!nb::try_cast(value, boolValue) || boolValue > 1U) {
          throw std::runtime_error(
              "Qiskit Boolean expression has an invalid value");
        }
        result->boolValue = boolValue != 0U;
        break;
      }
      case ClassicalType::Uint:
        if (!nb::try_cast(value, result->uintValue) ||
            (result->width < 64U &&
             result->uintValue >= (uint64_t{1} << result->width))) {
          throw std::runtime_error(
              "Qiskit Uint literal does not fit its declared width");
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
          nodeCount, depth + 1U);
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
          nodeCount, depth + 1U);
      result->right = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "right",
                          "Qiskit binary expression has no right operand"),
          nodeCount, depth + 1U);
      return result;
    }
    if (className == "Cast") {
      result->kind = ExpressionKind::Cast;
      result->left = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "operand",
                          "Qiskit cast expression has no operand"),
          nodeCount, depth + 1U);
      return result;
    }
    if (className == "Index") {
      result->kind = ExpressionKind::Index;
      result->left = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "target",
                          "Qiskit index expression has no target"),
          nodeCount, depth + 1U);
      result->right = normalizePythonExpressionOnly(
          pythonAttribute(pythonExpression, "index",
                          "Qiskit index expression has no index"),
          nodeCount, depth + 1U);
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

  const QkCircuit* rootCircuit_ = nullptr;
  const QkCircuit* circuit_ = nullptr;
  const QkControlFlowInstruction* parent_ = nullptr;
  nb::object instruction_;
  nb::object operation_;
  nb::object containingPythonCircuit_;
  QkControlFlowInstruction* controlFlow_ = nullptr;
};
} // namespace

std::unique_ptr<ControlFlowReader>
NativeCircuitReader::controlFlow(const size_t index) const {
  return std::make_unique<NativeControlFlowReader>(
      rootCircuit_, circuit_, index, parent_,
      nb::borrow<nb::object>(data_[index]), pythonCircuit_);
}

namespace {
class PythonClassicalBuilder final {
public:
  explicit PythonClassicalBuilder(const nb::handle circuit)
      : clbits_(pythonAttribute(circuit, "clbits",
                                "Qiskit circuit has no classical bits")),
        cregs_(pythonAttribute(circuit, "cregs",
                               "Qiskit circuit has no classical registers")),
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
    if (reg.name.empty()) {
      return std::nullopt;
    }
    for (const nb::handle candidateHandle : nb::iter(cregs_)) {
      auto candidate = nb::borrow<nb::object>(candidateHandle);
      if (pythonStringAttribute(candidate, "name",
                                "Qiskit classical register has no name") ==
          reg.name) {
        return candidate;
      }
    }
    return std::nullopt;
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

  nb::object clbits_;
  nb::object cregs_;
  nb::object expressionModule_;
  nb::object typesModule_;
};

struct NativeSymbol {
  NativeSymbol(const std::string_view name,
               std::optional<ParameterGroup> sourceGroup)
      : group(std::move(sourceGroup)), parameter(name) {}

  std::optional<ParameterGroup> group;
  OwnedParameter parameter;
};

using NativeSymbolTable = llvm::StringMap<NativeSymbol>;
using PythonParameterGroups = llvm::StringMap<nb::object>;

class NativeCircuitWriter final : public CircuitWriter {
public:
  NativeCircuitWriter(const uint32_t looseQubits, const uint32_t looseClbits,
                      std::shared_ptr<NativeSymbolTable> symbols)
      : circuit_(qk_circuit_new(looseQubits, looseClbits)),
        symbols_(std::move(symbols)) {
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

  void
  addControlFlow(const ControlFlowKind kind, ClassicalTarget target, Loop loop,
                 std::vector<SwitchCase> switchCases,
                 std::vector<std::unique_ptr<CircuitWriter>> blocks) override {
    const bool validBlockCount = [&]() {
      switch (kind) {
      case ControlFlowKind::IfElse:
        return blocks.size() == 1U || blocks.size() == 2U;
      case ControlFlowKind::While:
      case ControlFlowKind::For:
        return blocks.size() == 1U;
      case ControlFlowKind::Switch:
        return !blocks.empty() && blocks.size() == switchCases.size();
      case ControlFlowKind::Box:
      case ControlFlowKind::Break:
      case ControlFlowKind::Continue:
        return false;
      }
      return false;
    }();
    if (!validBlockCount) {
      throw std::runtime_error(
          "Qiskit control flow has an unexpected number of blocks");
    }
    const auto numQubits = qk_circuit_num_qubits(circuit_);
    const auto numClbits = qk_circuit_num_clbits(circuit_);
    for (const auto& block : blocks) {
      const auto* const native =
          dynamic_cast<const NativeCircuitWriter*>(block.get());
      if (native == nullptr) {
        throw std::runtime_error(
            "Qiskit control-flow blocks use an incompatible writer");
      }
      if (native->circuit_ == nullptr ||
          qk_circuit_num_qubits(native->circuit_) != numQubits ||
          qk_circuit_num_clbits(native->circuit_) != numClbits) {
        throw std::runtime_error(
            "Qiskit control-flow block has incompatible bit counts");
      }
    }
    const auto instructionIndex = qk_circuit_num_instructions(circuit_);
    checkExitCode(qk_circuit_barrier(circuit_, nullptr, 0U),
                  "adding control-flow placeholder");
    pendingControlFlow_.push_back({.instructionIndex = instructionIndex,
                                   .kind = kind,
                                   .target = std::move(target),
                                   .loop = std::move(loop),
                                   .switchCases = std::move(switchCases),
                                   .blockWriters = std::move(blocks)});
  }

  [[nodiscard]] nb::object finish() override {
    PythonParameterGroups groups;
    return finishImpl(false, nb::none(), nb::none(), groups);
  }

private:
  [[nodiscard]] nb::object finishImpl(const bool rebase,
                                      const nb::handle exactQubits,
                                      const nb::handle exactClbits,
                                      PythonParameterGroups& groups) {
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
      replacePendingControlledUnitaries(pythonCircuit);
      restoreParameterGroups(pythonCircuit, *symbols_, groups);
      replacePendingControlFlow(pythonCircuit, groups);
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
  };

  static void restoreParameterGroups(const nb::handle circuit,
                                     const NativeSymbolTable& symbols,
                                     PythonParameterGroups& groups) {
    if (!std::ranges::any_of(symbols, [](const auto& entry) {
          return entry.second.group.has_value();
        })) {
      return;
    }

    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    const auto parameterVector = circuitModule.attr("ParameterVector");
    const auto parameterVectorElement =
        circuitModule.attr("ParameterVectorElement");
    nb::dict replacements;
    const auto parameters = pythonAttribute(
        circuit, "parameters", "Qiskit circuit has no parameter collection");
    for (const nb::handle parameter : nb::iter(parameters)) {
      const auto name = pythonStringAttribute(
          parameter, "name", "Qiskit circuit parameter has no name");
      const auto symbol = symbols.find(name);
      if (symbol == symbols.end() || !symbol->second.group) {
        continue;
      }
      const auto& metadata = *symbol->second.group;
      const auto [group, inserted] = groups.try_emplace(metadata.identity);
      if (inserted) {
        group->second = parameterVector(metadata.name, metadata.size);
      }
      replacements[parameter] =
          parameterVectorElement(group->second, metadata.index);
    }
    if (nb::len(replacements) == 0U) {
      return;
    }
    pythonAttribute(circuit, "assign_parameters",
                    "Qiskit circuit cannot replace output parameters")(
        replacements, nb::arg("inplace") = true, nb::arg("flat_input") = true);
  }

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

  [[nodiscard]] static nb::object rebaseCircuit(const nb::handle circuit,
                                                const nb::handle exactQubits,
                                                const nb::handle exactClbits) {
    auto rebased = nb::module_::import_("qiskit.circuit")
                       .attr("QuantumCircuit")(exactQubits, exactClbits);
    pythonAttribute(rebased, "compose",
                    "Qiskit circuit cannot compose a control-flow block")(
        circuit, nb::arg("inplace") = true, nb::arg("copy") = false);
    return rebased;
  }

  [[nodiscard]] static nb::object loopIndexSet(const Loop& loop) {
    if (!loop.isRange) {
      throw std::runtime_error(
          "Qiskit circuit export supports only range-based for loops");
    }
    return nb::module_::import_("builtins")
        .attr("range")(loop.start, loop.stop, loop.step);
  }

  [[nodiscard]] static nb::object loopParameter(const Loop& loop,
                                                const nb::handle body) {
    if (!loop.parameter) {
      return nb::borrow<nb::object>(nb::none());
    }
    const auto* symbol = loop.parameter->getSymbol();
    if (symbol == nullptr) {
      throw std::runtime_error(
          "Qiskit for-loop parameter has invalid symbol metadata");
    }
    const auto parameterName =
        symbol->group ? symbol->group->name + "[" +
                            std::to_string(symbol->group->index) + "]"
                      : symbol->name;
    return pythonAttribute(body, "get_parameter",
                           "Qiskit circuit cannot find its loop parameter")(
        parameterName);
  }

  [[nodiscard]] static nb::object constructControlFlowOperation(
      const PendingControlFlow& pending, const std::vector<nb::object>& blocks,
      const PythonClassicalBuilder& classical, const nb::handle circuitModule) {
    switch (pending.kind) {
    case ControlFlowKind::IfElse:
      return circuitModule.attr("IfElseOp")(
          classical.condition(pending.target), blocks.front(),
          blocks.size() == 2U ? blocks[1] : nb::borrow<nb::object>(nb::none()));
    case ControlFlowKind::While:
      return circuitModule.attr("WhileLoopOp")(
          classical.condition(pending.target), blocks.front());
    case ControlFlowKind::For:
      return circuitModule.attr("ForLoopOp")(
          loopIndexSet(pending.loop),
          loopParameter(pending.loop, blocks.front()), blocks.front());
    case ControlFlowKind::Switch: {
      nb::list cases;
      for (size_t index = 0U; index < pending.switchCases.size(); ++index) {
        const auto& switchCase = pending.switchCases[index];
        nb::object labels;
        if (switchCase.isDefault) {
          labels = nb::borrow<nb::object>(circuitModule.attr("CASE_DEFAULT"));
        } else {
          if (switchCase.labels.size() != 1U) {
            throw std::runtime_error(
                "Qiskit circuit export requires one label per switch case");
          }
          labels = nb::int_(switchCase.labels.front());
        }
        cases.append(nb::make_tuple(labels, blocks[index]));
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

  void replacePendingControlFlow(const nb::handle pythonCircuit,
                                 PythonParameterGroups& groups) {
    auto data = pythonAttribute(pythonCircuit, "data",
                                "Qiskit circuit has no instruction data");
    const auto circuitQubits = pythonAttribute(pythonCircuit, "qubits",
                                               "Qiskit circuit has no qubits");
    const auto circuitClbits = pythonAttribute(
        pythonCircuit, "clbits", "Qiskit circuit has no classical bits");
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    const auto circuitInstruction = circuitModule.attr("CircuitInstruction");
    const PythonClassicalBuilder classical(pythonCircuit);
    for (auto& pending : pendingControlFlow_) {
      if (pending.instructionIndex >= nb::len(data)) {
        throw std::runtime_error("Qiskit control-flow placeholder is missing");
      }
      std::vector<nb::object> blocks;
      blocks.reserve(pending.blockWriters.size());
      for (const auto& blockWriter : pending.blockWriters) {
        auto* const writer =
            dynamic_cast<NativeCircuitWriter*>(blockWriter.get());
        if (writer == nullptr) {
          throw std::runtime_error(
              "Qiskit control-flow blocks use an incompatible writer");
        }
        blocks.emplace_back(
            writer->finishImpl(true, circuitQubits, circuitClbits, groups));
      }
      pending.blockWriters.clear();
      auto operation = constructControlFlowOperation(pending, blocks, classical,
                                                     circuitModule);
      if (pythonUnsignedAttribute(operation, "num_qubits",
                                  "Qiskit control flow has no qubit count") !=
              nb::len(circuitQubits) ||
          pythonUnsignedAttribute(
              operation, "num_clbits",
              "Qiskit control flow has no classical-bit count") !=
              nb::len(circuitClbits)) {
        throw std::runtime_error(
            "Qiskit control-flow operation has incompatible bit counts");
      }
      data[pending.instructionIndex] =
          circuitInstruction(operation, circuitQubits, circuitClbits);
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
    if (const auto* number = parameter.getNumber()) {
      ownedParameters.emplace_back(
          std::make_unique<OwnedParameter>(number->value));
      return ownedParameters.back()->get();
    }
    if (const auto* symbol = parameter.getSymbol()) {
      if (symbol->name.empty()) {
        throw std::runtime_error(
            "cannot export a symbolic parameter without a name");
      }
      return symbols_->try_emplace(symbol->name, symbol->name, symbol->group)
          .first->second.parameter.get();
    }

    auto output = std::make_unique<OwnedParameter>();
    QkExitCode result = QkExitCode_Success;
    if (const auto* unary = parameter.getUnary()) {
      const auto* operand = nativeParameter(*unary->operand, ownedParameters,
                                            nodeCount, depth + 1U);
      switch (unary->operation) {
      case UnaryParameterKind::Negate:
        result = qk_param_neg(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Sin:
        result = qk_param_sin(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Cos:
        result = qk_param_cos(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Tan:
        result = qk_param_tan(output->getMutable(), operand);
        break;
      case UnaryParameterKind::ArcSin:
        result = qk_param_asin(output->getMutable(), operand);
        break;
      case UnaryParameterKind::ArcCos:
        result = qk_param_acos(output->getMutable(), operand);
        break;
      case UnaryParameterKind::ArcTan:
        result = qk_param_atan(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Exp:
        result = qk_param_exp(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Log:
        result = qk_param_log(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Abs:
        result = qk_param_abs(output->getMutable(), operand);
        break;
      case UnaryParameterKind::Conjugate:
        result = qk_param_conjugate(output->getMutable(), operand);
        break;
      }
    } else if (const auto* binary = parameter.getBinary()) {
      const auto* left = nativeParameter(*binary->left, ownedParameters,
                                         nodeCount, depth + 1U);
      const auto* right = nativeParameter(*binary->right, ownedParameters,
                                          nodeCount, depth + 1U);
      switch (binary->operation) {
      case BinaryParameterKind::Add:
        result = qk_param_add(output->getMutable(), left, right);
        break;
      case BinaryParameterKind::Subtract:
        result = qk_param_sub(output->getMutable(), left, right);
        break;
      case BinaryParameterKind::Multiply:
        result = qk_param_mul(output->getMutable(), left, right);
        break;
      case BinaryParameterKind::Divide:
        result = qk_param_div(output->getMutable(), left, right);
        break;
      case BinaryParameterKind::Power:
        result = qk_param_pow(output->getMutable(), left, right);
        break;
      }
    } else {
      throw std::runtime_error("unknown normalized parameter expression");
    }
    checkExitCode(result, "constructing a parameter expression");
    const auto* value = output->get();
    ownedParameters.push_back(std::move(output));
    return value;
  }

  QkCircuit* circuit_ = nullptr;
  std::vector<PendingControlledUnitary> pendingControlledUnitaries_;
  std::vector<PendingControlFlow> pendingControlFlow_;
  std::shared_ptr<NativeSymbolTable> symbols_;
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
    return std::make_unique<NativeCircuitWriter>(looseQubits, looseClbits,
                                                 symbols_);
  }

private:
  std::shared_ptr<NativeSymbolTable> symbols_ =
      std::make_shared<NativeSymbolTable>();
};

} // namespace

std::unique_ptr<VersionedTranslation>
MQT_QISKIT_VERSION_FACTORY() { // NOLINT(misc-use-internal-linkage): declared in
                               // the version registry.
  static const auto VERSION = []() {
    if (qk_import() < 0) {
      throwPythonError(
          "failed to initialize the Qiskit " MQT_QISKIT_VERSION_LABEL " C API");
    }
    return qk_api_version();
  }();
  const auto major = (VERSION >> 24U) & 0xffU;
  const auto minor = (VERSION >> 16U) & 0xffU;
  if (major != MQT_QISKIT_VERSION_EXPECTED_MAJOR ||
      minor != MQT_QISKIT_VERSION_EXPECTED_MINOR ||
      (MQT_QISKIT_VERSION_EXACT_API != 0 && VERSION != QISKIT_VERSION_HEX)) {
    throw std::runtime_error("Qiskit C API capsule version does not match the "
                             "selected " MQT_QISKIT_VERSION_LABEL
                             " translation");
  }
  return std::make_unique<NativeTranslation>();
}

} // namespace mqt::bindings::qiskit

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
// NOLINTEND(cppcoreguidelines-pro-type-cstyle-cast)
