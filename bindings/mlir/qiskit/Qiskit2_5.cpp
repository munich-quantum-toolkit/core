/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/Translation/StandardGate.h"

#include "QiskitTranslation.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/complex.h" // NOLINT(misc-include-cleaner): enables the std::complex caster.
#include "nanobind/stl/string.h" // NOLINT(misc-include-cleaner): enables the std::string caster.
#include "qiskit/complex.h"
#include "qiskit/version.h"
#include <qiskit.h> // Must precede the extension function table.
#include <qiskit/funcs_py.h>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

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

constexpr size_t MAX_EXPRESSION_DEPTH = 64U;
constexpr size_t MAX_EXPRESSION_NODES = 16384U;
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

[[nodiscard]] static std::string pythonHex(const nb::handle object,
                                           const std::string_view error) {
  try {
    return nb::cast<std::string>(
        nb::module_::import_("builtins").attr("hex")(object));
  } catch (const nb::python_error&) {
    throw std::runtime_error(std::string(error));
  }
}

[[nodiscard]] static std::string
pythonStringAttribute(const nb::handle object, const char* name,
                      const std::string_view error) {
  auto text = pythonText(pythonAttribute(object, name, error), error);
  if (text.find('\0') != std::string::npos) {
    throw std::runtime_error("Qiskit names cannot contain null characters");
  }
  return text;
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

[[nodiscard]] static llvm::APInt
pythonUnsignedValue(const nb::handle object, const uint32_t width,
                    const std::string_view error) {
  if (!nb::isinstance<nb::int_>(object)) {
    throw std::runtime_error(std::string(error));
  }
  const auto text = pythonHex(object, error);
  auto value = llvm::StringRef(text);
  llvm::APInt result;
  if (!value.consume_front("0x") || value.getAsInteger(16, result) ||
      result.getActiveBits() > width) {
    throw std::runtime_error(std::string(error));
  }
  return result;
}

[[nodiscard]] static nb::object pythonInteger(const llvm::APInt& value,
                                              const std::string_view error) {
  const auto text = llvm::toString(value, 16, false);
  try {
    return nb::module_::import_("builtins")
        .attr("int")(nb::str(text.c_str()), nb::int_(16));
  } catch (const nb::python_error&) {
    throw std::runtime_error(std::string(error));
  }
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
                           ParameterGroup{
                               .identity = std::move(groupIdentity),
                               .name = std::move(groupName),
                               .index = groupIndex,
                               .size = groupSize,
                           });
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
        operand.value = Parameter::unary(unaryParameterKind(opcode),
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
      stack.push_back({
          .value =
              Parameter::binary(binaryParameterKind(opcode),
                                std::move(left.value), std::move(right.value)),
          .depth = depth,
      });
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
  modifiers.push_back({
      .kind = GateModifierKind::Control,
      .numControls = static_cast<uint32_t>(controls),
      .exponent = {},
  });
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

[[nodiscard]] static bool isPythonStore(const nb::handle operation) {
  return nb::isinstance(operation,
                        nb::module_::import_("qiskit.circuit").attr("Store"));
}

[[nodiscard]] static bool isPythonGate(nb::handle operation) {
  const auto terminal = terminalPythonGate(operation);
  return nb::isinstance(terminal,
                        nb::module_::import_("qiskit.circuit").attr("Gate"));
}

[[nodiscard]] static bool isPythonStandardGate(nb::handle operation) {
  const auto terminal = terminalPythonGate(operation);
  const auto baseClass = pythonAttribute(
      terminal, "base_class", "Qiskit Gate does not expose its base class");
  return !pythonAttribute(baseClass, "_standard_gate",
                          "Qiskit Gate base class has no standard identity")
              .is_none();
}

static void normalizePythonModifier(const nb::handle modifier,
                                    std::vector<GateModifier>& modifiers) {
  const auto type = pythonAttribute(modifier, "__class__",
                                    "Qiskit modifier does not expose its type");
  const auto name = pythonStringAttribute(
      type, "__name__", "Qiskit modifier has an invalid type name");
  if (name == "InverseModifier") {
    modifiers.push_back({
        .kind = GateModifierKind::Inverse,
        .numControls = 0,
        .exponent = {},
    });
    return;
  }
  if (name == "ControlModifier") {
    appendControlModifier(modifier, modifiers);
    return;
  }
  if (name == "PowerModifier") {
    auto power = pythonAttribute(modifier, "power",
                                 "Qiskit power modifier has no exponent");
    modifiers.push_back({
        .kind = GateModifierKind::Power,
        .exponent = normalizePythonParameter(power),
    });
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

class DefinitionRegistry final {
public:
  [[nodiscard]] uintptr_t identify(nb::handle definition,
                                   const std::string_view name,
                                   nb::handle parameters) {
    const nb::tuple parameterTuple(parameters);
    const auto parameterHash = PyObject_Hash(parameterTuple.ptr());
    if (parameterHash == -1) {
      throwPythonError("Qiskit Gate parameters are not hashable");
    }
    // ponytail: use a structural circuit hash if many same-signature Gate
    // definitions become common.
    auto& bucket =
        definitions_[llvm::StringRef(name.data(), name.size())][parameterHash];
    for (const auto& entry : bucket) {
      if (entry.definition.is(definition) ||
          entry.definition.equal(definition)) {
        return entry.identity;
      }
    }
    const auto identity = nextIdentity_++;
    bucket.push_back({
        .definition = nb::borrow<nb::object>(definition),
        .identity = identity,
    });
    return identity;
  }

private:
  struct Definition {
    nb::object definition;
    uintptr_t identity;
  };

  using ParameterBuckets =
      std::unordered_map<Py_hash_t, std::vector<Definition>>;
  llvm::StringMap<ParameterBuckets> definitions_;
  uintptr_t nextIdentity_ = 1U;
};

class NativeCircuitReader final : public CircuitReader {
public:
  NativeCircuitReader(nb::handle circuit,
                      std::shared_ptr<DefinitionRegistry> definitions)
      : pythonCircuit_(nb::borrow<nb::object>(circuit)),
        data_(pythonAttribute(
            circuit, "_data",
            "expected a Qiskit QuantumCircuit with native CircuitData")),
        circuit_(qk_circuit_borrow_from_python(data_.ptr())),
        definitions_(std::move(definitions)) {
    if (circuit_ == nullptr) {
      throwPythonError("Qiskit rejected QuantumCircuit._data");
    }
  }

  NativeCircuitReader(nb::object pythonCircuit, const QkCircuit* circuit,
                      const QkControlFlowInstruction* parent,
                      std::shared_ptr<DefinitionRegistry> definitions)
      : pythonCircuit_(std::move(pythonCircuit)),
        data_(pythonAttribute(
            pythonCircuit_, "_data",
            "Qiskit control-flow block has no native CircuitData")),
        circuit_(circuit), parent_(parent),
        definitions_(std::move(definitions)) {}

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
  [[nodiscard]] std::vector<ClassicalVariable> variables() const override;

  [[nodiscard]] Register quantumRegister(const size_t index) const override {
    const auto* reg = qk_circuit_get_quantum_register(circuit_, index);
    Register result{
        .name = pythonStringAttribute(pythonCircuit_.attr("qregs")[index],
                                      "name", "Qiskit register has no name"),
        .bits = {},
    };
    result.bits.resize(qk_quantum_register_num_bits(reg));
    if (!result.bits.empty()) {
      qk_quantum_register_circuit_bits(reg, circuit_, result.bits.data());
    }
    return result;
  }

  [[nodiscard]] Register classicalRegister(const size_t index) const override {
    const auto* reg = qk_circuit_get_classical_register(circuit_, index);
    Register result{
        .name = pythonStringAttribute(pythonCircuit_.attr("cregs")[index],
                                      "name", "Qiskit register has no name"),
        .bits = {},
    };
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

  [[nodiscard]] OperationKind instructionKind(size_t index) const override {
    return normalizeKind(qk_circuit_instruction_kind(circuit_, index));
  }

  [[nodiscard]] Instruction instruction(const size_t index) const override {
    const auto kind = instructionKind(index);
    if (kind == OperationKind::Delay) {
      return {
          .kind = kind,
          .name = "delay",
          .qubits = {},
          .clbits = {},
          .parameters = {},
          .modifiers = {},
          .standardGate = {},
      };
    }
    if (kind == OperationKind::ControlFlow) {
      return {
          .kind = kind,
          .name = "control_flow",
          .qubits = {},
          .clbits = {},
          .parameters = {},
          .modifiers = {},
          .standardGate = {},
      };
    }
    const auto operation = pythonOperation(index);
    if (pythonStringAttribute(operation, "name",
                              "Qiskit operation has an invalid name") ==
            "store" &&
        isPythonStore(operation)) {
      return {
          .kind = OperationKind::Store,
          .name = "store",
          .qubits = {},
          .clbits = {},
          .parameters = {},
          .modifiers = {},
          .standardGate = {},
      };
    }
    std::optional<Instruction> normalizedUnknown;
    if (kind == OperationKind::Unknown) {
      if (isPythonUnitaryGate(operation)) {
        Instruction result{
            .kind = OperationKind::Unitary,
            .name = "unitary",
            .qubits = {},
            .clbits = {},
            .parameters = {},
            .modifiers = {},
            .standardGate = {},
        };
        normalizePythonGate(operation, result);
        result.name = "unitary";
        result.qubits = pythonInstructionQubits(index);
        return result;
      }
      normalizedUnknown.emplace();
      normalizePythonGate(operation, *normalizedUnknown);
      if (isPythonGate(operation)) {
        normalizedUnknown->kind = OperationKind::Gate;
      }
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
          pythonAttribute(operation, "params",
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
    } else if (native.num_params != 0U) {
      throw std::runtime_error(
          "Qiskit non-gate instruction has unexpected scalar parameters");
    }
    if (kind == OperationKind::Unknown) {
      result.name = std::move(normalizedUnknown->name);
      result.modifiers = std::move(normalizedUnknown->modifiers);
      result.kind = normalizedUnknown->kind;
    }
    if (kind != OperationKind::Unknown || isPythonStandardGate(operation)) {
      result.standardGate = standardGateMapping(result.name);
    }
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

  [[nodiscard]] ClassicalAssignment store(size_t index) const override;

  [[nodiscard]] std::unique_ptr<CircuitReader>
  definition(const size_t index) const override {
    const auto operation = terminalPythonGate(pythonOperation(index));
    const auto definition = pythonAttribute(
        operation, "definition",
        "Qiskit instruction does not expose a circuit definition");
    if (definition.is_none()) {
      throw std::runtime_error("Qiskit instruction '" +
                               instruction(index).name +
                               "' has no circuit definition");
    }
    return std::make_unique<NativeCircuitReader>(definition, definitions_);
  }

  [[nodiscard]] uintptr_t
  definitionIdentity(const size_t index) const override {
    const auto operation = terminalPythonGate(pythonOperation(index));
    const auto definition = pythonAttribute(
        operation, "definition",
        "Qiskit instruction does not expose a circuit definition");
    if (definition.is_none()) {
      return 0U;
    }
    const auto name = pythonStringAttribute(
        operation, "name", "Qiskit instruction has no valid name");
    const auto parameters = pythonAttribute(
        operation, "params", "Qiskit instruction has no parameter list");
    return definitions_->identify(definition, name, parameters);
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
  const QkControlFlowInstruction* parent_ = nullptr;
  std::shared_ptr<DefinitionRegistry> definitions_;
};
} // namespace

using ClassicalBitResolver = llvm::function_ref<uint32_t(nb::handle)>;

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
    if (width == 0U || width > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(
          "Qiskit unsigned classical value width is out of range");
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
  const auto operation = llvm::StringSwitch<std::optional<UnaryOperation>>(name)
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

static void normalizePythonVariable(Expression& result,
                                    const nb::handle pythonExpression,
                                    const ClassicalBitResolver& resolveBit) {
  const auto variable = pythonAttribute(
      pythonExpression, "var", "Qiskit variable expression has no value");
  const auto circuitModule = nb::module_::import_("qiskit.circuit");
  if (nb::isinstance(variable, circuitModule.attr("Clbit"))) {
    if (result.type != ClassicalType::Bool || result.width != 1U) {
      throw std::runtime_error(
          "Qiskit classical-bit variable must have Boolean type");
    }
    result.kind = ExpressionKind::ClassicalBit;
    result.bit = resolveBit(variable);
    return;
  }
  if (nb::isinstance(variable, circuitModule.attr("ClassicalRegister"))) {
    if (result.type != ClassicalType::Uint || nb::len(variable) == 0U ||
        result.width < nb::len(variable)) {
      throw std::runtime_error(
          "Qiskit classical-register variable has an invalid type");
    }
    result.kind = ExpressionKind::ClassicalRegister;
    result.reg.name = pythonStringAttribute(
        variable, "name", "Qiskit classical register has no name");
    result.reg.bits.reserve(nb::len(variable));
    for (const nb::handle bit : nb::iter(variable)) {
      result.reg.bits.push_back(resolveBit(bit));
    }
    return;
  }
  if (!nb::isinstance(variable, nb::module_::import_("uuid").attr("UUID"))) {
    throw std::runtime_error(
        "Qiskit classical variable has an invalid identity");
  }
  result.kind = ExpressionKind::Variable;
  result.variable = nb::cast<std::string>(nb::str(variable));
}

[[nodiscard]] static std::unique_ptr<Expression> normalizePythonExpressionOnly(
    const nb::handle pythonExpression, size_t& nodeCount,
    const ClassicalBitResolver& resolveBit, const size_t depth = 0U) {
  if (depth >= MAX_EXPRESSION_DEPTH) {
    throw std::runtime_error(
        "Qiskit classical expressions exceed the nesting limit of 64");
  }
  if (nodeCount >= MAX_EXPRESSION_NODES) {
    throw std::runtime_error(
        "Qiskit classical expressions exceed the node limit of 16384");
  }
  ++nodeCount;
  auto result = std::make_unique<Expression>();
  setPythonExpressionType(*result, pythonExpression);
  const auto className = pythonStringAttribute(
      pythonAttribute(pythonExpression, "__class__",
                      "Qiskit expression has no Python class"),
      "__name__", "Qiskit expression has no class name");
  if (className == "Var") {
    normalizePythonVariable(*result, pythonExpression, resolveBit);
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
      result->uintValue = pythonUnsignedValue(
          value, result->width,
          "Qiskit Uint literal does not fit its declared width");
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
        nodeCount, resolveBit, depth + 1U);
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
        nodeCount, resolveBit, depth + 1U);
    result->right = normalizePythonExpressionOnly(
        pythonAttribute(pythonExpression, "right",
                        "Qiskit binary expression has no right operand"),
        nodeCount, resolveBit, depth + 1U);
    return result;
  }
  if (className == "Cast") {
    result->kind = ExpressionKind::Cast;
    result->left = normalizePythonExpressionOnly(
        pythonAttribute(pythonExpression, "operand",
                        "Qiskit cast expression has no operand"),
        nodeCount, resolveBit, depth + 1U);
    return result;
  }
  if (className == "Index") {
    result->kind = ExpressionKind::Index;
    result->left = normalizePythonExpressionOnly(
        pythonAttribute(pythonExpression, "target",
                        "Qiskit index expression has no target"),
        nodeCount, resolveBit, depth + 1U);
    result->right = normalizePythonExpressionOnly(
        pythonAttribute(pythonExpression, "index",
                        "Qiskit index expression has no index"),
        nodeCount, resolveBit, depth + 1U);
    return result;
  }
  if (className == "Stretch") {
    throw std::runtime_error(
        "Qiskit circuit import does not support stretch expressions");
  }
  throw std::runtime_error("Qiskit expression has an unknown Python node");
}

[[nodiscard]] static ClassicalTarget
normalizePythonTarget(const nb::handle target,
                      const ClassicalBitResolver& resolveBit) {
  ClassicalTarget result;
  const auto circuitModule = nb::module_::import_("qiskit.circuit");
  if (nb::isinstance(target, circuitModule.attr("Clbit"))) {
    result.kind = ClassicalTargetKind::ClassicalBit;
    result.bit = resolveBit(target);
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
      result.reg.bits.push_back(resolveBit(bit));
    }
    result.width = static_cast<uint32_t>(size);
    return result;
  }
  const auto expressionModule =
      nb::module_::import_("qiskit.circuit.classical.expr");
  if (nb::isinstance(target, expressionModule.attr("Expr"))) {
    result.kind = ClassicalTargetKind::Expression;
    size_t nodeCount = 0U;
    result.expression =
        normalizePythonExpressionOnly(target, nodeCount, resolveBit);
    return result;
  }
  throw std::runtime_error("Qiskit classical target has an unknown type");
}

namespace {
class NativeControlFlowReader final : public ControlFlowReader {
public:
  NativeControlFlowReader(const QkCircuit* circuit, const size_t index,
                          const QkControlFlowInstruction* parent,
                          nb::object instruction,
                          nb::object containingPythonCircuit,
                          std::shared_ptr<DefinitionRegistry> definitions)
      : circuit_(circuit), parent_(parent),
        instruction_(std::move(instruction)),
        operation_(pythonAttribute(
            instruction_, "operation",
            "Qiskit circuit instruction has no control-flow operation")),
        containingPythonCircuit_(std::move(containingPythonCircuit)),
        definitions_(std::move(definitions)),
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
        block, qk_control_flow_block_circuit(controlFlow_, index), controlFlow_,
        definitions_);
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
    const auto expected = pythonUnsignedValue(
        condition[1], std::numeric_limits<uint32_t>::max(),
        "Qiskit control-flow condition has an invalid value");
    auto result =
        normalizePythonTarget(expressionModule.attr("lift")(condition[0]));
    const auto& target = *result.expression;
    if (target.kind == ExpressionKind::ClassicalBit) {
      if (expected.getActiveBits() > 1U) {
        throw std::runtime_error(
            "Qiskit classical-bit condition must compare against zero or one");
      }
      return normalizePythonTarget(expressionModule.attr("equal")(
          condition[0], nb::bool_(!expected.isZero())));
    }
    if (target.kind == ExpressionKind::ClassicalRegister) {
      if (expected.getActiveBits() > target.reg.bits.size()) {
        return normalizePythonTarget(
            expressionModule.attr("lift")(nb::bool_(false)));
      }
      return normalizePythonTarget(expressionModule.attr("equal")(
          condition[0],
          pythonInteger(expected,
                        "Qiskit control-flow condition has an invalid value")));
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
      const auto parameters = pythonAttribute(
          operation_, "params", "Qiskit for-loop operation has no parameters");
      if (nb::len(parameters) < 2U) {
        throw std::runtime_error(
            "Qiskit for-loop operation has no loop parameter");
      }
      result.parameter = normalizePythonParameter(parameters[1]);
      if (result.parameter->getSymbol() == nullptr) {
        throw std::runtime_error("Qiskit for-loop parameter is not a symbol");
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
      SwitchCase entry{
          .isDefault =
              qk_control_flow_switch_is_case_default(controlFlow_, index),
      };
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
    return mqt::bindings::qiskit::normalizePythonTarget(
        target, [&](const nb::handle bit) { return rootClbitIndex(bit); });
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

  const QkCircuit* circuit_ = nullptr;
  const QkControlFlowInstruction* parent_ = nullptr;
  nb::object instruction_;
  nb::object operation_;
  nb::object containingPythonCircuit_;
  std::shared_ptr<DefinitionRegistry> definitions_;
  QkControlFlowInstruction* controlFlow_ = nullptr;
};
} // namespace

std::vector<ClassicalVariable> NativeCircuitReader::variables() const {
  std::vector<ClassicalVariable> result;
  const auto append = [&](const char* method, bool captured, bool input) {
    for (auto variable : nb::iter(pythonCircuit_.attr(method)())) {
      size_t nodes = 0;
      auto normalized = normalizePythonExpressionOnly(
          variable, nodes, [](nb::handle) -> uint32_t {
            throw std::runtime_error("expected a standalone Qiskit variable");
          });
      if (normalized->kind != ExpressionKind::Variable) {
        throw std::runtime_error(
            "Qiskit local variable has no stable identity");
      }
      result.push_back({
          .identity = normalized->variable,
          .name = pythonStringAttribute(variable, "name",
                                        "Qiskit variable has no name"),
          .type = normalized->type,
          .width = normalized->width,
          .captured = captured,
          .input = input,
      });
    }
  };
  append("iter_declared_vars", false, false);
  append("iter_captured_vars", true, false);
  append("iter_input_vars", false, true);
  return result;
}

ClassicalAssignment NativeCircuitReader::store(const size_t index) const {
  const auto operation = pythonOperation(index);
  if (!isPythonStore(operation)) {
    throw std::runtime_error(
        "requested classical assignment for a non-Store instruction");
  }
  const auto resolveBit = [&](const nb::handle bit) -> uint32_t {
    try {
      const auto location =
          pythonAttribute(pythonCircuit_, "find_bit",
                          "Qiskit circuit cannot resolve Store variables")(bit);
      const auto position = pythonUnsignedAttribute(
          location, "index", "Qiskit Store variable has an invalid index");
      if (position >= numClbits()) {
        throw std::runtime_error(
            "Qiskit Store variable has an invalid classical-bit index");
      }
      return static_cast<uint32_t>(position);
    } catch (const nb::python_error& error) {
      throwPythonError("Qiskit Store variable is absent from its circuit",
                       error);
    }
  };
  auto target = normalizePythonTarget(
      pythonAttribute(operation, "lvalue", "Qiskit Store has no lvalue"),
      resolveBit);
  if (target.kind == ClassicalTargetKind::Expression && target.expression) {
    if (target.expression->kind == ExpressionKind::ClassicalBit) {
      target.kind = ClassicalTargetKind::ClassicalBit;
      target.bit = target.expression->bit;
      target.expression.reset();
    } else if (target.expression->kind == ExpressionKind::ClassicalRegister) {
      target.kind = ClassicalTargetKind::ClassicalRegister;
      target.width = target.expression->width;
      target.reg = std::move(target.expression->reg);
      target.expression.reset();
    }
  }
  size_t nodeCount = 0U;
  return {
      .target = std::move(target),
      .value = normalizePythonExpressionOnly(
          pythonAttribute(operation, "rvalue", "Qiskit Store has no rvalue"),
          nodeCount, resolveBit),
  };
}

std::unique_ptr<ControlFlowReader>
NativeCircuitReader::controlFlow(const size_t index) const {
  return std::make_unique<NativeControlFlowReader>(
      circuit_, index, parent_, nb::borrow<nb::object>(data_[index]),
      pythonCircuit_, definitions_);
}

namespace {
using PythonVariables = llvm::StringMap<nb::object>;

class PythonClassicalBuilder final {
public:
  explicit PythonClassicalBuilder(const nb::handle circuit,
                                  const PythonVariables& variables)
      : circuit_(nb::borrow<nb::object>(circuit)), variables_(variables),
        clbits_(pythonAttribute(circuit, "clbits",
                                "Qiskit circuit has no classical bits")),
        cregs_(pythonAttribute(circuit, "cregs",
                               "Qiskit circuit has no classical registers")),
        expressionModule_(
            nb::module_::import_("qiskit.circuit.classical.expr")),
        typesModule_(nb::module_::import_("qiskit.circuit.classical.types")) {}

  [[nodiscard]] nb::object expression(const Expression& value) const {
    return expression(value, 0U);
  }

  [[nodiscard]] nb::object lvalue(const ClassicalTarget& target) const {
    switch (target.kind) {
    case ClassicalTargetKind::ClassicalBit:
      return expressionModule_.attr("lift")(classicalBit(target.bit));
    case ClassicalTargetKind::ClassicalRegister:
      return expressionModule_.attr("lift")(
          registeredClassicalRegister(target.reg),
          classicalType(ClassicalType::Uint,
                        static_cast<uint32_t>(target.reg.bits.size())));
    case ClassicalTargetKind::Expression:
      if (!target.expression) {
        throw std::runtime_error("Qiskit Store has no lvalue expression");
      }
      return expression(*target.expression);
    }
    throw std::runtime_error("Qiskit Store has an unknown lvalue");
  }

  [[nodiscard]] nb::object condition(const ClassicalTarget& target) const {
    if (target.kind != ClassicalTargetKind::Expression || !target.expression) {
      throw std::runtime_error(
          "Qiskit control-flow condition has no expression");
    }
    if (target.expression->type != ClassicalType::Bool) {
      throw std::runtime_error(
          "Qiskit control-flow condition expression must be Boolean");
    }
    return expression(*target.expression);
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
      return registeredClassicalRegister(target.reg);
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

  [[nodiscard]] nb::object classicalType(const ClassicalType type,
                                         const uint32_t width) const {
    switch (type) {
    case ClassicalType::Bool:
      if (width != 1U) {
        throw std::runtime_error("Qiskit Boolean expressions require width 1");
      }
      return typesModule_.attr("Bool")();
    case ClassicalType::Uint:
      if (width == 0U) {
        throw std::runtime_error("Qiskit unsigned expressions require a width");
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

private:
  [[nodiscard]] nb::object classicalBit(const uint32_t bit) const {
    if (bit >= nb::len(clbits_)) {
      throw std::runtime_error(
          "Qiskit classical expression references an invalid bit");
    }
    return nb::borrow<nb::object>(clbits_[bit]);
  }

  [[nodiscard]] nb::object
  registeredClassicalRegister(const Register& reg) const {
    for (const nb::handle candidateHandle : nb::iter(cregs_)) {
      auto candidate = nb::borrow<nb::object>(candidateHandle);
      if (pythonStringAttribute(candidate, "name",
                                "Qiskit classical register has no name") ==
          reg.name) {
        return candidate;
      }
    }
    throw std::runtime_error(
        "Qiskit classical expression references a missing register");
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
    case ExpressionKind::Variable: {
      const auto found = variables_.find(value.variable);
      if (found == variables_.end()) {
        throw std::runtime_error("Qiskit expression refers to an unavailable "
                                 "local variable capture");
      }
      if (!nb::cast<bool>(circuit_.attr("has_var")(found->second))) {
        circuit_.attr("add_capture")(found->second);
      }
      return found->second;
    }
    case ExpressionKind::Value: {
      const auto type = classicalType(value.type, value.width);
      switch (value.type) {
      case ClassicalType::Bool:
        return expressionModule_.attr("lift")(nb::bool_(value.boolValue), type);
      case ClassicalType::Uint: {
        if (value.uintValue.getActiveBits() > value.width) {
          throw std::runtime_error(
              "Qiskit unsigned expression value exceeds its width");
        }
        return expressionModule_.attr("lift")(
            pythonInteger(value.uintValue,
                          "Qiskit failed to convert a Uint literal"),
            type);
      }
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
          value.width < value.reg.bits.size()) {
        throw std::runtime_error(
            "Qiskit classical-register expression has an invalid type");
      }
      return expressionModule_.attr("lift")(
          registeredClassicalRegister(value.reg),
          classicalType(ClassicalType::Uint, value.width));
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
  const PythonVariables& variables_;
  nb::object clbits_;
  nb::object cregs_;
  nb::object expressionModule_;
  nb::object typesModule_;
};

using PythonParameterGroups = llvm::StringMap<nb::object>;
using PythonSymbols = llvm::StringMap<nb::object>;
using NativeGateRegistry = llvm::StringMap<nb::object>;

struct OutputParameters {
  PythonSymbols symbols;
  PythonParameterGroups groups;
};

class NativeCircuitWriter final : public CircuitWriter {
public:
  NativeCircuitWriter(uint32_t looseQubits, uint32_t looseClbits,
                      std::shared_ptr<OutputParameters> parameters,
                      std::shared_ptr<NativeGateRegistry> gates)
      : parameters_(std::move(parameters)), gates_(std::move(gates)) {
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    pythonCircuit_ = circuitModule.attr("QuantumCircuit")();
    nb::list bits;
    for (uint32_t i = 0; i < looseQubits; ++i) {
      bits.append(circuitModule.attr("Qubit")());
    }
    for (uint32_t i = 0; i < looseClbits; ++i) {
      bits.append(circuitModule.attr("Clbit")());
    }
    pythonCircuit_.attr("add_bits")(bits);
  }

  NativeCircuitWriter(nb::object circuit,
                      std::shared_ptr<OutputParameters> parameters,
                      std::shared_ptr<NativeGateRegistry> gates,
                      PythonVariables variables)
      : pythonCircuit_(std::move(circuit)), variables_(std::move(variables)),
        parameters_(std::move(parameters)), gates_(std::move(gates)) {}

  [[nodiscard]] std::unique_ptr<CircuitWriter> createBlock() const override {
    auto block = nb::module_::import_("qiskit.circuit")
                     .attr("QuantumCircuit")(pythonCircuit_.attr("qubits"),
                                             pythonCircuit_.attr("clbits"));
    for (auto reg : nb::iter(pythonCircuit_.attr("qregs"))) {
      block.attr("add_register")(reg);
    }
    for (auto reg : nb::iter(pythonCircuit_.attr("cregs"))) {
      block.attr("add_register")(reg);
    }
    return std::make_unique<NativeCircuitWriter>(std::move(block), parameters_,
                                                 gates_, variables_);
  }

  void addQuantumRegister(std::string_view name, uint32_t size) override {
    pythonCircuit_.attr("add_register")(
        nb::module_::import_("qiskit.circuit")
            .attr("QuantumRegister")(size, nb::str(name.data(), name.size())));
  }

  void addClassicalRegister(std::string_view name, uint32_t size) override {
    pythonCircuit_.attr("add_register")(
        nb::module_::import_("qiskit.circuit")
            .attr("ClassicalRegister")(size,
                                       nb::str(name.data(), name.size())));
  }

  void setGlobalPhase(const Parameter& phase) override {
    pythonCircuit_.attr("global_phase") = pythonParameter(phase);
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
    if (std::ranges::all_of(parameters, [](const Parameter& parameter) {
          return parameter.getNumber() != nullptr;
        })) {
      std::vector<double> numbers;
      numbers.reserve(parameters.size());
      for (const auto& parameter : parameters) {
        const auto value = parameter.getNumber()->value;
        if (!std::isfinite(value)) {
          throw std::runtime_error(
              "cannot construct a non-finite Qiskit parameter");
        }
        numbers.push_back(value);
      }
      checkExitCode(qk_circuit_gate(nativeCircuit(), gate->native,
                                    qubits.data(), numbers.data()),
                    "adding gate");
      return;
    }
    nb::list values;
    for (const auto& parameter : parameters) {
      values.append(pythonParameter(parameter));
    }
    if (!standardGates_.is_valid()) {
      standardGates_ = nb::module_::import_("qiskit.circuit.library")
                           .attr("get_standard_gate_name_mapping")();
      standardInstruction_ = nb::module_::import_("qiskit.circuit")
                                 .attr("CircuitInstruction")
                                 .attr("from_standard");
    }
    const nb::object standard =
        standardGates_[nb::str(gate->name.data(), gate->name.size())];
    auto instruction = standardInstruction_(standard.attr("_standard_gate"),
                                            pythonQubits(qubits), values);
    /// As with numeric appends, this private circuit has no builder scope or
    /// cached duration.
    pythonCircuit_.attr("_data").attr("append")(instruction);
  }

  void addCustomGate(std::string_view name, const std::vector<uint32_t>& qubits,
                     const std::vector<Parameter>& parameters,
                     const std::vector<GateModifier>& gateModifiers) override {
    const auto circuitModule = nb::module_::import_("qiskit.circuit");
    const auto gate = gates_->find(std::string(name));
    if (gate == gates_->end()) {
      throw std::runtime_error("Qiskit custom Gate '" + std::string(name) +
                               "' has no registered definition");
    }
    const nb::object formalParameters = gate->second.attr("params");
    if (nb::len(formalParameters) != parameters.size()) {
      throw std::runtime_error("Qiskit custom Gate '" + std::string(name) +
                               "' has incompatible parameters");
    }
    nb::dict parameterMap;
    for (size_t index = 0U; index < parameters.size(); ++index) {
      parameterMap[formalParameters[index]] =
          pythonParameter(parameters[index]);
    }
    nb::object operation = gate->second;
    if (!parameters.empty()) {
      operation =
          pythonAttribute(operation.attr("definition"), "to_gate",
                          "Qiskit custom definition cannot become a Gate")(
              nb::arg("parameter_map") = parameterMap);
    }
    if (!gateModifiers.empty()) {
      nb::list modifiers;
      for (const auto& modifier : gateModifiers) {
        switch (modifier.kind) {
        case GateModifierKind::Inverse:
          modifiers.append(circuitModule.attr("InverseModifier")());
          break;
        case GateModifierKind::Control:
          modifiers.append(
              circuitModule.attr("ControlModifier")(modifier.numControls));
          break;
        case GateModifierKind::Power: {
          const auto* exponent = modifier.exponent.getNumber();
          if (exponent == nullptr) {
            throw std::runtime_error(
                "Qiskit custom Gate power must be numeric");
          }
          modifiers.append(
              circuitModule.attr("PowerModifier")(exponent->value));
          break;
        }
        }
      }
      operation =
          circuitModule.attr("AnnotatedOperation")(operation, modifiers);
    }
    pythonCircuit_.attr("append")(operation, pythonQubits(qubits));
  }

  void addMeasure(const uint32_t qubit, const uint32_t clbit) override {
    checkExitCode(qk_circuit_measure(nativeCircuit(), qubit, clbit),
                  "adding measurement");
  }

  void addReset(const uint32_t qubit) override {
    checkExitCode(qk_circuit_reset(nativeCircuit(), qubit), "adding reset");
  }

  void addBarrier(const std::vector<uint32_t>& qubits) override {
    checkExitCode(qk_circuit_barrier(nativeCircuit(), qubits.data(),
                                     static_cast<uint32_t>(qubits.size())),
                  "adding barrier");
  }

  void addStore(ClassicalTarget target,
                std::unique_ptr<Expression> value) override {
    if (!value) {
      throw std::runtime_error("Qiskit Store has no rvalue");
    }
    const PythonClassicalBuilder classical(pythonCircuit_, variables_);
    pythonCircuit_.attr("store")(classical.lvalue(target),
                                 classical.expression(*value));
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
    checkExitCode(
        qk_circuit_unitary(nativeCircuit(), native.data(), targets.data(),
                           static_cast<uint32_t>(targets.size()), true),
        "adding unitary");
    if (numControls != 0U) {
      nb::object data = pythonCircuit_.attr("data");
      auto instruction = data[nb::len(data) - 1U];
      auto controlled =
          instruction.attr("operation")
              .attr("control")(numControls, nb::arg("annotated") = true);
      data[nb::len(data) - 1U] =
          instruction.attr("replace")(nb::arg("operation") = controlled,
                                      nb::arg("qubits") = pythonQubits(qubits));
    }
  }

  void declareVariable(ClassicalVariable variable) override {
    const PythonClassicalBuilder classical(pythonCircuit_, variables_);
    auto local =
        nb::module_::import_("qiskit.circuit.classical.expr")
            .attr("Var")
            .attr("new")(variable.name, classical.classicalType(
                                            variable.type, variable.width));
    pythonCircuit_.attr("add_uninitialized_var")(local);
    if (!variables_.try_emplace(variable.identity, local).second) {
      throw std::runtime_error(
          "Qiskit export contains a duplicate local variable identity");
    }
  }

  void
  addControlFlow(const ControlFlowKind kind, ClassicalTarget target, Loop loop,
                 std::vector<SwitchCase> switchCases,
                 std::vector<std::unique_ptr<CircuitWriter>> blocks) override {
    const bool validBlockCount = [&] {
      switch (kind) {
      case ControlFlowKind::IfElse:
        return blocks.size() == 1U || blocks.size() == 2U;
      case ControlFlowKind::While:
      case ControlFlowKind::For:
        return blocks.size() == 1U;
      case ControlFlowKind::Switch:
        return !blocks.empty() && blocks.size() == switchCases.size();
      case ControlFlowKind::Break:
      case ControlFlowKind::Continue:
        return blocks.empty();
      case ControlFlowKind::Box:
        return false;
      }
      return false;
    }();
    if (!validBlockCount) {
      throw std::runtime_error(
          "Qiskit control flow has an unexpected number of blocks");
    }
    const auto numQubits = qk_circuit_num_qubits(nativeCircuit());
    const auto numClbits = qk_circuit_num_clbits(nativeCircuit());
    std::vector<nb::object> pythonBlocks;
    for (auto& block : blocks) {
      auto body = block->finish();
      if (nb::cast<uint32_t>(body.attr("num_qubits")) != numQubits ||
          nb::cast<uint32_t>(body.attr("num_clbits")) != numClbits) {
        throw std::runtime_error(
            "Qiskit control-flow block has incompatible bit counts");
      }
      for (auto captured : nb::iter(body.attr("iter_captured_vars")())) {
        if (!nb::cast<bool>(pythonCircuit_.attr("has_var")(captured))) {
          pythonCircuit_.attr("add_capture")(captured);
        }
      }
      pythonBlocks.push_back(std::move(body));
    }
    const PythonClassicalBuilder classical(pythonCircuit_, variables_);
    auto operation = constructControlFlowOperation(
        kind, target, loop, switchCases, pythonBlocks, classical,
        nb::module_::import_("qiskit.circuit"), numQubits, numClbits);
    pythonCircuit_.attr("append")(operation, pythonCircuit_.attr("qubits"),
                                  pythonCircuit_.attr("clbits"));
  }

  [[nodiscard]] nb::object finish() override {
    return std::move(pythonCircuit_);
  }

private:
  [[nodiscard]] QkCircuit* nativeCircuit() const {
    /// Reborrow after Python calls; no native pointer outlives its data owner.
    auto data = pythonCircuit_.attr("_data");
    auto* circuit = qk_circuit_borrow_from_python(data.ptr());
    if (circuit == nullptr) {
      throwPythonError("Qiskit rejected output CircuitData");
    }
    return circuit;
  }

  [[nodiscard]] nb::list
  pythonQubits(const std::vector<uint32_t>& qubits) const {
    nb::list result;
    const auto bits = pythonCircuit_.attr("qubits");
    for (auto qubit : qubits) {
      result.append(bits[qubit]);
    }
    return result;
  }

  [[nodiscard]] nb::object pythonParameter(const Parameter& parameter,
                                           PythonSymbols& symbols,
                                           size_t& nodeCount, size_t depth) {
    countParameterExpressionNode(nodeCount);
    if (depth > MAX_PARAMETER_EXPRESSION_DEPTH) {
      throwParameterExpressionDepthError();
    }
    if (const auto* number = parameter.getNumber()) {
      if (!std::isfinite(number->value)) {
        throw std::runtime_error(
            "cannot construct a non-finite Qiskit parameter");
      }
      return nb::float_(number->value);
    }
    if (const auto* symbol = parameter.getSymbol()) {
      auto pythonSymbol = symbols.find(symbol->name);
      if (pythonSymbol == symbols.end()) {
        nb::object value;
        if (symbol->group) {
          const auto& metadata = *symbol->group;
          const auto [group, inserted] =
              parameters_->groups.try_emplace(metadata.identity);
          if (inserted) {
            group->second =
                nb::module_::import_("qiskit.circuit")
                    .attr("ParameterVector")(metadata.name, metadata.size);
          }
          value = nb::module_::import_("qiskit.circuit")
                      .attr("ParameterVectorElement")(group->second,
                                                      metadata.index);
        } else {
          value = nb::module_::import_("qiskit.circuit")
                      .attr("Parameter")(symbol->name);
        }
        pythonSymbol =
            symbols.try_emplace(symbol->name, std::move(value)).first;
      }
      return nb::borrow<nb::object>(pythonSymbol->second);
    }
    if (const auto* unary = parameter.getUnary()) {
      auto operand =
          pythonParameter(*unary->operand, symbols, nodeCount, depth + 1U);
      if (nb::isinstance<nb::float_>(operand)) {
        auto numeric = nb::cast<double>(operand);
        switch (unary->operation) {
        case UnaryParameterKind::Negate:
          numeric = -numeric;
          break;
        case UnaryParameterKind::Sin:
          numeric = std::sin(numeric);
          break;
        case UnaryParameterKind::Cos:
          numeric = std::cos(numeric);
          break;
        case UnaryParameterKind::Tan:
          numeric = std::tan(numeric);
          break;
        case UnaryParameterKind::ArcSin:
          numeric = std::asin(numeric);
          break;
        case UnaryParameterKind::ArcCos:
          numeric = std::acos(numeric);
          break;
        case UnaryParameterKind::ArcTan:
          numeric = std::atan(numeric);
          break;
        case UnaryParameterKind::Exp:
          numeric = std::exp(numeric);
          break;
        case UnaryParameterKind::Log:
          numeric = std::log(numeric);
          break;
        case UnaryParameterKind::Abs:
          numeric = std::abs(numeric);
          break;
        case UnaryParameterKind::Conjugate:
          break;
        }
        if (!std::isfinite(numeric)) {
          throw std::runtime_error(
              "cannot construct a non-finite Qiskit parameter");
        }
        return nb::float_(numeric);
      }
      switch (unary->operation) {
      case UnaryParameterKind::Negate:
        return -operand;
      case UnaryParameterKind::Sin:
        return operand.attr("sin")();
      case UnaryParameterKind::Cos:
        return operand.attr("cos")();
      case UnaryParameterKind::Tan:
        return operand.attr("tan")();
      case UnaryParameterKind::ArcSin:
        return operand.attr("arcsin")();
      case UnaryParameterKind::ArcCos:
        return operand.attr("arccos")();
      case UnaryParameterKind::ArcTan:
        return operand.attr("arctan")();
      case UnaryParameterKind::Exp:
        return operand.attr("exp")();
      case UnaryParameterKind::Log:
        return operand.attr("log")();
      case UnaryParameterKind::Abs:
        return operand.attr("abs")();
      case UnaryParameterKind::Conjugate:
        return operand.attr("conjugate")();
      }
    }
    if (const auto* binary = parameter.getBinary()) {
      auto left =
          pythonParameter(*binary->left, symbols, nodeCount, depth + 1U);
      auto right =
          pythonParameter(*binary->right, symbols, nodeCount, depth + 1U);
      switch (binary->operation) {
      case BinaryParameterKind::Add:
        return left + right;
      case BinaryParameterKind::Subtract:
        return left - right;
      case BinaryParameterKind::Multiply:
        return left * right;
      case BinaryParameterKind::Divide:
        return left / right;
      case BinaryParameterKind::Power:
        return nb::module_::import_("builtins").attr("pow")(left, right);
      }
    }
    throw std::runtime_error("unknown normalized parameter expression");
  }

  [[nodiscard]] nb::object pythonParameter(const Parameter& parameter) {
    size_t nodeCount = 0U;
    return pythonParameter(parameter, parameters_->symbols, nodeCount, 1U);
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
      ControlFlowKind kind, const ClassicalTarget& target, const Loop& loop,
      const std::vector<SwitchCase>& switchCases,
      const std::vector<nb::object>& blocks,
      const PythonClassicalBuilder& classical, const nb::handle circuitModule,
      uint32_t numQubits, uint32_t numClbits) {
    switch (kind) {
    case ControlFlowKind::IfElse:
      return circuitModule.attr("IfElseOp")(
          classical.condition(target), blocks.front(),
          blocks.size() == 2U ? blocks[1] : nb::borrow<nb::object>(nb::none()));
    case ControlFlowKind::While:
      return circuitModule.attr("WhileLoopOp")(classical.condition(target),
                                               blocks.front());
    case ControlFlowKind::For:
      return circuitModule.attr("ForLoopOp")(
          loopIndexSet(loop), loopParameter(loop, blocks.front()),
          blocks.front());
    case ControlFlowKind::Switch: {
      nb::list cases;
      for (size_t index = 0U; index < switchCases.size(); ++index) {
        const auto& switchCase = switchCases[index];
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
      return circuitModule.attr("SwitchCaseOp")(classical.switchTarget(target),
                                                cases);
    }
    case ControlFlowKind::Break:
      return circuitModule.attr("BreakLoopOp")(numQubits, numClbits);
    case ControlFlowKind::Continue:
      return circuitModule.attr("ContinueLoopOp")(numQubits, numClbits);
    case ControlFlowKind::Box:
      break;
    }
    throw std::runtime_error(
        "Qiskit circuit export encountered an unsupported control-flow kind");
  }

  nb::object pythonCircuit_;
  nb::object standardGates_;
  nb::object standardInstruction_;
  PythonVariables variables_;
  std::shared_ptr<OutputParameters> parameters_;
  std::shared_ptr<NativeGateRegistry> gates_;
};

class NativeTranslation final : public VersionedTranslation {
public:
  [[nodiscard]] std::unique_ptr<CircuitReader>
  openCircuit(const nb::handle circuit) const override {
    return std::make_unique<NativeCircuitReader>(
        circuit, std::make_shared<DefinitionRegistry>());
  }
  [[nodiscard]] bool
  supportsGate(const StandardGateMapping gate) const override {
    return versionGate(gate) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<CircuitWriter>
  createCircuit(const uint32_t looseQubits,
                const uint32_t looseClbits) const override {
    return std::make_unique<NativeCircuitWriter>(looseQubits, looseClbits,
                                                 parameters_, gates_);
  }

  void registerCustomGate(std::string_view symbol, std::string_view name,
                          const std::vector<std::string>& formalParameters,
                          std::unique_ptr<CircuitWriter> definition) override {
    if (gates_->contains(symbol)) {
      throw std::runtime_error("Qiskit custom Gate '" + std::string(symbol) +
                               "' is already registered");
    }
    auto circuit = definition->finish();
    circuit.attr("name") = nb::str(name.data(), name.size());
    nb::list parameters;
    try {
      for (const auto& parameter : formalParameters) {
        parameters.append(pythonAttribute(
            circuit, "get_parameter",
            "Qiskit custom definition cannot resolve a formal parameter")(
            parameter));
      }
    } catch (const nb::python_error& error) {
      throwPythonError(
          "Qiskit custom definition cannot resolve a formal parameter", error);
    }
    auto gate = nb::module_::import_("qiskit.circuit")
                    .attr("Gate")(nb::str(name.data(), name.size()),
                                  circuit.attr("num_qubits"), parameters);
    gate.attr("definition") = circuit;
    gates_->try_emplace(symbol, std::move(gate));
  }

private:
  std::shared_ptr<OutputParameters> parameters_ =
      std::make_shared<OutputParameters>();
  std::shared_ptr<NativeGateRegistry> gates_ =
      std::make_shared<NativeGateRegistry>();
};

} // namespace

std::unique_ptr<VersionedTranslation>
MQT_QISKIT_VERSION_FACTORY() { // NOLINT(misc-use-internal-linkage): declared in
                               // the version registry.
  static const auto VERSION = [] {
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
      (MQT_QISKIT_VERSION_EXACT_API != 0 &&
       // QISKIT_VERSION_HEX uses signed bitwise operations in Qiskit's header.
       // NOLINTNEXTLINE(bugprone-signed-bitwise)
       VERSION != QISKIT_VERSION_HEX)) {
    throw std::runtime_error("Qiskit C API capsule version does not match the "
                             "selected " MQT_QISKIT_VERSION_LABEL
                             " translation");
  }
  return std::make_unique<NativeTranslation>();
}

} // namespace mqt::bindings::qiskit

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
// NOLINTEND(cppcoreguidelines-pro-type-cstyle-cast)
