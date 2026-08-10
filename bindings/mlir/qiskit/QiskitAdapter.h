/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mqt::bindings::qiskit {

using PythonHandle = void;

// These value types cross the adapter boundary as explicit, zero-initialized
// aggregates. Keep that property visible even for members whose library types
// already default-initialize themselves.
// NOLINTBEGIN(readability-redundant-member-init)

enum class OperationKind : std::uint8_t { // NOLINT(performance-enum-size)
  Gate,
  Barrier,
  Delay,
  Measure,
  Reset,
  Unitary,
  ControlFlow,
  Unknown,
};

struct Register {
  std::string name{};
  std::vector<std::uint32_t> bits{};
};

enum class ParameterKind : std::uint8_t { // NOLINT(performance-enum-size)
  Number,
  Symbol,
  Expression,
};

struct Parameter {
  ParameterKind kind = ParameterKind::Number;
  double number = 0.0;
  std::string text{};
};

enum class GateModifierKind : std::uint8_t { // NOLINT(performance-enum-size)
  Control,
  Inverse,
  Power,
};

struct GateModifier {
  GateModifierKind kind = GateModifierKind::Inverse;
  std::uint32_t numControls = 0;
  Parameter exponent{.kind = ParameterKind::Number, .number = 1.0};
};

struct Instruction {
  OperationKind kind = OperationKind::Unknown;
  std::string name{};
  std::vector<std::uint32_t> qubits{};
  std::vector<std::uint32_t> clbits{};
  std::vector<Parameter> parameters{};
  std::vector<GateModifier> modifiers{};
};

enum class ClassicalType : std::uint8_t { // NOLINT(performance-enum-size)
  Bool,
  Uint,
  Float,
};
enum class ExpressionKind : std::uint8_t { // NOLINT(performance-enum-size)
  Unary,
  Binary,
  Cast,
  Value,
  Variable,
  Index,
};
enum class BinaryOperation : std::uint8_t { // NOLINT(performance-enum-size)
  BitAnd,
  BitOr,
  BitXor,
  LogicAnd,
  LogicOr,
  Equal,
  NotEqual,
  Less,
  LessEqual,
  Greater,
  GreaterEqual,
  ShiftLeft,
  ShiftRight,
  Add,
  Subtract,
  Multiply,
  Divide,
};
enum class UnaryOperation : std::uint8_t { // NOLINT(performance-enum-size)
  BitNot,
  LogicNot,
  Negate,
};

/** One normalized Qiskit classical-expression tree. */
struct Expression {
  ExpressionKind kind = ExpressionKind::Value;
  ClassicalType type = ClassicalType::Bool;
  std::uint32_t width = 1;
  BinaryOperation binaryOperation = BinaryOperation::Equal;
  UnaryOperation unaryOperation = UnaryOperation::LogicNot;
  bool boolValue = false;
  std::uint64_t uintValue = 0;
  double floatValue = 0.0;
  std::string variableName{};
  std::unique_ptr<Expression> left{};
  std::unique_ptr<Expression> right{};
};

enum class ControlFlowKind : std::uint8_t { // NOLINT(performance-enum-size)
  Box,
  Break,
  Continue,
  For,
  IfElse,
  Switch,
  While,
};
enum class ClassicalTargetKind : std::uint8_t { // NOLINT(performance-enum-size)
  ClassicalBit,
  ClassicalRegister,
  Expression,
};

struct ClassicalTarget {
  ClassicalTargetKind kind = ClassicalTargetKind::ClassicalBit;
  std::uint32_t bit = 0;
  bool expectedBit = false;
  Register reg;
  std::uint64_t expectedRegister = 0;
  std::uint32_t width = 1;
  std::unique_ptr<Expression> expression{};
};

struct Loop {
  bool isRange = true;
  std::int64_t start = 0;
  std::int64_t stop = 0;
  std::int64_t step = 1;
  std::vector<std::int64_t> values{};
  std::optional<std::string> parameter{};
};

struct SwitchCase {
  bool isDefault = false;
  std::vector<std::uint64_t> labels{};
};

// NOLINTEND(readability-redundant-member-init)

class ControlFlowView;

class CircuitView {
public:
  CircuitView() = default;
  CircuitView(const CircuitView&) = delete;
  CircuitView& operator=(const CircuitView&) = delete;
  CircuitView(CircuitView&&) = delete;
  CircuitView& operator=(CircuitView&&) = delete;
  virtual ~CircuitView() = default;

  [[nodiscard]] virtual std::uint32_t numQubits() const = 0;
  [[nodiscard]] virtual std::uint32_t numClbits() const = 0;
  [[nodiscard]] virtual std::size_t numInstructions() const = 0;
  [[nodiscard]] virtual std::size_t numQuantumRegisters() const = 0;
  [[nodiscard]] virtual std::size_t numClassicalRegisters() const = 0;
  [[nodiscard]] virtual Register quantumRegister(std::size_t index) const = 0;
  [[nodiscard]] virtual Register classicalRegister(std::size_t index) const = 0;
  [[nodiscard]] virtual Parameter globalPhase() const = 0;
  [[nodiscard]] virtual Instruction instruction(std::size_t index) const = 0;
  [[nodiscard]] virtual std::vector<std::complex<double>>
  unitary(std::size_t index) const = 0;
  [[nodiscard]] virtual std::unique_ptr<ControlFlowView>
  controlFlow(std::size_t index) const = 0;
};

class ControlFlowView {
public:
  ControlFlowView() = default;
  ControlFlowView(const ControlFlowView&) = delete;
  ControlFlowView& operator=(const ControlFlowView&) = delete;
  ControlFlowView(ControlFlowView&&) = delete;
  ControlFlowView& operator=(ControlFlowView&&) = delete;
  virtual ~ControlFlowView() = default;

  [[nodiscard]] virtual ControlFlowKind kind() const = 0;
  [[nodiscard]] virtual std::size_t numBlocks() const = 0;
  [[nodiscard]] virtual std::unique_ptr<CircuitView>
  block(std::size_t index) const = 0;
  [[nodiscard]] virtual std::vector<std::uint32_t> qubitMap() const = 0;
  [[nodiscard]] virtual std::vector<std::uint32_t> clbitMap() const = 0;
  [[nodiscard]] virtual ClassicalTarget condition() const = 0;
  [[nodiscard]] virtual Loop loop() const = 0;
  [[nodiscard]] virtual ClassicalTarget switchTarget() const = 0;
  [[nodiscard]] virtual std::vector<SwitchCase> switchCases() const = 0;
};

class CircuitWriter {
public:
  CircuitWriter() = default;
  CircuitWriter(const CircuitWriter&) = delete;
  CircuitWriter& operator=(const CircuitWriter&) = delete;
  CircuitWriter(CircuitWriter&&) = delete;
  CircuitWriter& operator=(CircuitWriter&&) = delete;
  virtual ~CircuitWriter() = default;

  virtual void addQuantumRegister(std::string_view name,
                                  std::uint32_t size) = 0;
  virtual void addClassicalRegister(std::string_view name,
                                    std::uint32_t size) = 0;
  virtual void setGlobalPhase(const Parameter& parameter) = 0;
  virtual void addGate(std::string_view name,
                       const std::vector<std::uint32_t>& qubits,
                       const std::vector<Parameter>& parameters) = 0;
  virtual void addMeasure(std::uint32_t qubit, std::uint32_t clbit) = 0;
  virtual void addReset(std::uint32_t qubit) = 0;
  virtual void addBarrier(const std::vector<std::uint32_t>& qubits) = 0;
  virtual void addUnitary(const std::vector<std::complex<double>>& matrix,
                          const std::vector<std::uint32_t>& qubits) = 0;

  /** Transfer the native circuit to a new owned Python QuantumCircuit. */
  [[nodiscard]] virtual PythonHandle* finish() = 0;
};

class Adapter {
public:
  Adapter() = default;
  Adapter(const Adapter&) = delete;
  Adapter& operator=(const Adapter&) = delete;
  Adapter(Adapter&&) = delete;
  Adapter& operator=(Adapter&&) = delete;
  virtual ~Adapter() = default;

  [[nodiscard]] virtual std::unique_ptr<CircuitView>
  openCircuit(PythonHandle* circuit) const = 0;
  [[nodiscard]] virtual std::unique_ptr<CircuitWriter>
  createCircuit(std::uint32_t looseQubits, std::uint32_t looseClbits) const = 0;
};

#define MQT_QISKIT_DECLARE_ADAPTER_IMPL(suffix)                                \
  [[nodiscard]] std::unique_ptr<Adapter> createAdapter##suffix();
#define MQT_QISKIT_DECLARE_ADAPTER(major, minor, suffix, minimum, range)       \
  MQT_QISKIT_DECLARE_ADAPTER_IMPL(suffix)
#define MQT_QISKIT_ADAPTER MQT_QISKIT_DECLARE_ADAPTER
#include "SupportedAdapters.inc"
#undef MQT_QISKIT_ADAPTER
#undef MQT_QISKIT_DECLARE_ADAPTER
#undef MQT_QISKIT_DECLARE_ADAPTER_IMPL

#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
[[nodiscard]] std::unique_ptr<Adapter> createCandidateAdapter();
#endif

} // namespace mqt::bindings::qiskit
