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

#include "mlir/Dialect/QC/Translation/StandardGate.h"

#include <llvm/ADT/StringMap.h>
#include <nanobind/nanobind.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bindings::qiskit {

namespace nb = nanobind;

enum class OperationKind : uint8_t {
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
  std::string name;
  std::vector<uint32_t> bits;
};

/** Validate canonical register membership and return the leading loose bits. */
[[nodiscard]] uint32_t
validateRegisterLayout(const std::vector<Register>& registers, uint32_t total,
                       std::string_view kind);

inline constexpr size_t MAX_PARAMETER_EXPRESSION_DEPTH = 64U;
inline constexpr size_t MAX_PARAMETER_EXPRESSION_NODES = 4096U;
inline constexpr uint64_t MAX_PARAMETER_GROUP_SIZE = 65'536U;

/** Source-level vector metadata for one scalar parameter. */
struct ParameterGroup {
  std::string identity;
  std::string name;
  uint64_t index = 0U;
  uint64_t size = 0U;

  [[nodiscard]] bool operator==(const ParameterGroup&) const = default;
};

class ParameterGroupRegistry {
public:
  void add(const ParameterGroup& group);

private:
  llvm::StringMap<ParameterGroup> groups;
  uint64_t totalSize = 0U;
};

enum class UnaryParameterKind : uint8_t {
  Negate,
  Sin,
  Cos,
  Tan,
  ArcSin,
  ArcCos,
  ArcTan,
  Exp,
  Log,
  Abs,
  Conjugate,
};

enum class BinaryParameterKind : uint8_t {
  Add,
  Subtract,
  Multiply,
  Divide,
  Power,
};

/** One normalized scalar parameter-expression tree. */
class Parameter {
public:
  struct Number {
    double value;
  };

  struct Symbol {
    std::string name;
    std::optional<ParameterGroup> group;
  };

  struct Unary {
    UnaryParameterKind operation;
    std::shared_ptr<const Parameter> operand;
  };

  struct Binary {
    BinaryParameterKind operation;
    std::shared_ptr<const Parameter> left;
    std::shared_ptr<const Parameter> right;
  };

  Parameter() = default;

  [[nodiscard]] static Parameter number(const double value) {
    return Parameter(Number{value});
  }

  [[nodiscard]] static Parameter
  symbol(std::string name, std::optional<ParameterGroup> group = std::nullopt) {
    return Parameter(
        Symbol{.name = std::move(name), .group = std::move(group)});
  }

  [[nodiscard]] static Parameter unary(const UnaryParameterKind operation,
                                       Parameter operand) {
    return Parameter(Unary{
        .operation = operation,
        .operand = std::make_shared<const Parameter>(std::move(operand))});
  }

  [[nodiscard]] static Parameter binary(const BinaryParameterKind operation,
                                        Parameter left, Parameter right) {
    return Parameter(
        Binary{.operation = operation,
               .left = std::make_shared<const Parameter>(std::move(left)),
               .right = std::make_shared<const Parameter>(std::move(right))});
  }

  [[nodiscard]] const Number* getNumber() const {
    return std::get_if<Number>(&storage);
  }

  [[nodiscard]] const Symbol* getSymbol() const {
    return std::get_if<Symbol>(&storage);
  }

  [[nodiscard]] const Unary* getUnary() const {
    return std::get_if<Unary>(&storage);
  }

  [[nodiscard]] const Binary* getBinary() const {
    return std::get_if<Binary>(&storage);
  }

private:
  using Value = std::variant<Number, Symbol, Unary, Binary>;

  explicit Parameter(Value value) : storage(std::move(value)) {}

  Value storage = Number{0.0};
};

enum class GateModifierKind : uint8_t {
  Control,
  Inverse,
  Power,
};

struct GateModifier {
  GateModifierKind kind = GateModifierKind::Inverse;
  uint32_t numControls = 0;
  Parameter exponent;
};

struct StandardGateMapping {
  constexpr StandardGateMapping() = default;
  constexpr StandardGateMapping(const mlir::qc::StandardGate gate,
                                const uint32_t controls)
      : gate(gate), controls(controls) {}

  mlir::qc::StandardGate gate = mlir::qc::StandardGate::Id;
  uint32_t controls = 0;

  [[nodiscard]] bool operator==(const StandardGateMapping&) const = default;
};

struct Instruction {
  OperationKind kind = OperationKind::Unknown;
  std::string name;
  std::vector<uint32_t> qubits;
  std::vector<uint32_t> clbits;
  std::vector<Parameter> parameters;
  std::vector<GateModifier> modifiers;
  std::optional<StandardGateMapping> standardGate;
};

enum class ClassicalType : uint8_t {
  Bool,
  Uint,
  Float,
};
enum class ExpressionKind : uint8_t {
  Unary,
  Binary,
  Cast,
  Value,
  Index,
  ClassicalBit,
  ClassicalRegister,
};
enum class BinaryOperation : uint8_t {
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
enum class UnaryOperation : uint8_t {
  BitNot,
  LogicNot,
  Negate,
};

/** One normalized Qiskit classical-expression tree. */
struct Expression {
  ExpressionKind kind = ExpressionKind::Value;
  ClassicalType type = ClassicalType::Bool;
  uint32_t width = 1;
  BinaryOperation binaryOperation = BinaryOperation::Equal;
  UnaryOperation unaryOperation = UnaryOperation::LogicNot;
  bool boolValue = false;
  uint64_t uintValue = 0;
  double floatValue = 0.0;
  uint32_t bit = 0;
  Register reg;
  std::unique_ptr<Expression> left;
  std::unique_ptr<Expression> right;
};

enum class ControlFlowKind : uint8_t {
  Box,
  Break,
  Continue,
  For,
  IfElse,
  Switch,
  While,
};
enum class ClassicalTargetKind : uint8_t {
  ClassicalBit,
  ClassicalRegister,
  Expression,
};

struct ClassicalTarget {
  ClassicalTargetKind kind = ClassicalTargetKind::ClassicalBit;
  uint32_t bit = 0;
  bool expectedBit = false;
  Register reg;
  uint64_t expectedRegister = 0;
  uint32_t width = 1;
  std::unique_ptr<Expression> expression;
};

struct Loop {
  bool isRange = true;
  int64_t start = 0;
  int64_t stop = 0;
  int64_t step = 1;
  std::vector<int64_t> values;
  std::optional<Parameter> parameter;
};

struct SwitchCase {
  bool isDefault = false;
  std::vector<uint64_t> labels;
};

class ControlFlowReader;

class CircuitReader {
public:
  CircuitReader() = default;
  CircuitReader(const CircuitReader&) = delete;
  CircuitReader& operator=(const CircuitReader&) = delete;
  CircuitReader(CircuitReader&&) = delete;
  CircuitReader& operator=(CircuitReader&&) = delete;
  virtual ~CircuitReader() = default;

  [[nodiscard]] virtual uint32_t numQubits() const = 0;
  [[nodiscard]] virtual uint32_t numClbits() const = 0;
  [[nodiscard]] virtual size_t numInstructions() const = 0;
  [[nodiscard]] virtual size_t numQuantumRegisters() const = 0;
  [[nodiscard]] virtual size_t numClassicalRegisters() const = 0;
  [[nodiscard]] virtual bool hasClassicalVariables() const = 0;
  [[nodiscard]] virtual Register quantumRegister(size_t index) const = 0;
  [[nodiscard]] virtual Register classicalRegister(size_t index) const = 0;
  /** Return the circuit's free scalar parameters in a stable order. */
  [[nodiscard]] virtual std::vector<Parameter> parameters() const = 0;
  [[nodiscard]] virtual Parameter globalPhase() const = 0;
  [[nodiscard]] virtual Instruction instruction(size_t index) const = 0;
  [[nodiscard]] virtual std::vector<std::complex<double>>
  unitary(size_t index) const = 0;
  [[nodiscard]] virtual std::unique_ptr<ControlFlowReader>
  controlFlow(size_t index) const = 0;
  [[nodiscard]] virtual std::unique_ptr<CircuitReader>
  definition(size_t index) const = 0;
  [[nodiscard]] virtual uintptr_t definitionIdentity(size_t index) const = 0;
};

class ControlFlowReader {
public:
  ControlFlowReader() = default;
  ControlFlowReader(const ControlFlowReader&) = delete;
  ControlFlowReader& operator=(const ControlFlowReader&) = delete;
  ControlFlowReader(ControlFlowReader&&) = delete;
  ControlFlowReader& operator=(ControlFlowReader&&) = delete;
  virtual ~ControlFlowReader() = default;

  [[nodiscard]] virtual ControlFlowKind kind() const = 0;
  [[nodiscard]] virtual size_t numBlocks() const = 0;
  [[nodiscard]] virtual std::unique_ptr<CircuitReader>
  block(size_t index) const = 0;
  [[nodiscard]] virtual std::vector<uint32_t> qubitMap() const = 0;
  [[nodiscard]] virtual std::vector<uint32_t> clbitMap() const = 0;
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

  virtual void addQuantumRegister(std::string_view name, uint32_t size) = 0;
  virtual void addClassicalRegister(std::string_view name, uint32_t size) = 0;
  virtual void setGlobalPhase(const Parameter& phase) = 0;
  virtual void addGate(StandardGateMapping gate,
                       const std::vector<uint32_t>& qubits,
                       const std::vector<Parameter>& parameters) = 0;
  virtual void addMeasure(uint32_t qubit, uint32_t clbit) = 0;
  virtual void addReset(uint32_t qubit) = 0;
  virtual void addBarrier(const std::vector<uint32_t>& qubits) = 0;
  virtual void addUnitary(const std::vector<std::complex<double>>& matrix,
                          const std::vector<uint32_t>& qubits,
                          uint32_t numControls) = 0;
  virtual void
  addControlFlow(ControlFlowKind kind, ClassicalTarget target, Loop loop,
                 std::vector<SwitchCase> switchCases,
                 std::vector<std::unique_ptr<CircuitWriter>> blocks) = 0;
  /** Transfer the native circuit to a new owned Python QuantumCircuit. */
  [[nodiscard]] virtual nb::object finish() = 0;
};

class VersionedTranslation {
public:
  VersionedTranslation() = default;
  VersionedTranslation(const VersionedTranslation&) = delete;
  VersionedTranslation& operator=(const VersionedTranslation&) = delete;
  VersionedTranslation(VersionedTranslation&&) = delete;
  VersionedTranslation& operator=(VersionedTranslation&&) = delete;
  virtual ~VersionedTranslation() = default;

  [[nodiscard]] virtual std::unique_ptr<CircuitReader>
  openCircuit(nb::handle circuit) const = 0;
  [[nodiscard]] virtual bool supportsGate(StandardGateMapping gate) const = 0;
  [[nodiscard]] virtual std::unique_ptr<CircuitWriter>
  createCircuit(uint32_t looseQubits, uint32_t looseClbits) const = 0;
};

#define MQT_QISKIT_DECLARE_VERSION_IMPL(suffix)                                \
  [[nodiscard]] std::unique_ptr<VersionedTranslation> createQiskit##suffix();
#define MQT_QISKIT_DECLARE_VERSION(major, minor, suffix, minimumPatch,         \
                                   minimum, range)                             \
  MQT_QISKIT_DECLARE_VERSION_IMPL(suffix)
#define MQT_QISKIT_VERSION MQT_QISKIT_DECLARE_VERSION
#include "SupportedVersions.inc"
#undef MQT_QISKIT_VERSION
#undef MQT_QISKIT_DECLARE_VERSION
#undef MQT_QISKIT_DECLARE_VERSION_IMPL

#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
[[nodiscard]] std::unique_ptr<VersionedTranslation>
createCandidateTranslation();
#endif

} // namespace mqt::bindings::qiskit
