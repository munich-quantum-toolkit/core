/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"
#include "ir/Permutation.hpp"
#include "qasm3/InstVisitor.hpp"
#include "qasm3/NestedEnvironment.hpp"
#include "qasm3/Statement_fwd.hpp"
#include "qasm3/passes/ConstEvalPass.hpp"
#include "qasm3/passes/TypeCheckPass.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// forward declarations
namespace qc {
enum ComparisonKind : std::uint8_t;
class ClassicalRegister;
struct Control;
class QuantumRegister;
class Operation;
class QuantumComputation;
using QuantumRegisterMap = std::unordered_map<std::string, QuantumRegister>;
} // namespace qc
namespace qasm3 {
class Statement;
struct Gate;

class Importer final : public InstVisitor {
public:
  /**
   * Imports a QASM3 file into a @ref qc::QuantumComputation instance
   * @param filename The path to the QASM3 file to import
   * @return The imported @ref qc::QuantumComputation instance
   */
  [[nodiscard]] static auto importf(const std::string& filename)
      -> qc::QuantumComputation;

  /**
   * Imports a QASM3 program from a string into a @ref qc::QuantumComputation
   * @param qasm The QASM3 program to import
   * @return The imported @ref qc::QuantumComputation instance
   */
  [[nodiscard]] static auto imports(const std::string& qasm)
      -> qc::QuantumComputation;

  /**
   * Imports a QASM3 program from a stream into a @ref qc::QuantumComputation
   * @param is The input stream to read the QASM3 program from
   * @return The imported @ref qc::QuantumComputation instance
   */
  [[nodiscard]] static auto import(std::istream& is) -> qc::QuantumComputation;

private:
  /**
   * @brief Construct a new instance for importing QASM3 code
   * @param quantumComputation The @ref qc::QuantumComputation to import the
   * QASM3 code into.
   */
  explicit Importer(qc::QuantumComputation& quantumComputation);

  /**
   * @brief Import the given QASM3 program into the @ref qc::QuantumComputation
   * @param program The parsed QASM3 program AST
   */
  void visitProgram(const std::vector<std::shared_ptr<Statement>>& program);

  const_eval::ConstEvalPass constEvalPass;
  type_checking::TypeCheckPass typeCheckPass;

  NestedEnvironment<std::shared_ptr<DeclarationStatement>> declarations;
  qc::QuantumComputation* qc{};

  std::map<std::string, std::shared_ptr<Gate>> gates;

  bool openQASM2CompatMode{false};

  qc::Permutation initialLayout;
  qc::Permutation outputPermutation;

  static std::map<std::string, std::pair<const_eval::ConstEvalValue,
                                         type_checking::InferredType>>
  initializeBuiltins();

  void translateGateOperand(const std::shared_ptr<GateOperand>& gateOperand,
                            std::vector<qc::Qubit>& qubits,
                            const qc::QuantumRegisterMap& qregs,
                            const std::shared_ptr<DebugInfo>& debugInfo) const;

  void translateBitOperand(
      const std::shared_ptr<IndexedIdentifier>& indexedIdentifier,
      std::vector<qc::Bit>& bits,
      const std::shared_ptr<DebugInfo>& debugInfo) const;

  /**
   * @brief Translates a condition expression
   * @param condition The condition expression to translate.
   * @param debugInfo The debug information of the condition expression.
   * @return Either a pair of a bit and a boolean value, or a triple of a
   * classical register, a comparison kind, and an integer value.
   * @throws CompilerError If the condition is neither of the expected types.
   */
  [[nodiscard]] std::variant<
      std::pair<qc::Bit, bool>,
      std::tuple<qc::ClassicalRegister, qc::ComparisonKind, uint64_t>>
  translateCondition(const std::shared_ptr<Expression>& condition,
                     const std::shared_ptr<DebugInfo>& debugInfo) const;

  static uint64_t
  evaluatePositiveConstant(const std::shared_ptr<Expression>& expr,
                           const std::shared_ptr<DebugInfo>& debugInfo,
                           uint64_t defaultValue = 0);

  static int64_t
  evaluateIntegerConstant(const std::shared_ptr<Expression>& expr,
                          const std::shared_ptr<DebugInfo>& debugInfo,
                          int64_t defaultValue = 0);

  void visitVersionDeclaration(
      std::shared_ptr<VersionDeclaration> versionDeclaration) override;

  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override;

  void visitAssignmentStatement(
      std::shared_ptr<AssignmentStatement> assignmentStatement) override;

  void visitInitialLayout(std::shared_ptr<InitialLayout> layout) override;

  void visitOutputPermutation(
      std::shared_ptr<OutputPermutation> permutation) override;

  void
  visitGateStatement(std::shared_ptr<GateDeclaration> gateStatement) override;

  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override;

  auto
  evaluateGateCall(const std::shared_ptr<GateCallStatement>& gateCallStatement,
                   const std::string& identifier,
                   const std::vector<std::shared_ptr<Expression>>& parameters,
                   std::vector<std::shared_ptr<GateOperand>> targets,
                   const qc::QuantumRegisterMap& qregs)
      -> std::unique_ptr<qc::Operation>;

  static std::shared_ptr<Gate>
  getMcGateDefinition(const std::string& identifier, size_t operandSize,
                      const std::shared_ptr<DebugInfo>& debugInfo);

  auto applyQuantumOperation(const std::shared_ptr<Gate>& gate,
                             const qc::Targets& targetBits,
                             const std::vector<qc::Control>& controlBits,
                             const std::vector<qc::fp>& evaluatedParameters,
                             bool invertOperation,
                             const std::shared_ptr<DebugInfo>& debugInfo)
      -> std::unique_ptr<qc::Operation>;

  void visitMeasureAssignment(
      const std::shared_ptr<IndexedIdentifier>& indexedIdentifier,
      const std::shared_ptr<MeasureExpression>& measureExpression,
      const std::shared_ptr<DebugInfo>& debugInfo);

  void visitBarrierStatement(
      std::shared_ptr<BarrierStatement> barrierStatement) override;

  void
  visitResetStatement(std::shared_ptr<ResetStatement> resetStatement) override;

  void visitIfStatement(std::shared_ptr<IfStatement> ifStatement) override;

  [[nodiscard]] auto translateBlockOperations(
      const std::vector<std::shared_ptr<Statement>>& statements)
      -> std::unique_ptr<qc::Operation>;

  std::pair<std::string, size_t>
  parseGateIdentifierCompatMode(const std::string& identifier);
};
} // namespace qasm3
