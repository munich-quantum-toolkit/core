/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Target/OpenQASM/Frontend.h"

#include "ir/Definitions.hpp"
#include "mlir/Dialect/OQ3/IR/GateCatalog.h"
#include "qasm3/Exception.hpp"
#include "qasm3/Parser.hpp"
#include "qasm3/Statement.hpp"
#include "qasm3/Types.hpp"
#include "qasm3/passes/ConstEvalPass.hpp"
#include "qasm3/passes/TypeCheckPass.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::oq3::frontend {

struct ParsedProgram::Impl {
  std::vector<std::shared_ptr<qasm3::Statement>> statements;
  std::vector<std::string> includedFiles;
  std::size_t implicitStatementCount = 0;
};

ParsedProgram::ParsedProgram(std::unique_ptr<Impl> implementation)
    : impl(std::move(implementation)) {}
ParsedProgram::ParsedProgram(ParsedProgram&&) noexcept = default;
ParsedProgram& ParsedProgram::operator=(ParsedProgram&&) noexcept = default;
ParsedProgram::~ParsedProgram() = default;

namespace {

class FrontendError final : public std::runtime_error {
public:
  Diagnostic diagnostic;

  explicit FrontendError(Diagnostic value)
      : std::runtime_error(value.message), diagnostic(std::move(value)) {}
};

SourceLocation locationOf(const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
  if (!debugInfo) {
    return {};
  }
  return {.filename = debugInfo->filename,
          .line = static_cast<std::uint32_t>(debugInfo->line),
          .column = static_cast<std::uint32_t>(debugInfo->column)};
}

Diagnostic diagnosticOf(const qasm3::CompilerError& error) {
  return {.location = locationOf(error.debugInfo), .message = error.what()};
}

[[noreturn]] void fail(const std::shared_ptr<qasm3::DebugInfo>& debugInfo,
                       std::string message) {
  throw FrontendError(
      {.location = locationOf(debugInfo), .message = std::move(message)});
}

struct GateSignature {
  std::size_t parameterCount = 0;
  std::size_t qubitCount = 0;
  bool variadicControls = false;
};

struct OperandSelection {
  std::vector<QubitReference> qubits;
};

class SemanticAnalyzer {
public:
  SemanticAnalyzer(
      const std::vector<std::shared_ptr<qasm3::Statement>>& parsedStatements,
      const std::vector<std::string>& parsedIncludes,
      const std::size_t parserImplicitStatementCount,
      const FrontendOptions& frontendOptions)
      : statements(parsedStatements), options(frontendOptions),
        implicitStatementCount(parserImplicitStatementCount),
        typeCheckPass(constEvalPass) {
    program.gatePolicy = options.gatePolicy;
    program.standardLibraryIncluded =
        llvm::is_contained(parsedIncludes, "stdgates.inc");
    initializeBuiltins();
  }

  std::unique_ptr<TypedProgram> run() {
    for (const auto [index, statement] : llvm::enumerate(statements)) {
      if (index < implicitStatementCount) {
        continue;
      }
      constEvalPass.processStatement(*statement);
      typeCheckPass.processStatement(*statement);
      analyzeTopLevelStatement(statement);
    }
    finalizeOutputs();
    return std::make_unique<TypedProgram>(std::move(program));
  }

private:
  const std::vector<std::shared_ptr<qasm3::Statement>>& statements;
  FrontendOptions options;
  std::size_t implicitStatementCount;
  TypedProgram program;
  qasm3::const_eval::ConstEvalPass constEvalPass;
  qasm3::type_checking::TypeCheckPass typeCheckPass;
  llvm::StringMap<RegisterId> registerIds;
  llvm::StringMap<GateSignature> customGates;
  llvm::StringMap<std::uint32_t> gateParameters;
  llvm::StringMap<std::uint32_t> gateQubits;
  std::vector<std::vector<bool>> measuredBits;
  std::vector<RegisterId> bitRegisters;
  std::vector<RegisterId> explicitOutputs;
  bool insideGate = false;
  bool versionSeen = false;

  [[nodiscard]] bool isGateAvailable(const GateCatalogEntry& gate) const {
    if (gate.availability == GateAvailability::Language) {
      return true;
    }
    if (options.gatePolicy == GatePolicy::MQTCompatibility) {
      return true;
    }
    return gate.availability == GateAvailability::StandardLibrary &&
           program.standardLibraryIncluded;
  }

  void initializeBuiltins() {
    using qasm3::DesignatedType;
    using qasm3::ResolvedType;
    using qasm3::const_eval::ConstEvalValue;
    using qasm3::type_checking::InferredType;

    const auto floatType = InferredType{std::dynamic_pointer_cast<ResolvedType>(
        DesignatedType<std::uint64_t>::getFloatTy(64))};
    auto add = [&](const std::string& name, const double value) {
      constEvalPass.addConst(name, ConstEvalValue(value));
      typeCheckPass.addBuiltin(name, floatType);
    };
    add("pi", ::qc::PI);
    add("π", ::qc::PI);
    add("tau", ::qc::TAU);
    add("τ", ::qc::TAU);
    add("euler", ::qc::E);
    add("ℇ", ::qc::E);
  }

  StatementId addStatement(StatementData data,
                           const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    const auto id = static_cast<StatementId>(program.statements.size());
    program.statements.push_back(
        {.data = std::move(data), .location = locationOf(debugInfo)});
    return id;
  }

  ExpressionId addExpression(ScalarExpression expression) {
    const auto id = static_cast<ExpressionId>(program.expressions.size());
    program.expressions.push_back(std::move(expression));
    return id;
  }

  ExpressionId addConstant(const qasm3::const_eval::ConstEvalValue& value) {
    ScalarExpression expression;
    expression.kind = ExpressionKind::Constant;
    switch (value.type) {
    case qasm3::const_eval::ConstEvalValue::ConstBool:
      expression.type = ScalarType::Bool;
      expression.constant = std::get<bool>(value.value);
      break;
    case qasm3::const_eval::ConstEvalValue::ConstInt:
      expression.type = ScalarType::Int;
      expression.constant = std::get<std::int64_t>(value.value);
      break;
    case qasm3::const_eval::ConstEvalValue::ConstUint:
      expression.type = ScalarType::Uint;
      expression.constant =
          static_cast<std::uint64_t>(std::get<std::int64_t>(value.value));
      break;
    case qasm3::const_eval::ConstEvalValue::ConstFloat:
      expression.type = ScalarType::Float;
      expression.constant = std::get<double>(value.value);
      break;
    }
    return addExpression(std::move(expression));
  }

  ExpressionId
  convertExpression(const std::shared_ptr<qasm3::Expression>& expression,
                    const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    if (const auto evaluated = constEvalPass.visit(expression)) {
      return addConstant(*evaluated);
    }

    if (const auto identifier =
            std::dynamic_pointer_cast<qasm3::IdentifierExpression>(
                expression)) {
      const auto parameter = gateParameters.find(identifier->identifier);
      if (parameter == gateParameters.end()) {
        fail(debugInfo, "Unknown nonconstant scalar expression '" +
                            identifier->identifier + "'.");
      }
      return addExpression({.kind = ExpressionKind::GateParameter,
                            .type = ScalarType::Float,
                            .parameter = parameter->second});
    }
    if (const auto identifier =
            std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(expression)) {
      if (!identifier->indices.empty()) {
        fail(debugInfo,
             "Indexed scalar expressions are not implemented in this "
             "milestone.");
      }
      const auto parameter = gateParameters.find(identifier->identifier);
      if (parameter == gateParameters.end()) {
        fail(debugInfo, "Unknown nonconstant scalar expression '" +
                            identifier->identifier + "'.");
      }
      return addExpression({.kind = ExpressionKind::GateParameter,
                            .type = ScalarType::Float,
                            .parameter = parameter->second});
    }

    if (const auto unary =
            std::dynamic_pointer_cast<qasm3::UnaryExpression>(expression)) {
      ExpressionKind kind;
      switch (unary->op) {
      case qasm3::UnaryExpression::Negate:
        kind = ExpressionKind::Negate;
        break;
      case qasm3::UnaryExpression::BitwiseNot:
        kind = ExpressionKind::BitwiseNot;
        break;
      case qasm3::UnaryExpression::LogicalNot:
        kind = ExpressionKind::LogicalNot;
        break;
      case qasm3::UnaryExpression::Sin:
        kind = ExpressionKind::Sin;
        break;
      case qasm3::UnaryExpression::Cos:
        kind = ExpressionKind::Cos;
        break;
      case qasm3::UnaryExpression::Tan:
        kind = ExpressionKind::Tan;
        break;
      case qasm3::UnaryExpression::Exp:
        kind = ExpressionKind::Exp;
        break;
      case qasm3::UnaryExpression::Ln:
        kind = ExpressionKind::Ln;
        break;
      case qasm3::UnaryExpression::Sqrt:
        kind = ExpressionKind::Sqrt;
        break;
      default:
        fail(debugInfo,
             "This scalar unary expression is not supported by OQ3 emission.");
      }
      const auto operand = convertExpression(unary->operand, debugInfo);
      const auto type =
          kind == ExpressionKind::LogicalNot ? ScalarType::Bool
          : kind == ExpressionKind::BitwiseNot || kind == ExpressionKind::Negate
              ? program.expressions[operand].type
              : ScalarType::Float;
      return addExpression({.kind = kind, .type = type, .lhs = operand});
    }

    if (const auto binary =
            std::dynamic_pointer_cast<qasm3::BinaryExpression>(expression)) {
      ExpressionKind kind;
      switch (binary->op) {
      case qasm3::BinaryExpression::Add:
        kind = ExpressionKind::Add;
        break;
      case qasm3::BinaryExpression::Subtract:
        kind = ExpressionKind::Subtract;
        break;
      case qasm3::BinaryExpression::Multiply:
        kind = ExpressionKind::Multiply;
        break;
      case qasm3::BinaryExpression::Divide:
        kind = ExpressionKind::Divide;
        break;
      case qasm3::BinaryExpression::Power:
        kind = ExpressionKind::Power;
        break;
      default:
        fail(debugInfo,
             "This scalar binary expression is not supported by OQ3 emission.");
      }
      const auto lhs = convertExpression(binary->lhs, debugInfo);
      const auto rhs = convertExpression(binary->rhs, debugInfo);
      const auto lhsType = program.expressions[lhs].type;
      const auto rhsType = program.expressions[rhs].type;
      const auto type =
          lhsType == ScalarType::Float || rhsType == ScalarType::Float
              ? ScalarType::Float
              : lhsType;
      return addExpression(
          {.kind = kind, .type = type, .lhs = lhs, .rhs = rhs});
    }

    fail(debugInfo, "Unsupported scalar expression in OpenQASM frontend.");
  }

  std::uint64_t
  evaluateUnsigned(const std::shared_ptr<qasm3::Expression>& expression,
                   const std::shared_ptr<qasm3::DebugInfo>& debugInfo,
                   const std::uint64_t defaultValue = 0) {
    if (!expression) {
      return defaultValue;
    }
    const auto evaluated = constEvalPass.visit(expression);
    if (!evaluated ||
        (evaluated->type != qasm3::const_eval::ConstEvalValue::ConstInt &&
         evaluated->type != qasm3::const_eval::ConstEvalValue::ConstUint)) {
      fail(debugInfo, "Expected a constant integer expression.");
    }
    const auto value = std::get<std::int64_t>(evaluated->value);
    if (value < 0) {
      fail(debugInfo, "Expected a nonnegative integer expression.");
    }
    return static_cast<std::uint64_t>(value);
  }

  void
  analyzeTopLevelStatement(const std::shared_ptr<qasm3::Statement>& statement) {
    if (const auto version =
            std::dynamic_pointer_cast<qasm3::VersionDeclaration>(statement)) {
      analyzeVersion(version);
      return;
    }
    if (const auto gate =
            std::dynamic_pointer_cast<qasm3::GateDeclaration>(statement)) {
      analyzeGateDefinition(gate);
      return;
    }
    if (const auto declaration =
            std::dynamic_pointer_cast<qasm3::DeclarationStatement>(statement)) {
      analyzeDeclaration(declaration);
      return;
    }
    if (std::dynamic_pointer_cast<qasm3::InitialLayout>(statement) ||
        std::dynamic_pointer_cast<qasm3::OutputPermutation>(statement)) {
      return;
    }
    if (const auto gate =
            std::dynamic_pointer_cast<qasm3::GateCallStatement>(statement)) {
      auto applications = analyzeGateApplication(gate);
      for (auto& application : applications) {
        program.body.push_back(
            addStatement(std::move(application), gate->debugInfo));
      }
      return;
    }
    program.body.push_back(analyzeRuntimeStatement(statement, false));
  }

  void analyzeVersion(
      const std::shared_ptr<qasm3::VersionDeclaration>& declaration) {
    if (versionSeen) {
      fail(declaration->debugInfo,
           "OpenQASM source contains more than one version declaration.");
    }
    versionSeen = true;
    if (std::abs(declaration->version - 2.0) < 0.001) {
      program.openQASM2 = true;
      return;
    }
    if (std::abs(declaration->version - 3.0) < 0.001 ||
        std::abs(declaration->version - 3.1) < 0.001) {
      program.openQASM2 = false;
      return;
    }
    fail(declaration->debugInfo, "Unsupported OpenQASM version " +
                                     std::to_string(declaration->version) +
                                     ".");
  }

  void analyzeDeclaration(
      const std::shared_ptr<qasm3::DeclarationStatement>& declaration) {
    if (declaration->isConst) {
      return;
    }
    if (registerIds.contains(declaration->identifier)) {
      fail(declaration->debugInfo,
           "Identifier '" + declaration->identifier + "' already declared.");
    }
    const auto* resolved =
        std::get_if<std::shared_ptr<qasm3::ResolvedType>>(&declaration->type);
    if (!resolved || !*resolved) {
      fail(declaration->debugInfo, "Declaration type was not resolved.");
    }
    const auto sized =
        std::dynamic_pointer_cast<qasm3::DesignatedType<std::uint64_t>>(
            *resolved);
    if (!sized || sized->getDesignator() == 0) {
      fail(declaration->debugInfo,
           "Only nonempty sized declarations are supported by MLIR emission.");
    }

    RegisterKind kind;
    switch (sized->type) {
    case qasm3::Qubit:
      kind = RegisterKind::Qubit;
      break;
    case qasm3::Bit:
      kind = RegisterKind::Bit;
      break;
    case qasm3::Int:
    case qasm3::Uint:
      fail(declaration->debugInfo,
           "Integer declarations are not implemented by the current OQ3 "
           "foundation.");
    default:
      fail(declaration->debugInfo,
           "Unsupported declaration type for the current OQ3 foundation.");
    }

    const auto id = static_cast<RegisterId>(program.registers.size());
    registerIds[declaration->identifier] = id;
    program.registers.push_back(
        {.id = id,
         .kind = kind,
         .name = declaration->identifier,
         .width = sized->getDesignator(),
         .output = declaration->isOutput,
         .location = locationOf(declaration->debugInfo)});
    measuredBits.emplace_back(sized->getDesignator(), false);
    if (kind == RegisterKind::Bit) {
      bitRegisters.push_back(id);
      if (declaration->isOutput || program.openQASM2) {
        explicitOutputs.push_back(id);
      }
    }
    program.body.push_back(
        addStatement(DeclarationStatement{.reg = id}, declaration->debugInfo));

    if (!declaration->expression) {
      return;
    }
    const auto measurement =
        std::dynamic_pointer_cast<qasm3::MeasureExpression>(
            declaration->expression->expression);
    if (!measurement) {
      fail(declaration->debugInfo,
           "Only measurement initializers are supported in this milestone.");
    }
    auto target =
        std::make_shared<qasm3::IndexedIdentifier>(declaration->identifier);
    program.body.push_back(
        analyzeMeasurement(target, measurement, declaration->debugInfo));
  }

  void analyzeGateDefinition(
      const std::shared_ptr<qasm3::GateDeclaration>& declaration) {
    const auto* catalogGate = lookupGate(declaration->identifier);
    if ((catalogGate && isGateAvailable(*catalogGate)) ||
        customGates.contains(declaration->identifier)) {
      fail(declaration->debugInfo,
           "Gate '" + declaration->identifier + "' already declared.");
    }
    if (declaration->isOpaque) {
      fail(declaration->debugInfo, "Opaque gate '" + declaration->identifier +
                                       "' has no target implementation.");
    }

    GateDefinition definition;
    definition.name = declaration->identifier;
    definition.location = locationOf(declaration->debugInfo);
    for (const auto& parameter : declaration->parameters->identifiers) {
      if (gateParameters.contains(parameter->identifier)) {
        fail(declaration->debugInfo, "Gate parameter '" +
                                         parameter->identifier +
                                         "' is declared more than once.");
      }
      const auto index =
          static_cast<std::uint32_t>(definition.parameterNames.size());
      gateParameters[parameter->identifier] = index;
      definition.parameterNames.push_back(parameter->identifier);
    }
    for (const auto& qubit : declaration->qubits->identifiers) {
      if (gateQubits.contains(qubit->identifier)) {
        fail(declaration->debugInfo, "Gate qubit '" + qubit->identifier +
                                         "' is declared more than once.");
      }
      const auto index =
          static_cast<std::uint32_t>(definition.qubitNames.size());
      gateQubits[qubit->identifier] = index;
      definition.qubitNames.push_back(qubit->identifier);
    }

    insideGate = true;
    for (const auto& statement : declaration->statements) {
      const auto call =
          std::dynamic_pointer_cast<qasm3::GateCallStatement>(statement);
      if (!call) {
        fail(declaration->debugInfo,
             "Gate bodies currently support gate applications only.");
      }
      auto applications = analyzeGateApplication(call);
      definition.body.insert(definition.body.end(),
                             std::make_move_iterator(applications.begin()),
                             std::make_move_iterator(applications.end()));
    }
    insideGate = false;
    gateParameters.clear();
    gateQubits.clear();

    customGates[definition.name] = {.parameterCount =
                                        definition.parameterNames.size(),
                                    .qubitCount = definition.qubitNames.size()};
    program.gates.push_back(std::move(definition));
  }

  StatementId
  analyzeRuntimeStatement(const std::shared_ptr<qasm3::Statement>& statement,
                          const bool nested) {
    if (const auto gate =
            std::dynamic_pointer_cast<qasm3::GateCallStatement>(statement)) {
      auto applications = analyzeGateApplication(gate);
      if (applications.size() != 1) {
        fail(gate->debugInfo,
             "Broadcast gate applications inside conditionals are not yet "
             "represented as one statement.");
      }
      return addStatement(std::move(applications.front()), gate->debugInfo);
    }
    if (const auto assignment =
            std::dynamic_pointer_cast<qasm3::AssignmentStatement>(statement)) {
      const auto measurement =
          std::dynamic_pointer_cast<qasm3::MeasureExpression>(
              assignment->expression->expression);
      if (!measurement) {
        fail(assignment->debugInfo,
             "Classical assignments are not implemented in this milestone.");
      }
      if (nested) {
        fail(assignment->debugInfo,
             "Measurements inside conditionals require explicit carried "
             "classical state and are not implemented yet.");
      }
      return analyzeMeasurement(assignment->identifier, measurement,
                                assignment->debugInfo);
    }
    if (const auto reset =
            std::dynamic_pointer_cast<qasm3::ResetStatement>(statement)) {
      return analyzeReset(reset);
    }
    if (const auto barrier =
            std::dynamic_pointer_cast<qasm3::BarrierStatement>(statement)) {
      return analyzeBarrier(barrier);
    }
    if (const auto conditional =
            std::dynamic_pointer_cast<qasm3::IfStatement>(statement)) {
      return analyzeIf(conditional);
    }
    fail(statement->debugInfo,
         "Unsupported runtime statement in the typed OpenQASM frontend.");
  }

  StatementId analyzeMeasurement(
      const std::shared_ptr<qasm3::IndexedIdentifier>& target,
      const std::shared_ptr<qasm3::MeasureExpression>& measurement,
      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    auto bits = resolveBits(target, debugInfo);
    auto qubits = resolveQubitOperand(measurement->gate, debugInfo).qubits;
    if (bits.size() != qubits.size()) {
      fail(debugInfo,
           "Measurement target and qubit operand must have the same width.");
    }
    for (const auto bit : bits) {
      measuredBits[bit.reg][bit.index] = true;
    }
    return addStatement(MeasurementStatement{.targets = std::move(bits),
                                             .qubits = std::move(qubits)},
                        debugInfo);
  }

  StatementId
  analyzeReset(const std::shared_ptr<qasm3::ResetStatement>& reset) {
    return addStatement(
        ResetStatement{
            .qubits =
                resolveQubitOperand(reset->gate, reset->debugInfo).qubits},
        reset->debugInfo);
  }

  StatementId
  analyzeBarrier(const std::shared_ptr<qasm3::BarrierStatement>& barrier) {
    std::vector<QubitReference> qubits;
    for (const auto& operand : barrier->gates) {
      auto resolved = resolveQubitOperand(operand, barrier->debugInfo).qubits;
      qubits.insert(qubits.end(), resolved.begin(), resolved.end());
    }
    return addStatement(BarrierStatement{.qubits = std::move(qubits)},
                        barrier->debugInfo);
  }

  StatementId
  analyzeIf(const std::shared_ptr<qasm3::IfStatement>& conditional) {
    if (conditional->thenStatements.empty() &&
        conditional->elseStatements.empty()) {
      fail(conditional->debugInfo,
           "If statement must contain a nonempty branch.");
    }
    auto [condition, negated] =
        analyzeCondition(conditional->condition, conditional->debugInfo);
    IfStatement result{.condition = condition, .negated = negated};
    for (const auto& statement : conditional->thenStatements) {
      result.thenStatements.push_back(analyzeRuntimeStatement(statement, true));
    }
    for (const auto& statement : conditional->elseStatements) {
      result.elseStatements.push_back(analyzeRuntimeStatement(statement, true));
    }
    return addStatement(std::move(result), conditional->debugInfo);
  }

  std::pair<BitReference, bool>
  analyzeCondition(const std::shared_ptr<qasm3::Expression>& expression,
                   const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    if (const auto identifier =
            std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(expression)) {
      auto bits = resolveBits(identifier, debugInfo);
      if (bits.size() != 1) {
        fail(debugInfo, "If condition must select exactly one classical bit.");
      }
      ensureMeasured(bits.front(), debugInfo);
      return {bits.front(), false};
    }
    if (const auto unary =
            std::dynamic_pointer_cast<qasm3::UnaryExpression>(expression)) {
      if (unary->op != qasm3::UnaryExpression::LogicalNot &&
          unary->op != qasm3::UnaryExpression::BitwiseNot) {
        fail(debugInfo, "Only ! and ~ unary conditions are implemented.");
      }
      const auto identifier =
          std::dynamic_pointer_cast<qasm3::IndexedIdentifier>(unary->operand);
      if (!identifier) {
        fail(debugInfo, "Unary condition must operate on one classical bit.");
      }
      auto bits = resolveBits(identifier, debugInfo);
      if (bits.size() != 1) {
        fail(debugInfo, "If condition must select exactly one classical bit.");
      }
      ensureMeasured(bits.front(), debugInfo);
      return {bits.front(), true};
    }
    fail(debugInfo, "Unsupported condition expression in if statement.");
  }

  std::vector<GateApplication> analyzeGateApplication(
      const std::shared_ptr<qasm3::GateCallStatement>& call) {
    std::string callee = call->identifier;
    const GateCatalogEntry* standard = lookupGate(callee);
    auto custom = customGates.find(callee);
    std::uint64_t compatibilityControls = 0;
    if (!standard && custom == customGates.end() && program.openQASM2) {
      std::string stripped = callee;
      while (!stripped.empty() && stripped.front() == 'c') {
        stripped.erase(stripped.begin());
        ++compatibilityControls;
      }
      standard = lookupGate(stripped);
      custom = customGates.find(stripped);
      if (standard || custom != customGates.end()) {
        callee = std::move(stripped);
      }
    }

    if (standard && !isGateAvailable(*standard)) {
      standard = nullptr;
    }
    if (!standard && custom == customGates.end()) {
      fail(call->debugInfo,
           "No OpenQASM definition found for gate '" + call->identifier + "'.");
    }

    GateSignature signature;
    if (standard) {
      signature = {.parameterCount = standard->parameterCount,
                   .qubitCount = standard->qubitCount(),
                   .variadicControls = standard->variadicControls};
    } else {
      signature = custom->second;
    }
    if (signature.parameterCount != call->arguments.size()) {
      fail(call->debugInfo,
           "Invalid number of parameters for gate '" + call->identifier + "'.");
    }

    std::vector<ExpressionId> parameters;
    parameters.reserve(call->arguments.size());
    for (const auto& argument : call->arguments) {
      parameters.push_back(convertExpression(argument, call->debugInfo));
    }

    std::vector<GateModifier> modifiers;
    std::uint64_t addedControls = compatibilityControls;
    for (const auto& modifier : call->modifiers) {
      if (std::dynamic_pointer_cast<qasm3::InvGateModifier>(modifier)) {
        modifiers.push_back({.kind = ModifierKind::Inv});
        continue;
      }
      if (const auto control =
              std::dynamic_pointer_cast<qasm3::CtrlGateModifier>(modifier)) {
        const auto count =
            evaluateUnsigned(control->expression, call->debugInfo, 1);
        if (count == 0) {
          fail(call->debugInfo, "Gate control count must be positive.");
        }
        addedControls += count;
        std::optional<ExpressionId> operand;
        if (control->expression || count != 1) {
          operand = addConstant(qasm3::const_eval::ConstEvalValue(
              static_cast<std::int64_t>(count), false));
        }
        modifiers.push_back({.kind = control->ctrlType ? ModifierKind::Ctrl
                                                       : ModifierKind::NegCtrl,
                             .operand = operand});
        continue;
      }
      if (const auto power =
              std::dynamic_pointer_cast<qasm3::PowGateModifier>(modifier)) {
        modifiers.push_back(
            {.kind = ModifierKind::Pow,
             .operand = convertExpression(power->expression, call->debugInfo)});
        continue;
      }
      fail(call->debugInfo, "Unknown gate modifier.");
    }
    if (compatibilityControls != 0) {
      modifiers.insert(
          modifiers.begin(),
          {.kind = ModifierKind::Ctrl,
           .operand = addConstant(qasm3::const_eval::ConstEvalValue(
               static_cast<std::int64_t>(compatibilityControls), false))});
    }

    const std::size_t minimumOperands = signature.qubitCount + addedControls;
    if ((signature.variadicControls
             ? call->operands.size() < minimumOperands
             : call->operands.size() != minimumOperands)) {
      fail(call->debugInfo, "Invalid number of qubit operands for gate '" +
                                call->identifier + "'.");
    }

    std::vector<OperandSelection> selections;
    selections.reserve(call->operands.size());
    std::size_t broadcastWidth = 1;
    bool broadcasts = false;
    for (const auto& operand : call->operands) {
      auto selection = resolveQubitOperand(operand, call->debugInfo);
      if (selection.qubits.size() > 1) {
        if (broadcasts && broadcastWidth != selection.qubits.size()) {
          fail(call->debugInfo,
               "All broadcasting operands must have the same width.");
        }
        broadcasts = true;
        broadcastWidth = selection.qubits.size();
      }
      selections.push_back(std::move(selection));
    }
    if (broadcasts &&
        llvm::any_of(selections, [](const OperandSelection& item) {
          return item.qubits.size() == 1;
        })) {
      fail(call->debugInfo,
           "Gate operands must be either scalar qubits or equally sized "
           "registers, not a mixture.");
    }

    std::vector<GateApplication> applications;
    applications.reserve(broadcastWidth);
    for (std::size_t index = 0; index < broadcastWidth; ++index) {
      GateApplication application{.callee = callee,
                                  .parameters = parameters,
                                  .modifiers = modifiers,
                                  .location = locationOf(call->debugInfo)};
      application.qubits.reserve(selections.size());
      for (const auto& selection : selections) {
        application.qubits.push_back(selection.qubits[broadcasts ? index : 0]);
      }
      for (const auto [position, qubit] : llvm::enumerate(application.qubits)) {
        if (llvm::is_contained(
                llvm::ArrayRef(application.qubits).take_front(position),
                qubit)) {
          fail(call->debugInfo,
               "Gate operands must not reference the same qubit more than "
               "once.");
        }
      }
      applications.push_back(std::move(application));
    }
    return applications;
  }

  OperandSelection
  resolveQubitOperand(const std::shared_ptr<qasm3::GateOperand>& operand,
                      const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    if (operand->isHardwareQubit()) {
      return {.qubits = {{.kind = QubitReferenceKind::Hardware,
                          .index = operand->getHardwareQubit()}}};
    }
    const auto identifier = operand->getIdentifier();
    if (insideGate) {
      const auto local = gateQubits.find(identifier->identifier);
      if (local == gateQubits.end()) {
        fail(debugInfo,
             "Unknown gate-local qubit '" + identifier->identifier + "'.");
      }
      if (!identifier->indices.empty()) {
        fail(debugInfo, "Gate-local qubits cannot be indexed.");
      }
      return {.qubits = {{.kind = QubitReferenceKind::GateArgument,
                          .symbol = local->second}}};
    }

    const auto found = registerIds.find(identifier->identifier);
    if (found == registerIds.end() ||
        program.registers[found->second].kind != RegisterKind::Qubit) {
      fail(debugInfo,
           "Unknown qubit register '" + identifier->identifier + "'.");
    }
    const auto reg = found->second;
    const auto width = program.registers[reg].width;
    if (identifier->indices.empty()) {
      OperandSelection result;
      result.qubits.reserve(width);
      for (std::uint64_t index = 0; index < width; ++index) {
        result.qubits.push_back({.kind = QubitReferenceKind::Register,
                                 .symbol = reg,
                                 .index = index});
      }
      return result;
    }
    if (identifier->indices.size() != 1 ||
        identifier->indices.front()->indexExpressions.size() != 1) {
      fail(debugInfo, "Only one-dimensional scalar indices are supported.");
    }
    const auto index = evaluateUnsigned(
        identifier->indices.front()->indexExpressions.front(), debugInfo);
    if (index >= width) {
      fail(debugInfo, "Qubit index is out of bounds.");
    }
    return {.qubits = {{.kind = QubitReferenceKind::Register,
                        .symbol = reg,
                        .index = index}}};
  }

  std::vector<BitReference>
  resolveBits(const std::shared_ptr<qasm3::IndexedIdentifier>& identifier,
              const std::shared_ptr<qasm3::DebugInfo>& debugInfo) {
    const auto found = registerIds.find(identifier->identifier);
    if (found == registerIds.end() ||
        program.registers[found->second].kind == RegisterKind::Qubit) {
      fail(debugInfo,
           "Unknown classical register '" + identifier->identifier + "'.");
    }
    const auto reg = found->second;
    const auto width = program.registers[reg].width;
    if (identifier->indices.empty()) {
      std::vector<BitReference> result;
      result.reserve(width);
      for (std::uint64_t index = 0; index < width; ++index) {
        result.push_back({.reg = reg, .index = index});
      }
      return result;
    }
    if (identifier->indices.size() != 1 ||
        identifier->indices.front()->indexExpressions.size() != 1) {
      fail(debugInfo, "Only one-dimensional scalar indices are supported.");
    }
    const auto index = evaluateUnsigned(
        identifier->indices.front()->indexExpressions.front(), debugInfo);
    if (index >= width) {
      fail(debugInfo, "Classical bit index is out of bounds.");
    }
    return {{.reg = reg, .index = index}};
  }

  void
  ensureMeasured(const BitReference bit,
                 const std::shared_ptr<qasm3::DebugInfo>& debugInfo) const {
    if (!measuredBits[bit.reg][bit.index]) {
      fail(debugInfo,
           "Classical condition reads a bit that has not been measured.");
    }
  }

  void finalizeOutputs() {
    program.outputs = explicitOutputs.empty() ? bitRegisters : explicitOutputs;
    for (const auto reg : program.outputs) {
      if (llvm::any_of(measuredBits[reg],
                       [](const bool value) { return !value; })) {
        fail(nullptr, "Output register '" + program.registers[reg].name +
                          "' is not fully measured.");
      }
    }
  }
};

} // namespace

ParseResult parseOpenQASM(llvm::SourceMgr& sourceMgr) {
  ParseResult result;
  try {
    const auto* buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
    const auto contents = buffer->getBuffer();
    std::istringstream input(
        std::string(std::string_view(contents.data(), contents.size())));
    qasm3::Parser parser(input, true, buffer->getBufferIdentifier().str());
    auto implementation = std::make_unique<ParsedProgram::Impl>();
    implementation->statements = parser.parseProgram();
    implementation->includedFiles = parser.getIncludedFiles();
    implementation->implicitStatementCount = parser.getImplicitStatementCount();
    result.program = std::unique_ptr<ParsedProgram>(
        new ParsedProgram(std::move(implementation)));
  } catch (const qasm3::CompilerError& error) {
    result.diagnostics.push_back(diagnosticOf(error));
  } catch (const std::exception& error) {
    result.diagnostics.push_back({.message = error.what()});
  }
  return result;
}

ParseResult parseOpenQASM(const llvm::StringRef source) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(source),
                               llvm::SMLoc());
  return parseOpenQASM(sourceMgr);
}

AnalysisResult analyzeOpenQASM(const ParsedProgram& parsedProgram,
                               const FrontendOptions& options) {
  AnalysisResult result;
  try {
    result.program =
        SemanticAnalyzer(parsedProgram.impl->statements,
                         parsedProgram.impl->includedFiles,
                         parsedProgram.impl->implicitStatementCount, options)
            .run();
  } catch (const FrontendError& error) {
    result.diagnostics.push_back(error.diagnostic);
  } catch (const qasm3::CompilerError& error) {
    result.diagnostics.push_back(diagnosticOf(error));
  } catch (const std::exception& error) {
    result.diagnostics.push_back({.message = error.what()});
  }
  return result;
}

AnalysisResult analyzeOpenQASM(llvm::SourceMgr& sourceMgr,
                               const FrontendOptions& options) {
  auto parsed = parseOpenQASM(sourceMgr);
  if (!parsed) {
    return {.diagnostics = std::move(parsed.diagnostics)};
  }
  return analyzeOpenQASM(*parsed.program, options);
}

AnalysisResult analyzeOpenQASM(const llvm::StringRef source,
                               const FrontendOptions& options) {
  auto parsed = parseOpenQASM(source);
  if (!parsed) {
    return {.diagnostics = std::move(parsed.diagnostics)};
  }
  return analyzeOpenQASM(*parsed.program, options);
}

} // namespace mlir::oq3::frontend
