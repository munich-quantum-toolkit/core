/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qasm3/Importer.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qasm3/Exception.hpp"
#include "qasm3/Gate.hpp"
#include "qasm3/Parser.hpp"
#include "qasm3/Statement.hpp"
#include "qasm3/StdGates.hpp"
#include "qasm3/Types.hpp"
#include "qasm3/passes/ConstEvalPass.hpp"
#include "qasm3/passes/TypeCheckPass.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace qasm3 {

auto Importer::importf(const std::string& filename) -> qc::QuantumComputation {
  std::ifstream file(filename);
  if (!file.good()) {
    throw std::runtime_error("Could not open file " + filename);
  }
  return import(file);
}

auto Importer::import(std::istream& is) -> qc::QuantumComputation {
  // parse the program into an AST
  Parser parser(is);
  const auto program = parser.parseProgram();
  // translate the AST into a quantum computation
  qc::QuantumComputation qc;
  Importer importer(qc);
  importer.visitProgram(program);
  // initialize the initial layout and output permutation
  qc.initializeIOMapping();
  return qc;
}

auto Importer::imports(const std::string& qasm) -> qc::QuantumComputation {
  std::istringstream is(qasm);
  return import(is);
}

std::map<std::string,
         std::pair<const_eval::ConstEvalValue, type_checking::InferredType>>
Importer::initializeBuiltins() {
  std::map<std::string,
           std::pair<const_eval::ConstEvalValue, type_checking::InferredType>>
      builtins{};

  type_checking::InferredType const floatTy{
      std::dynamic_pointer_cast<ResolvedType>(
          std::make_shared<DesignatedType<uint64_t>>(Float, 64))};

  builtins.emplace("pi",
                   std::pair{const_eval::ConstEvalValue(qc::PI), floatTy});
  builtins.emplace("π", std::pair{const_eval::ConstEvalValue(qc::PI), floatTy});
  builtins.emplace("tau",
                   std::pair{const_eval::ConstEvalValue(qc::TAU), floatTy});
  builtins.emplace("τ",
                   std::pair{const_eval::ConstEvalValue(qc::TAU), floatTy});
  builtins.emplace("euler",
                   std::pair{const_eval::ConstEvalValue(qc::E), floatTy});
  builtins.emplace("ℇ", std::pair{const_eval::ConstEvalValue(qc::E), floatTy});

  return builtins;
}

void Importer::translateGateOperand(
    const std::shared_ptr<GateOperand>& gateOperand,
    std::vector<qc::Qubit>& qubits, const qc::QuantumRegisterMap& qregs,
    const std::shared_ptr<DebugInfo>& debugInfo) const {
  if (gateOperand->isHardwareQubit()) {
    const auto hardwareQubit = gateOperand->getHardwareQubit();
    // Ensure that the circuit has enough qubits.
    // Currently, we emulate hardware qubits via a single quantum register q.
    for (size_t i = qc->getNqubits(); i <= hardwareQubit; ++i) {
      const auto q = static_cast<qc::Qubit>(i);
      qc->addQubit(q, q, q);
    }

    qubits.emplace_back(
        static_cast<qc::Qubit>(gateOperand->getHardwareQubit()));
    return;
  }
  const auto indexedIdentifier = gateOperand->getIdentifier();
  const auto qregIterator = qregs.find(indexedIdentifier->identifier);
  if (qregIterator == qregs.end()) {
    throw CompilerError("Usage of unknown quantum register.", debugInfo);
  }
  const auto& qreg = qregIterator->second;

  // full register
  if (indexedIdentifier->indices.empty()) {
    for (size_t i = 0; i < qreg.getSize(); ++i) {
      qubits.emplace_back(static_cast<qc::Qubit>(qreg.getStartIndex() + i));
    }
    return;
  }

  if (indexedIdentifier->indices.size() > 1) {
    throw CompilerError("Only single index expressions are supported.",
                        debugInfo);
  }
  const auto indexOperator = indexedIdentifier->indices[0];
  if (indexOperator->indexExpressions.size() > 1) {
    throw CompilerError("Only single index expressions are supported.",
                        debugInfo);
  }
  const auto indexExpression = indexOperator->indexExpressions[0];
  const auto result = evaluatePositiveConstant(indexExpression, debugInfo);

  if (result >= qreg.getSize()) {
    throw CompilerError(
        "Index expression must be smaller than the width of the "
        "quantum register.",
        debugInfo);
  }
  qubits.emplace_back(qreg.getStartIndex() + static_cast<qc::Qubit>(result));
}

void Importer::translateBitOperand(
    const std::shared_ptr<IndexedIdentifier>& indexedIdentifier,
    std::vector<qc::Bit>& bits,
    const std::shared_ptr<DebugInfo>& debugInfo) const {
  const auto iter =
      qc->getClassicalRegisters().find(indexedIdentifier->identifier);
  if (iter == qc->getClassicalRegisters().end()) {
    throw CompilerError("Usage of unknown classical register.", debugInfo);
  }
  const auto& creg = iter->second;
  const auto& indices = indexedIdentifier->indices;
  // full register
  if (indices.empty()) {
    for (size_t i = 0; i < creg.getSize(); ++i) {
      bits.emplace_back(creg.getStartIndex() + i);
    }
    return;
  }

  if (indices.size() > 1) {
    throw CompilerError("Only single index expressions are supported.",
                        debugInfo);
  }
  const auto& indexExpressions = indices[0]->indexExpressions;
  if (indexExpressions.size() > 1) {
    throw CompilerError("Only single index expressions are supported.",
                        debugInfo);
  }
  const auto& indexExpression = indexExpressions[0];
  const auto index = evaluatePositiveConstant(indexExpression, debugInfo);
  if (index >= creg.getSize()) {
    throw CompilerError(
        "Index expression must be smaller than the width of the "
        "classical register.",
        debugInfo);
  }
  bits.emplace_back(creg.getStartIndex() + index);
}

std::variant<std::pair<qc::Bit, bool>,
             std::tuple<qc::ClassicalRegister, qc::ComparisonKind, uint64_t>>
Importer::translateCondition(
    const std::shared_ptr<Expression>& condition,
    const std::shared_ptr<DebugInfo>& debugInfo) const {
  if (const auto binaryExpression =
          std::dynamic_pointer_cast<BinaryExpression>(condition);
      binaryExpression != nullptr) {
    const auto comparisonKind = getComparisonKind(binaryExpression->op);
    if (!comparisonKind) {
      throw CompilerError("Unsupported comparison operator.", debugInfo);
    }
    auto lhsIsIdentifier = true;
    std::shared_ptr<Expression> lhs =
        std::dynamic_pointer_cast<IndexedIdentifier>(binaryExpression->lhs);
    if (lhs == nullptr) {
      lhsIsIdentifier = false;
      lhs = std::dynamic_pointer_cast<Constant>(binaryExpression->lhs);
    }
    std::shared_ptr<Expression> rhs{};
    if (lhsIsIdentifier) {
      rhs = std::dynamic_pointer_cast<Constant>(binaryExpression->rhs);
    } else {
      rhs = std::dynamic_pointer_cast<IndexedIdentifier>(binaryExpression->rhs);
    }
    if (lhs == nullptr || rhs == nullptr) {
      throw CompilerError("Only classical registers and constants are "
                          "supported in conditions.",
                          debugInfo);
    }

    const auto& indexedIdentifier =
        lhsIsIdentifier ? std::dynamic_pointer_cast<IndexedIdentifier>(lhs)
                        : std::dynamic_pointer_cast<IndexedIdentifier>(rhs);
    const auto& identifier = indexedIdentifier->identifier;
    const auto val = lhsIsIdentifier
                         ? std::dynamic_pointer_cast<Constant>(rhs)->getUInt()
                         : std::dynamic_pointer_cast<Constant>(lhs)->getUInt();

    const auto creg = qc->getClassicalRegisters().find(identifier);
    if (creg == qc->getClassicalRegisters().end()) {
      throw CompilerError("Usage of unknown or invalid identifier '" +
                              identifier + "' in condition.",
                          debugInfo);
    }
    return std::tuple{creg->second, *comparisonKind, val};
  }
  if (const auto unaryExpression =
          std::dynamic_pointer_cast<UnaryExpression>(condition);
      unaryExpression != nullptr) {
    // This should already be caught by the type checker.
    assert(unaryExpression->op == UnaryExpression::LogicalNot ||
           unaryExpression->op == UnaryExpression::BitwiseNot);
    const auto& indexedIdentifier =
        std::dynamic_pointer_cast<IndexedIdentifier>(unaryExpression->operand);
    std::vector<qc::Bit> bits{};
    translateBitOperand(indexedIdentifier, bits, debugInfo);
    // This should already be caught by the type checker.
    assert(bits.size() == 1);
    return std::pair{bits[0], false};
  }
  // must be a single bit at this point
  const auto& indexedIdentifier =
      std::dynamic_pointer_cast<IndexedIdentifier>(condition);
  // should also be caught by the type checker
  assert(indexedIdentifier != nullptr);
  std::vector<qc::Bit> bits{};
  translateBitOperand(indexedIdentifier, bits, debugInfo);
  // This should already be caught by the type checker.
  assert(bits.size() == 1);
  return std::pair{bits[0], true};
}

uint64_t
Importer::evaluatePositiveConstant(const std::shared_ptr<Expression>& expr,
                                   const std::shared_ptr<DebugInfo>& debugInfo,
                                   const uint64_t defaultValue) {
  if (expr == nullptr) {
    return defaultValue;
  }

  const auto constInt = std::dynamic_pointer_cast<Constant>(expr);
  if (!constInt) {
    throw CompilerError("Expected a constant integer expression.", debugInfo);
  }

  return constInt->getUInt();
}

Importer::Importer(qc::QuantumComputation& quantumComputation)
    : typeCheckPass(constEvalPass), qc(&quantumComputation),
      gates(STANDARD_GATES) {
  for (const auto& [identifier, builtin] : initializeBuiltins()) {
    constEvalPass.addConst(identifier, builtin.first);
    typeCheckPass.addBuiltin(identifier, builtin.second);
  }
}

void Importer::visitProgram(
    const std::vector<std::shared_ptr<Statement>>& program) {
  // TODO: in the future, don't exit early, but collect all errors
  // To do this, we need to insert make sure that erroneous declarations
  // actually insert a dummy entry; also, we need to synchronize to the next
  // semicolon, to make sure we don't do some weird stuff and report false
  // errors.
  for (const auto& statement : program) {
    constEvalPass.processStatement(*statement);
    typeCheckPass.processStatement(*statement);
    statement->accept(this);
  }

  // Finally, if we have a initial layout and output permutation specified,
  // apply them.
  if (!initialLayout.empty()) {
    qc->initialLayout = initialLayout;
  }
  if (!outputPermutation.empty()) {
    qc->outputPermutation = outputPermutation;
  }
}

void Importer::visitVersionDeclaration(
    const std::shared_ptr<VersionDeclaration> versionDeclaration) {
  if (versionDeclaration->version < 3) {
    openQASM2CompatMode = true;
  }
}

void Importer::visitDeclarationStatement(
    const std::shared_ptr<DeclarationStatement> declarationStatement) {
  const auto identifier = declarationStatement->identifier;
  if (declarations.find(identifier).has_value()) {
    // TODO: show the location of the previous declaration
    throw CompilerError("Identifier '" + identifier + "' already declared.",
                        declarationStatement->debugInfo);
  }

  std::shared_ptr<ResolvedType> const ty =
      std::get<1>(declarationStatement->type);

  if (const auto sizedTy =
          std::dynamic_pointer_cast<DesignatedType<uint64_t>>(ty)) {
    const auto designator = sizedTy->getDesignator();
    switch (sizedTy->type) {
    case Qubit:
      qc->addQubitRegister(designator, identifier);
      break;
    case Bit:
    case Int:
    case Uint:
      qc->addClassicalRegister(designator, identifier);
      break;
    case Float:
      // not adding to qc
      break;
    case Angle:
      throw CompilerError("Angle type is currently not supported.",
                          declarationStatement->debugInfo);
    }
  } else {
    throw CompilerError("Only sized types are supported.",
                        declarationStatement->debugInfo);
  }
  declarations.emplace(identifier, declarationStatement);

  if (declarationStatement->expression == nullptr) {
    // value is uninitialized
    return;
  }
  if (const auto measureExpression =
          std::dynamic_pointer_cast<MeasureExpression>(
              declarationStatement->expression->expression)) {
    assert(!declarationStatement->isConst &&
           "Type check pass should catch this");
    visitMeasureAssignment(std::make_shared<IndexedIdentifier>(identifier),
                           measureExpression, declarationStatement->debugInfo);
    return;
  }
  if (declarationStatement->isConst) {
    // nothing to do
    return;
  }

  throw CompilerError(
      "Only measure statements are supported for initialization.",
      declarationStatement->debugInfo);
}

void Importer::visitAssignmentStatement(
    const std::shared_ptr<AssignmentStatement> assignmentStatement) {
  const auto identifier = assignmentStatement->identifier->identifier;
  const auto declaration = declarations.find(identifier);
  assert(declaration.has_value() && "Checked by type check pass");
  assert(!declaration->get()->isConst && "Checked by type check pass");

  if (const auto measureExpression =
          std::dynamic_pointer_cast<MeasureExpression>(
              assignmentStatement->expression->expression)) {
    visitMeasureAssignment(assignmentStatement->identifier, measureExpression,
                           assignmentStatement->debugInfo);
    return;
  }

  // In the future, handle classical computation.
  throw CompilerError("Classical computation not supported.",
                      assignmentStatement->debugInfo);
}

void Importer::visitInitialLayout(const std::shared_ptr<InitialLayout> layout) {
  if (!initialLayout.empty()) {
    throw CompilerError("Multiple initial layout specifications found.",
                        layout->debugInfo);
  }
  initialLayout = layout->permutation;
}

void Importer::visitOutputPermutation(
    const std::shared_ptr<OutputPermutation> permutation) {
  if (!outputPermutation.empty()) {
    throw CompilerError("Multiple output permutation specifications found.",
                        permutation->debugInfo);
  }
  outputPermutation = permutation->permutation;
}

void Importer::visitGateStatement(
    const std::shared_ptr<GateDeclaration> gateStatement) {
  auto identifier = gateStatement->identifier;
  if (gateStatement->isOpaque) {
    if (!gates.contains(identifier)) {
      // only builtin gates may be declared as opaque.
      throw CompilerError("Unsupported opaque gate '" + identifier + "'.",
                          gateStatement->debugInfo);
    }

    return;
  }

  if (openQASM2CompatMode) {
    // we need to check if this is a standard gate
    identifier = parseGateIdentifierCompatMode(identifier).first;
  }

  if (auto prevDeclaration = gates.find(identifier);
      prevDeclaration != gates.end()) {
    if (std::dynamic_pointer_cast<StandardGate>(prevDeclaration->second)) {
      // we ignore redeclarations of standard gates
      return;
    }
    // TODO: print location of previous declaration
    throw CompilerError("Gate '" + identifier + "' already declared.",
                        gateStatement->debugInfo);
  }

  const auto parameters = gateStatement->parameters;
  const auto qubits = gateStatement->qubits;

  // first we check that all parameters and qubits are unique
  std::vector<std::string> parameterIdentifiers{};
  for (const auto& parameter : parameters->identifiers) {
    if (std::find(parameterIdentifiers.begin(), parameterIdentifiers.end(),
                  parameter->identifier) != parameterIdentifiers.end()) {
      throw CompilerError("Parameter '" + parameter->identifier +
                              "' already declared.",
                          gateStatement->debugInfo);
    }
    parameterIdentifiers.emplace_back(parameter->identifier);
  }
  std::vector<std::string> qubitIdentifiers{};
  for (const auto& qubit : qubits->identifiers) {
    if (std::find(qubitIdentifiers.begin(), qubitIdentifiers.end(),
                  qubit->identifier) != qubitIdentifiers.end()) {
      throw CompilerError("Qubit '" + qubit->identifier + "' already declared.",
                          gateStatement->debugInfo);
    }
    qubitIdentifiers.emplace_back(qubit->identifier);
  }

  auto compoundGate = std::make_shared<CompoundGate>(CompoundGate(
      parameterIdentifiers, qubitIdentifiers, gateStatement->statements));

  gates.emplace(identifier, compoundGate);
}

void Importer::visitGateCallStatement(
    const std::shared_ptr<GateCallStatement> gateCallStatement) {
  const auto& qregs = qc->getQuantumRegisters();
  if (auto op = evaluateGateCall(
          gateCallStatement, gateCallStatement->identifier,
          gateCallStatement->arguments, gateCallStatement->operands, qregs);
      op != nullptr) {
    qc->emplace_back(std::move(op));
  }
}

std::unique_ptr<qc::Operation> Importer::evaluateGateCall(
    const std::shared_ptr<GateCallStatement>& gateCallStatement,
    const std::string& identifier,
    const std::vector<std::shared_ptr<Expression>>& parameters,
    std::vector<std::shared_ptr<GateOperand>> targets,
    const qc::QuantumRegisterMap& qregs) {
  auto iter = gates.find(identifier);
  std::shared_ptr<Gate> gate;
  size_t implicitControls{0};

  if (iter == gates.end()) {
    if (identifier == "mcx" || identifier == "mcx_gray" ||
        identifier == "mcx_vchain" || identifier == "mcx_recursive" ||
        identifier == "mcphase") {
      // we create a temp gate definition for these gates
      gate = getMcGateDefinition(identifier, gateCallStatement->operands.size(),
                                 gateCallStatement->debugInfo);
    } else if (openQASM2CompatMode) {
      auto [updatedIdentifier, nControls] =
          parseGateIdentifierCompatMode(identifier);

      iter = gates.find(updatedIdentifier);
      if (iter == gates.end()) {
        throw CompilerError("Usage of unknown gate '" + identifier + "'.",
                            gateCallStatement->debugInfo);
      }
      gate = iter->second;
      implicitControls = nControls;
    } else {
      throw CompilerError("Usage of unknown gate '" + identifier + "'.",
                          gateCallStatement->debugInfo);
    }
  } else {
    gate = iter->second;
  }

  if (gate->getNParameters() != parameters.size()) {
    throw CompilerError(
        "Gate '" + identifier + "' takes " +
            std::to_string(gate->getNParameters()) + " parameters, but " +
            std::to_string(parameters.size()) + " were supplied.",
        gateCallStatement->debugInfo);
  }

  // here we count the number of controls
  std::vector<std::pair<std::shared_ptr<GateOperand>, bool>> controls{};
  // since standard gates may define a number of control targets, we first
  // need to handle those
  size_t nControls{gate->getNControls() + implicitControls};
  if (targets.size() < nControls) {
    throw CompilerError("Gate '" + identifier + "' takes " +
                            std::to_string(nControls) + " controls, but only " +
                            std::to_string(targets.size()) +
                            " qubits were supplied.",
                        gateCallStatement->debugInfo);
  }

  controls.reserve(nControls);
  for (size_t i = 0; i < nControls; ++i) {
    controls.emplace_back(targets[i], true);
  }

  bool invertOperation = false;
  for (const auto& modifier : gateCallStatement->modifiers) {
    if (auto ctrlModifier =
            std::dynamic_pointer_cast<CtrlGateModifier>(modifier);
        ctrlModifier != nullptr) {
      size_t const n = evaluatePositiveConstant(ctrlModifier->expression,
                                                gateCallStatement->debugInfo,
                                                /*defaultValue=*/
                                                1);
      if (targets.size() < n + nControls) {
        throw CompilerError(
            "Gate '" + identifier + "' takes " + std::to_string(n + nControls) +
                " controls, but only " + std::to_string(targets.size()) +
                " were supplied.",
            gateCallStatement->debugInfo);
      }

      for (size_t i = 0; i < n; ++i) {
        controls.emplace_back(targets[nControls + i], ctrlModifier->ctrlType);
      }
      nControls += n;
    } else if (auto invModifier =
                   std::dynamic_pointer_cast<InvGateModifier>(modifier);
               invModifier != nullptr) {
      // if we have an even number of inv modifiers, they cancel each other
      // out
      invertOperation = !invertOperation;
    } else {
      throw CompilerError("Only ctrl/negctrl/inv modifiers are supported.",
                          gateCallStatement->debugInfo);
    }
  }
  targets.erase(targets.begin(),
                targets.begin() + static_cast<int64_t>(nControls));

  if (gate->getNTargets() != targets.size()) {
    throw CompilerError("Gate '" + identifier + "' takes " +
                            std::to_string(gate->getNTargets()) +
                            " targets, but " + std::to_string(targets.size()) +
                            " were supplied.",
                        gateCallStatement->debugInfo);
  }

  // now evaluate all arguments; we only support const arguments.
  std::vector<qc::fp> evaluatedParameters{};
  for (const auto& param : parameters) {
    auto result = constEvalPass.visit(param);
    if (!result.has_value()) {
      throw CompilerError(
          "Only const expressions are supported as gate parameters, but "
          "found '" +
              param->getName() + "'.",
          gateCallStatement->debugInfo);
    }

    evaluatedParameters.emplace_back(result->toExpr()->asFP());
  }

  size_t broadcastingWidth{1};
  qc::Targets targetBits{};
  std::vector<size_t> targetBroadcastingIndices{};
  size_t i{0};
  for (const auto& target : targets) {
    qc::Targets t{};
    translateGateOperand(target, t, qregs, gateCallStatement->debugInfo);

    targetBits.emplace_back(t[0]);

    if (t.size() > 1) {
      if (broadcastingWidth != 1 && t.size() != broadcastingWidth) {
        throw CompilerError(
            "When broadcasting, all registers must be of the same width.",
            gateCallStatement->debugInfo);
      }
      broadcastingWidth = t.size();

      targetBroadcastingIndices.emplace_back(i);
    }

    i++;
  }

  std::vector<qc::Control> controlBits{};
  std::vector<size_t> controlBroadcastingIndices{};
  i = 0;
  for (const auto& [control, type] : controls) {
    qc::Targets c{};
    translateGateOperand(control, c, qregs, gateCallStatement->debugInfo);

    controlBits.emplace_back(c[0], type ? qc::Control::Type::Pos
                                        : qc::Control::Type::Neg);

    if (c.size() > 1) {
      if (broadcastingWidth != 1 && c.size() != broadcastingWidth) {
        throw CompilerError(
            "When broadcasting, all registers must be of the same width.",
            gateCallStatement->debugInfo);
      }
      broadcastingWidth = c.size();

      controlBroadcastingIndices.emplace_back(i);
    }

    i++;
  }

  auto op = std::make_unique<qc::CompoundOperation>();
  for (size_t j = 0; j < broadcastingWidth; ++j) {
    // check if any of the bits are duplicate
    std::unordered_set<qc::Qubit> allQubits;
    for (const auto& control : controlBits) {
      if (allQubits.contains(control.qubit)) {
        throw CompilerError("Duplicate qubit in control list.",
                            gateCallStatement->debugInfo);
      }
      allQubits.emplace(control.qubit);
    }
    for (const auto& qubit : targetBits) {
      if (allQubits.contains(qubit)) {
        throw CompilerError("Duplicate qubit in target list.",
                            gateCallStatement->debugInfo);
      }
      allQubits.emplace(qubit);
    }

    // first we apply the operation
    auto nestedOp = applyQuantumOperation(gate, targetBits, controlBits,
                                          evaluatedParameters, invertOperation,
                                          gateCallStatement->debugInfo);
    if (nestedOp == nullptr || broadcastingWidth == 1) {
      return nestedOp;
    }
    op->getOps().emplace_back(std::move(nestedOp));

    // after applying the operation, we update the broadcast bits
    if (j == broadcastingWidth - 1) {
      break;
    }
    for (auto index : targetBroadcastingIndices) {
      targetBits[index] = qc::Qubit{targetBits[index] + 1};
    }
    for (auto index : controlBroadcastingIndices) {
      controlBits[index].qubit = qc::Qubit{controlBits[index].qubit + 1};
    }
  }
  return op;
}

std::shared_ptr<Gate>
Importer::getMcGateDefinition(const std::string& identifier, size_t operandSize,
                              const std::shared_ptr<DebugInfo>& debugInfo) {
  std::vector<std::string> targetParams{};
  std::vector<std::shared_ptr<GateOperand>> operands;
  size_t nTargets = operandSize;
  if (identifier == "mcx_vchain") {
    nTargets -= (nTargets + 1) / 2 - 2;
  } else if (identifier == "mcx_recursive" && nTargets > 5) {
    nTargets -= 1;
  }
  for (size_t i = 0; i < operandSize; ++i) {
    targetParams.emplace_back("q" + std::to_string(i));
    if (i < nTargets) {
      operands.emplace_back(std::make_shared<GateOperand>(
          std::make_shared<IndexedIdentifier>("q" + std::to_string(i))));
    }
  }
  const size_t nControls = nTargets - 1;

  std::string nestedGateIdentifier = "x";
  std::vector<std::shared_ptr<Expression>> nestedParameters{};
  std::vector<std::string> nestedParameterNames{};
  if (identifier == "mcphase") {
    nestedGateIdentifier = "p";
    nestedParameters.emplace_back(std::make_shared<IdentifierExpression>("x"));
    nestedParameterNames.emplace_back("x");
  }

  // ctrl(nTargets - 1) @ x q0, ..., q(nTargets - 1)
  const auto gateCall = GateCallStatement(
      debugInfo, nestedGateIdentifier,
      std::vector<std::shared_ptr<GateModifier>>{
          std::make_shared<CtrlGateModifier>(
              true, std::make_shared<Constant>(nControls, false))},
      nestedParameters, operands);
  const auto inner = std::make_shared<GateCallStatement>(gateCall);

  const CompoundGate g{nestedParameterNames, targetParams, {inner}};
  return std::make_shared<CompoundGate>(g);
}

std::unique_ptr<qc::Operation> Importer::applyQuantumOperation(
    const std::shared_ptr<Gate>& gate, const qc::Targets& targetBits,
    const std::vector<qc::Control>& controlBits,
    const std::vector<qc::fp>& evaluatedParameters, const bool invertOperation,
    const std::shared_ptr<DebugInfo>& debugInfo) {
  if (auto* standardGate = dynamic_cast<StandardGate*>(gate.get())) {
    auto op = std::make_unique<qc::StandardOperation>(
        qc::Controls{controlBits.begin(), controlBits.end()}, targetBits,
        standardGate->info.type, evaluatedParameters);
    if (invertOperation) {
      op->invert();
    }
    return op;
  }
  if (auto* compoundGate = dynamic_cast<CompoundGate*>(gate.get())) {
    constEvalPass.pushEnv();

    for (size_t i = 0; i < compoundGate->parameterNames.size(); ++i) {
      constEvalPass.addConst(compoundGate->parameterNames[i],
                             evaluatedParameters[i]);
    }

    auto nestedQubits = qc::QuantumRegisterMap{};
    size_t index = 0;
    for (const auto& qubitIdentifier : compoundGate->targetNames) {
      nestedQubits.try_emplace(qubitIdentifier, targetBits[index], 1,
                               qubitIdentifier);
      index++;
    }

    auto op = std::make_unique<qc::CompoundOperation>(true);
    for (const auto& nestedGate : compoundGate->body) {
      if (auto barrierStatement =
              std::dynamic_pointer_cast<BarrierStatement>(nestedGate);
          barrierStatement != nullptr) {
        std::vector<qc::Qubit> qubits{};
        for (const auto& g : barrierStatement->gates) {
          translateGateOperand(g, qubits, nestedQubits,
                               barrierStatement->debugInfo);
        }
        op->emplace_back<qc::StandardOperation>(qubits, qc::Barrier);
      } else if (auto resetStatement =
                     std::dynamic_pointer_cast<ResetStatement>(nestedGate);
                 resetStatement != nullptr) {
        std::vector<qc::Qubit> qubits{};
        translateGateOperand(resetStatement->gate, qubits, nestedQubits,
                             resetStatement->debugInfo);
        op->emplace_back<qc::NonUnitaryOperation>(qubits, qc::Reset);
      } else if (auto gateCallStatement =
                     std::dynamic_pointer_cast<GateCallStatement>(nestedGate);
                 gateCallStatement != nullptr) {
        for (const auto& operand : gateCallStatement->operands) {
          if (operand->isHardwareQubit()) {
            continue;
          }
          const auto& identifier = operand->getIdentifier();
          // OpenQASM 3.0 doesn't support indexing of gate arguments.
          if (!identifier->indices.empty() &&
              std::find(compoundGate->targetNames.begin(),
                        compoundGate->targetNames.end(),
                        identifier->identifier) !=
                  compoundGate->targetNames.end()) {
            throw CompilerError(
                "Gate arguments cannot be indexed within gate body.",
                debugInfo);
          }
        }

        auto nestedOp =
            evaluateGateCall(gateCallStatement, gateCallStatement->identifier,
                             gateCallStatement->arguments,
                             gateCallStatement->operands, nestedQubits);
        if (nestedOp == nullptr) {
          return nullptr;
        }
        op->getOps().emplace_back(std::move(nestedOp));
      } else {
        throw CompilerError("Unhandled quantum statement.", debugInfo);
      }
    }
    op->setControls(qc::Controls{controlBits.begin(), controlBits.end()});
    if (invertOperation) {
      op->invert();
    }

    constEvalPass.popEnv();

    if (op->getOps().size() == 1) {
      return std::move(op->getOps()[0]);
    }

    return op;
  }

  throw CompilerError("Unknown gate type.", debugInfo);
}

void Importer::visitMeasureAssignment(
    const std::shared_ptr<IndexedIdentifier>& indexedIdentifier,
    const std::shared_ptr<MeasureExpression>& measureExpression,
    const std::shared_ptr<DebugInfo>& debugInfo) {
  const auto& identifier = indexedIdentifier->identifier;
  const auto decl = declarations.find(identifier);
  if (!decl.has_value()) {
    throw CompilerError("Usage of unknown identifier '" + identifier + "'.",
                        debugInfo);
  }

  if (!std::get<1>(decl.value()->type)->isBit()) {
    throw CompilerError(
        "Measure expression can only be assigned to a bit register.",
        debugInfo);
  }

  std::vector<qc::Qubit> qubits{};
  std::vector<qc::Bit> bits{};
  translateGateOperand(measureExpression->gate, qubits,
                       qc->getQuantumRegisters(), debugInfo);
  translateBitOperand(indexedIdentifier, bits, debugInfo);

  if (qubits.size() != bits.size()) {
    throw CompilerError(
        "Classical and quantum register must have the same width in "
        "measure statement. Classical register '" +
            identifier + "' has " + std::to_string(bits.size()) +
            " bits, but quantum register '" +
            measureExpression->gate->getName() + "' has " +
            std::to_string(qubits.size()) + " qubits.",
        debugInfo);
  }

  qc->measure(qubits, bits);
}

void Importer::visitBarrierStatement(
    const std::shared_ptr<BarrierStatement> barrierStatement) {
  std::vector<qc::Qubit> qubits{};
  for (const auto& gate : barrierStatement->gates) {
    translateGateOperand(gate, qubits, qc->getQuantumRegisters(),
                         barrierStatement->debugInfo);
  }
  qc->barrier(qubits);
}

void Importer::visitResetStatement(
    const std::shared_ptr<ResetStatement> resetStatement) {
  std::vector<qc::Qubit> qubits{};
  translateGateOperand(resetStatement->gate, qubits, qc->getQuantumRegisters(),
                       resetStatement->debugInfo);
  qc->reset(qubits);
}

void Importer::visitIfStatement(
    const std::shared_ptr<IfStatement> ifStatement) {
  const auto& condition =
      translateCondition(ifStatement->condition, ifStatement->debugInfo);

  // translate statements in then/else blocks
  if (!ifStatement->thenStatements.empty()) {
    auto thenOps = translateBlockOperations(ifStatement->thenStatements);
    if (std::holds_alternative<std::pair<qc::Bit, bool>>(condition)) {
      const auto& [bit, val] = std::get<std::pair<qc::Bit, bool>>(condition);
      qc->emplace_back<qc::ClassicControlledOperation>(std::move(thenOps), bit,
                                                       val ? 1 : 0);
    } else {
      const auto& [creg, comparisonKind, rhs] = std::get<
          std::tuple<qc::ClassicalRegister, qc::ComparisonKind, uint64_t>>(
          condition);
      qc->emplace_back<qc::ClassicControlledOperation>(std::move(thenOps), creg,
                                                       rhs, comparisonKind);
    }
  }

  if (!ifStatement->elseStatements.empty()) {
    auto elseOps = translateBlockOperations(ifStatement->elseStatements);
    if (std::holds_alternative<std::pair<qc::Bit, bool>>(condition)) {
      const auto& [bit, val] = std::get<std::pair<qc::Bit, bool>>(condition);
      qc->emplace_back<qc::ClassicControlledOperation>(std::move(elseOps), bit,
                                                       val ? 0 : 1);
    } else {
      const auto& [creg, comparisonKind, rhs] = std::get<
          std::tuple<qc::ClassicalRegister, qc::ComparisonKind, uint64_t>>(
          condition);
      const auto invertedComparisonKind =
          qc::getInvertedComparisonKind(comparisonKind);
      qc->emplace_back<qc::ClassicControlledOperation>(
          std::move(elseOps), creg, rhs, invertedComparisonKind);
    }
  }
}

std::unique_ptr<qc::Operation> Importer::translateBlockOperations(
    const std::vector<std::shared_ptr<Statement>>& statements) {
  auto blockOps = std::make_unique<qc::CompoundOperation>();
  for (const auto& statement : statements) {
    auto gateCall = std::dynamic_pointer_cast<GateCallStatement>(statement);
    if (gateCall == nullptr) {
      throw CompilerError("Only quantum statements are supported in blocks.",
                          statement->debugInfo);
    }
    const auto& qregs = qc->getQuantumRegisters();

    auto op = evaluateGateCall(gateCall, gateCall->identifier,
                               gateCall->arguments, gateCall->operands, qregs);

    blockOps->emplace_back(std::move(op));
  }

  return blockOps;
}

std::pair<std::string, size_t>
Importer::parseGateIdentifierCompatMode(const std::string& identifier) {
  // we need to copy as we modify the string and need to return the original
  // string if we don't find a match.
  std::string gateIdentifier = identifier;
  size_t implicitControls = 0;
  while (!gateIdentifier.empty() && gateIdentifier[0] == 'c') {
    gateIdentifier = gateIdentifier.substr(1);
    implicitControls++;
  }

  if (!gates.contains(gateIdentifier)) {
    return std::pair{identifier, 0};
  }
  return std::pair{gateIdentifier, implicitControls};
}
} // namespace qasm3
