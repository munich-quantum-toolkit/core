// Generated from qasm3Parser.g4 by ANTLR 4.13.2

#pragma once

#include "antlr4-runtime.h"
#include "qasm3Parser.h"

/**
 * This class defines an abstract visitor for a parse tree
 * produced by qasm3Parser.
 */
class qasm3ParserVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:
  /**
   * Visit parse trees produced by qasm3Parser.
   */
  virtual std::any visitProgram(qasm3Parser::ProgramContext* context) = 0;

  virtual std::any visitVersion(qasm3Parser::VersionContext* context) = 0;

  virtual std::any visitStatement(qasm3Parser::StatementContext* context) = 0;

  virtual std::any visitAnnotation(qasm3Parser::AnnotationContext* context) = 0;

  virtual std::any visitScope(qasm3Parser::ScopeContext* context) = 0;

  virtual std::any visitPragma(qasm3Parser::PragmaContext* context) = 0;

  virtual std::any
  visitStatementOrScope(qasm3Parser::StatementOrScopeContext* context) = 0;

  virtual std::any visitCalibrationGrammarStatement(
      qasm3Parser::CalibrationGrammarStatementContext* context) = 0;

  virtual std::any
  visitIncludeStatement(qasm3Parser::IncludeStatementContext* context) = 0;

  virtual std::any
  visitBreakStatement(qasm3Parser::BreakStatementContext* context) = 0;

  virtual std::any
  visitContinueStatement(qasm3Parser::ContinueStatementContext* context) = 0;

  virtual std::any
  visitEndStatement(qasm3Parser::EndStatementContext* context) = 0;

  virtual std::any
  visitForStatement(qasm3Parser::ForStatementContext* context) = 0;

  virtual std::any
  visitIfStatement(qasm3Parser::IfStatementContext* context) = 0;

  virtual std::any
  visitReturnStatement(qasm3Parser::ReturnStatementContext* context) = 0;

  virtual std::any
  visitWhileStatement(qasm3Parser::WhileStatementContext* context) = 0;

  virtual std::any
  visitSwitchStatement(qasm3Parser::SwitchStatementContext* context) = 0;

  virtual std::any
  visitSwitchCaseItem(qasm3Parser::SwitchCaseItemContext* context) = 0;

  virtual std::any
  visitBarrierStatement(qasm3Parser::BarrierStatementContext* context) = 0;

  virtual std::any
  visitBoxStatement(qasm3Parser::BoxStatementContext* context) = 0;

  virtual std::any
  visitDelayStatement(qasm3Parser::DelayStatementContext* context) = 0;

  virtual std::any
  visitGateCallStatement(qasm3Parser::GateCallStatementContext* context) = 0;

  virtual std::any visitMeasureArrowAssignmentStatement(
      qasm3Parser::MeasureArrowAssignmentStatementContext* context) = 0;

  virtual std::any
  visitResetStatement(qasm3Parser::ResetStatementContext* context) = 0;

  virtual std::any visitAliasDeclarationStatement(
      qasm3Parser::AliasDeclarationStatementContext* context) = 0;

  virtual std::any visitClassicalDeclarationStatement(
      qasm3Parser::ClassicalDeclarationStatementContext* context) = 0;

  virtual std::any visitConstDeclarationStatement(
      qasm3Parser::ConstDeclarationStatementContext* context) = 0;

  virtual std::any visitIoDeclarationStatement(
      qasm3Parser::IoDeclarationStatementContext* context) = 0;

  virtual std::any visitOldStyleDeclarationStatement(
      qasm3Parser::OldStyleDeclarationStatementContext* context) = 0;

  virtual std::any visitQuantumDeclarationStatement(
      qasm3Parser::QuantumDeclarationStatementContext* context) = 0;

  virtual std::any
  visitDefStatement(qasm3Parser::DefStatementContext* context) = 0;

  virtual std::any
  visitExternStatement(qasm3Parser::ExternStatementContext* context) = 0;

  virtual std::any
  visitGateStatement(qasm3Parser::GateStatementContext* context) = 0;

  virtual std::any visitAssignmentStatement(
      qasm3Parser::AssignmentStatementContext* context) = 0;

  virtual std::any visitExpressionStatement(
      qasm3Parser::ExpressionStatementContext* context) = 0;

  virtual std::any
  visitCalStatement(qasm3Parser::CalStatementContext* context) = 0;

  virtual std::any
  visitDefcalStatement(qasm3Parser::DefcalStatementContext* context) = 0;

  virtual std::any visitBitwiseXorExpression(
      qasm3Parser::BitwiseXorExpressionContext* context) = 0;

  virtual std::any
  visitAdditiveExpression(qasm3Parser::AdditiveExpressionContext* context) = 0;

  virtual std::any visitDurationofExpression(
      qasm3Parser::DurationofExpressionContext* context) = 0;

  virtual std::any visitParenthesisExpression(
      qasm3Parser::ParenthesisExpressionContext* context) = 0;

  virtual std::any visitComparisonExpression(
      qasm3Parser::ComparisonExpressionContext* context) = 0;

  virtual std::any visitMultiplicativeExpression(
      qasm3Parser::MultiplicativeExpressionContext* context) = 0;

  virtual std::any visitLogicalOrExpression(
      qasm3Parser::LogicalOrExpressionContext* context) = 0;

  virtual std::any
  visitCastExpression(qasm3Parser::CastExpressionContext* context) = 0;

  virtual std::any
  visitPowerExpression(qasm3Parser::PowerExpressionContext* context) = 0;

  virtual std::any visitBitwiseOrExpression(
      qasm3Parser::BitwiseOrExpressionContext* context) = 0;

  virtual std::any
  visitCallExpression(qasm3Parser::CallExpressionContext* context) = 0;

  virtual std::any
  visitBitshiftExpression(qasm3Parser::BitshiftExpressionContext* context) = 0;

  virtual std::any visitBitwiseAndExpression(
      qasm3Parser::BitwiseAndExpressionContext* context) = 0;

  virtual std::any
  visitEqualityExpression(qasm3Parser::EqualityExpressionContext* context) = 0;

  virtual std::any visitLogicalAndExpression(
      qasm3Parser::LogicalAndExpressionContext* context) = 0;

  virtual std::any
  visitIndexExpression(qasm3Parser::IndexExpressionContext* context) = 0;

  virtual std::any
  visitUnaryExpression(qasm3Parser::UnaryExpressionContext* context) = 0;

  virtual std::any
  visitLiteralExpression(qasm3Parser::LiteralExpressionContext* context) = 0;

  virtual std::any
  visitAliasExpression(qasm3Parser::AliasExpressionContext* context) = 0;

  virtual std::any visitDeclarationExpression(
      qasm3Parser::DeclarationExpressionContext* context) = 0;

  virtual std::any
  visitMeasureExpression(qasm3Parser::MeasureExpressionContext* context) = 0;

  virtual std::any
  visitRangeExpression(qasm3Parser::RangeExpressionContext* context) = 0;

  virtual std::any
  visitSetExpression(qasm3Parser::SetExpressionContext* context) = 0;

  virtual std::any
  visitArrayLiteral(qasm3Parser::ArrayLiteralContext* context) = 0;

  virtual std::any
  visitIndexOperator(qasm3Parser::IndexOperatorContext* context) = 0;

  virtual std::any
  visitIndexedIdentifier(qasm3Parser::IndexedIdentifierContext* context) = 0;

  virtual std::any
  visitReturnSignature(qasm3Parser::ReturnSignatureContext* context) = 0;

  virtual std::any
  visitGateModifier(qasm3Parser::GateModifierContext* context) = 0;

  virtual std::any visitScalarType(qasm3Parser::ScalarTypeContext* context) = 0;

  virtual std::any visitQubitType(qasm3Parser::QubitTypeContext* context) = 0;

  virtual std::any visitArrayType(qasm3Parser::ArrayTypeContext* context) = 0;

  virtual std::any
  visitArrayReferenceType(qasm3Parser::ArrayReferenceTypeContext* context) = 0;

  virtual std::any visitDesignator(qasm3Parser::DesignatorContext* context) = 0;

  virtual std::any
  visitDefcalTarget(qasm3Parser::DefcalTargetContext* context) = 0;

  virtual std::any visitDefcalArgumentDefinition(
      qasm3Parser::DefcalArgumentDefinitionContext* context) = 0;

  virtual std::any
  visitDefcalOperand(qasm3Parser::DefcalOperandContext* context) = 0;

  virtual std::any
  visitGateOperand(qasm3Parser::GateOperandContext* context) = 0;

  virtual std::any
  visitExternArgument(qasm3Parser::ExternArgumentContext* context) = 0;

  virtual std::any
  visitArgumentDefinition(qasm3Parser::ArgumentDefinitionContext* context) = 0;

  virtual std::any visitArgumentDefinitionList(
      qasm3Parser::ArgumentDefinitionListContext* context) = 0;

  virtual std::any visitDefcalArgumentDefinitionList(
      qasm3Parser::DefcalArgumentDefinitionListContext* context) = 0;

  virtual std::any
  visitDefcalOperandList(qasm3Parser::DefcalOperandListContext* context) = 0;

  virtual std::any
  visitExpressionList(qasm3Parser::ExpressionListContext* context) = 0;

  virtual std::any
  visitIdentifierList(qasm3Parser::IdentifierListContext* context) = 0;

  virtual std::any
  visitGateOperandList(qasm3Parser::GateOperandListContext* context) = 0;

  virtual std::any
  visitExternArgumentList(qasm3Parser::ExternArgumentListContext* context) = 0;
};
