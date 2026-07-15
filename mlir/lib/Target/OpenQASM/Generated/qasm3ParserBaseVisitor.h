// Generated from qasm3Parser.g4 by ANTLR 4.13.2

#pragma once

#include "antlr4-runtime.h"
#include "qasm3ParserVisitor.h"

/**
 * This class provides an empty implementation of qasm3ParserVisitor, which can
 * be extended to create a visitor which only needs to handle a subset of the
 * available methods.
 */
class qasm3ParserBaseVisitor : public qasm3ParserVisitor {
public:
  virtual std::any visitProgram(qasm3Parser::ProgramContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVersion(qasm3Parser::VersionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStatement(qasm3Parser::StatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitAnnotation(qasm3Parser::AnnotationContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScope(qasm3Parser::ScopeContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPragma(qasm3Parser::PragmaContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitStatementOrScope(qasm3Parser::StatementOrScopeContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCalibrationGrammarStatement(
      qasm3Parser::CalibrationGrammarStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIncludeStatement(qasm3Parser::IncludeStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBreakStatement(qasm3Parser::BreakStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitContinueStatement(qasm3Parser::ContinueStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitEndStatement(qasm3Parser::EndStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitForStatement(qasm3Parser::ForStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIfStatement(qasm3Parser::IfStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitReturnStatement(qasm3Parser::ReturnStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitWhileStatement(qasm3Parser::WhileStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSwitchStatement(qasm3Parser::SwitchStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSwitchCaseItem(qasm3Parser::SwitchCaseItemContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBarrierStatement(qasm3Parser::BarrierStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitBoxStatement(qasm3Parser::BoxStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDelayStatement(qasm3Parser::DelayStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGateCallStatement(qasm3Parser::GateCallStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMeasureArrowAssignmentStatement(
      qasm3Parser::MeasureArrowAssignmentStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitResetStatement(qasm3Parser::ResetStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAliasDeclarationStatement(
      qasm3Parser::AliasDeclarationStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitClassicalDeclarationStatement(
      qasm3Parser::ClassicalDeclarationStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConstDeclarationStatement(
      qasm3Parser::ConstDeclarationStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIoDeclarationStatement(
      qasm3Parser::IoDeclarationStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOldStyleDeclarationStatement(
      qasm3Parser::OldStyleDeclarationStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQuantumDeclarationStatement(
      qasm3Parser::QuantumDeclarationStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDefStatement(qasm3Parser::DefStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitExternStatement(qasm3Parser::ExternStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGateStatement(qasm3Parser::GateStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAssignmentStatement(
      qasm3Parser::AssignmentStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExpressionStatement(
      qasm3Parser::ExpressionStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitCalStatement(qasm3Parser::CalStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDefcalStatement(qasm3Parser::DefcalStatementContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitwiseXorExpression(
      qasm3Parser::BitwiseXorExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAdditiveExpression(
      qasm3Parser::AdditiveExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDurationofExpression(
      qasm3Parser::DurationofExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenthesisExpression(
      qasm3Parser::ParenthesisExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonExpression(
      qasm3Parser::ComparisonExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMultiplicativeExpression(
      qasm3Parser::MultiplicativeExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalOrExpression(
      qasm3Parser::LogicalOrExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitCastExpression(qasm3Parser::CastExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitPowerExpression(qasm3Parser::PowerExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitwiseOrExpression(
      qasm3Parser::BitwiseOrExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitCallExpression(qasm3Parser::CallExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitshiftExpression(
      qasm3Parser::BitshiftExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBitwiseAndExpression(
      qasm3Parser::BitwiseAndExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEqualityExpression(
      qasm3Parser::EqualityExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalAndExpression(
      qasm3Parser::LogicalAndExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIndexExpression(qasm3Parser::IndexExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitUnaryExpression(qasm3Parser::UnaryExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitLiteralExpression(qasm3Parser::LiteralExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitAliasExpression(qasm3Parser::AliasExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDeclarationExpression(
      qasm3Parser::DeclarationExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitMeasureExpression(qasm3Parser::MeasureExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitRangeExpression(qasm3Parser::RangeExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitSetExpression(qasm3Parser::SetExpressionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitArrayLiteral(qasm3Parser::ArrayLiteralContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIndexOperator(qasm3Parser::IndexOperatorContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIndexedIdentifier(qasm3Parser::IndexedIdentifierContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitReturnSignature(qasm3Parser::ReturnSignatureContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGateModifier(qasm3Parser::GateModifierContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitScalarType(qasm3Parser::ScalarTypeContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitQubitType(qasm3Parser::QubitTypeContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArrayType(qasm3Parser::ArrayTypeContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArrayReferenceType(
      qasm3Parser::ArrayReferenceTypeContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDesignator(qasm3Parser::DesignatorContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDefcalTarget(qasm3Parser::DefcalTargetContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefcalArgumentDefinition(
      qasm3Parser::DefcalArgumentDefinitionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDefcalOperand(qasm3Parser::DefcalOperandContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGateOperand(qasm3Parser::GateOperandContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitExternArgument(qasm3Parser::ExternArgumentContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArgumentDefinition(
      qasm3Parser::ArgumentDefinitionContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArgumentDefinitionList(
      qasm3Parser::ArgumentDefinitionListContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefcalArgumentDefinitionList(
      qasm3Parser::DefcalArgumentDefinitionListContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitDefcalOperandList(qasm3Parser::DefcalOperandListContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitExpressionList(qasm3Parser::ExpressionListContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitIdentifierList(qasm3Parser::IdentifierListContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any
  visitGateOperandList(qasm3Parser::GateOperandListContext* ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExternArgumentList(
      qasm3Parser::ExternArgumentListContext* ctx) override {
    return visitChildren(ctx);
  }
};
