/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qasm3/Token.hpp"

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qasm3 {

std::string Token::kindToString(const Kind kind) {
  // Print a token kind string representation.
  // This is the representation used in the error messages.
  switch (kind) {
  case Kind::None:
    return "None";
  case Kind::OpenQasm:
    return "OPENQASM";
  case Kind::Include:
    return "include";
  case Kind::DefCalGrammar:
    return "DefCalGrammar";
  case Kind::Def:
    return "Def";
  case Kind::Cal:
    return "Cal";
  case Kind::DefCal:
    return "DefCal";
  case Kind::Gate:
    return "gate";
  case Kind::Opaque:
    return "opaque";
  case Kind::Extern:
    return "extern";
  case Kind::Box:
    return "box";
  case Kind::Let:
    return "let";
  case Kind::Break:
    return "break";
  case Kind::Continue:
    return "continue";
  case Kind::If:
    return "if";
  case Kind::Else:
    return "else";
  case Kind::End:
    return "end";
  case Kind::Return:
    return "return";
  case Kind::For:
    return "for";
  case Kind::While:
    return "while";
  case Kind::In:
    return "in";
  case Kind::Pragma:
    return "pragma";
  case Kind::Input:
    return "input";
  case Kind::Output:
    return "output";
  case Kind::Const:
    return "const";
  case Kind::ReadOnly:
    return "readOnly";
  case Kind::Mutable:
    return "mutable";
  case Kind::Qreg:
    return "qreg";
  case Kind::Qubit:
    return "qubit";
  case Kind::CReg:
    return "cReg";
  case Kind::Bool:
    return "bool";
  case Kind::Bit:
    return "bit";
  case Kind::Int:
    return "int";
  case Kind::Uint:
    return "uint";
  case Kind::Float:
    return "float";
  case Kind::Angle:
    return "angle";
  case Kind::Complex:
    return "complex";
  case Kind::Array:
    return "array";
  case Kind::Void:
    return "void";
  case Kind::Duration:
    return "duration";
  case Kind::Stretch:
    return "stretch";
  case Kind::Gphase:
    return "gphase";
  case Kind::Inv:
    return "inv";
  case Kind::Pow:
    return "pow";
  case Kind::Ctrl:
    return "ctrl";
  case Kind::NegCtrl:
    return "negCtrl";
  case Kind::Dim:
    return "#dim";
  case Kind::DurationOf:
    return "durationof";
  case Kind::Delay:
    return "delay";
  case Kind::Reset:
    return "reset";
  case Kind::Measure:
    return "measure";
  case Kind::Barrier:
    return "barrier";
  case Kind::True:
    return "true";
  case Kind::False:
    return "false";
  case Kind::LBracket:
    return "[";
  case Kind::RBracket:
    return "]";
  case Kind::LBrace:
    return "{";
  case Kind::RBrace:
    return "}";
  case Kind::LParen:
    return "(";
  case Kind::RParen:
    return ")";
  case Kind::Colon:
    return ":";
  case Kind::Semicolon:
    return ";";
  case Kind::Eof:
    return "Eof";
  case Kind::Dot:
    return ".";
  case Kind::Comma:
    return ",";
  case Kind::Equals:
    return "=";
  case Kind::Arrow:
    return "->";
  case Kind::Plus:
    return "+";
  case Kind::DoublePlus:
    return "++";
  case Kind::Minus:
    return "-";
  case Kind::Asterisk:
    return "*";
  case Kind::DoubleAsterisk:
    return "**";
  case Kind::Slash:
    return "/";
  case Kind::Percent:
    return "%";
  case Kind::Pipe:
    return "|";
  case Kind::DoublePipe:
    return "||";
  case Kind::Ampersand:
    return "&";
  case Kind::DoubleAmpersand:
    return "&&";
  case Kind::Caret:
    return "^";
  case Kind::At:
    return "@";
  case Kind::Tilde:
    return "~";
  case Kind::ExclamationPoint:
    return "!";
  case Kind::DoubleEquals:
    return "==";
  case Kind::NotEquals:
    return "!=";
  case Kind::PlusEquals:
    return "+=";
  case Kind::MinusEquals:
    return "-=";
  case Kind::AsteriskEquals:
    return "*=";
  case Kind::SlashEquals:
    return "/=";
  case Kind::AmpersandEquals:
    return "&=";
  case Kind::PipeEquals:
    return "|=";
  case Kind::TildeEquals:
    return "~=";
  case Kind::CaretEquals:
    return "^=";
  case Kind::LeftShitEquals:
    return "<<=";
  case Kind::RightShiftEquals:
    return ">>=";
  case Kind::PercentEquals:
    return "%=";
  case Kind::DoubleAsteriskEquals:
    return "**=";
  case Kind::LessThan:
    return "<";
  case Kind::LessThanEquals:
    return "<=";
  case Kind::GreaterThan:
    return ">";
  case Kind::GreaterThanEquals:
    return ">=";
  case Kind::LeftShift:
    return "<<";
  case Kind::RightShift:
    return ">>";
  case Kind::Imag:
    return "imag";
  case Kind::Underscore:
    return "underscore";
  case Kind::DoubleQuote:
    return "\"";
  case Kind::SingleQuote:
    return "'";
  case Kind::BackSlash:
    return "\\";
  case Kind::Identifier:
    return "Identifier";
  case Kind::HardwareQubit:
    return "HardwareQubit";
  case Kind::StringLiteral:
    return "StringLiteral";
  case Kind::IntegerLiteral:
    return "IntegerLiteral";
  case Kind::FloatLiteral:
    return "FloatLiteral";
  case Kind::TimingLiteral:
    return "TimingLiteral";
  case Kind::Sin:
    return "sin";
  case Kind::Cos:
    return "cos";
  case Kind::Tan:
    return "tan";
  case Kind::Exp:
    return "exp";
  case Kind::Ln:
    return "ln";
  case Kind::Sqrt:
    return "sqrt";
  case Kind::InitialLayout:
    return "InitialLayout";
  case Kind::OutputPermutation:
    return "OutputPermutation";

  default:
    // This cannot happen, as we have a case for every enum value.
    // The default case is only here to silence compiler warnings.
    throw std::runtime_error("Unknown token kind");
  }
}

std::string Token::toString() const {
  std::stringstream ss;
  ss << kindToString(kind);
  switch (kind) {
  case Kind::Identifier:
    ss << " (" << str << ")";
    break;
  case Kind::StringLiteral:
    ss << " (\"" << str << "\")";
    break;
  case Kind::InitialLayout:
  case Kind::OutputPermutation:
    ss << " (" << str << ")";
    break;
  case Kind::IntegerLiteral:
    ss << " (" << val << ")";
    break;
  case Kind::FloatLiteral:
    ss << " (" << valReal << ")";
    break;
  case Kind::TimingLiteral:
    ss << " (" << valReal << " [s])";
    break;
  default:
    break;
  }
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Token::Kind& k) {
  os << Token::kindToString(k);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Token& t) {
  os << t.toString();
  return os;
}
} // namespace qasm3
