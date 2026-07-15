// Generated from qasm3Lexer.g4 by ANTLR 4.13.2

#pragma once

#include "antlr4-runtime.h"

class qasm3Lexer : public antlr4::Lexer {
public:
  enum {
    OPENQASM = 1,
    INCLUDE = 2,
    DEFCALGRAMMAR = 3,
    DEF = 4,
    CAL = 5,
    DEFCAL = 6,
    GATE = 7,
    EXTERN = 8,
    BOX = 9,
    LET = 10,
    BREAK = 11,
    CONTINUE = 12,
    IF = 13,
    ELSE = 14,
    END = 15,
    RETURN = 16,
    FOR = 17,
    WHILE = 18,
    IN = 19,
    SWITCH = 20,
    CASE = 21,
    DEFAULT = 22,
    PRAGMA = 23,
    AnnotationKeyword = 24,
    INPUT = 25,
    OUTPUT = 26,
    CONST = 27,
    READONLY = 28,
    MUTABLE = 29,
    QREG = 30,
    QUBIT = 31,
    CREG = 32,
    BOOL = 33,
    BIT = 34,
    INT = 35,
    UINT = 36,
    FLOAT = 37,
    ANGLE = 38,
    COMPLEX = 39,
    ARRAY = 40,
    VOID = 41,
    DURATION = 42,
    STRETCH = 43,
    GPHASE = 44,
    INV = 45,
    POW = 46,
    CTRL = 47,
    NEGCTRL = 48,
    DIM = 49,
    DURATIONOF = 50,
    DELAY = 51,
    RESET = 52,
    MEASURE = 53,
    BARRIER = 54,
    BooleanLiteral = 55,
    LBRACKET = 56,
    RBRACKET = 57,
    LBRACE = 58,
    RBRACE = 59,
    LPAREN = 60,
    RPAREN = 61,
    COLON = 62,
    SEMICOLON = 63,
    DOT = 64,
    COMMA = 65,
    EQUALS = 66,
    ARROW = 67,
    PLUS = 68,
    DOUBLE_PLUS = 69,
    MINUS = 70,
    ASTERISK = 71,
    DOUBLE_ASTERISK = 72,
    SLASH = 73,
    PERCENT = 74,
    PIPE = 75,
    DOUBLE_PIPE = 76,
    AMPERSAND = 77,
    DOUBLE_AMPERSAND = 78,
    CARET = 79,
    AT = 80,
    TILDE = 81,
    EXCLAMATION_POINT = 82,
    EqualityOperator = 83,
    CompoundAssignmentOperator = 84,
    ComparisonOperator = 85,
    BitshiftOperator = 86,
    IMAG = 87,
    ImaginaryLiteral = 88,
    BinaryIntegerLiteral = 89,
    OctalIntegerLiteral = 90,
    DecimalIntegerLiteral = 91,
    HexIntegerLiteral = 92,
    Identifier = 93,
    HardwareQubit = 94,
    FloatLiteral = 95,
    TimingLiteral = 96,
    BitstringLiteral = 97,
    Whitespace = 98,
    Newline = 99,
    LineComment = 100,
    BlockComment = 101,
    VERSION_IDENTIFIER_WHITESPACE = 102,
    VersionSpecifier = 103,
    ARBITRARY_STRING_WHITESPACE = 104,
    StringLiteral = 105,
    EAT_INITIAL_SPACE = 106,
    EAT_LINE_END = 107,
    RemainingLineContent = 108,
    CAL_PRELUDE_WHITESPACE = 109,
    CAL_PRELUDE_COMMENT = 110,
    DEFCAL_PRELUDE_WHITESPACE = 111,
    DEFCAL_PRELUDE_COMMENT = 112,
    CalibrationBlock = 113
  };

  enum {
    VERSION_IDENTIFIER = 1,
    ARBITRARY_STRING = 2,
    EAT_TO_LINE_END = 3,
    CAL_PRELUDE = 4,
    DEFCAL_PRELUDE = 5,
    CAL_BLOCK = 6
  };

  explicit qasm3Lexer(antlr4::CharStream* input);

  ~qasm3Lexer() override;

  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily
  // initialized during the first call to the constructor. You can call this
  // function if you wish to initialize the static state ahead of time.
  static void initialize();

private:
  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.
};
