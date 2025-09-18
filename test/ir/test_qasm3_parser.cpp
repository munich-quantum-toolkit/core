/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "qasm3/Exception.hpp"
#include "qasm3/Importer.hpp"
#include "qasm3/Parser.hpp"
#include "qasm3/Scanner.hpp"
#include "qasm3/Statement.hpp"
#include "qasm3/Token.hpp"
#include "qasm3/passes/ConstEvalPass.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace qc;

class Qasm3ParserTest : public testing::TestWithParam<std::size_t> {};

TEST_F(Qasm3ParserTest, ImportQasm3) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "/* this is a comment, which can span multiple\n"
                               "\n"
                               "\n"
                               "// lines */\n"
                               "bit[3] c;\n";
  const auto qc = qasm3::Importer::imports(testfile);
  EXPECT_EQ(qc.getNqubits(), 3);
  EXPECT_EQ(qc.getNcbits(), 3);
}

TEST_F(Qasm3ParserTest, ImportQasm3OldSyntax) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qreg q[3];\n"
                               "creg r[3];\n";
  const auto qc = qasm3::Importer::imports(testfile);
  EXPECT_EQ(qc.getNqubits(), 3);
  EXPECT_EQ(qc.getNcbits(), 3);
}

TEST_F(Qasm3ParserTest, ImportQasm3GateDecl) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_x q1, q2 {\n"
                               "  x q1;\n"
                               "  x q2;\n"
                               "}\n"
                               "my_x q[0], q[1];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "x q[0];\n"
                               "x q[1];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3CtrlModifier) {
  const std::string testfile =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "qubit[5] q;\n"
      "x q[0];\n"
      "ctrl @ x q[0], q[1];\n"
      "ctrl(2) @ x q[0], q[1], q[2];\n"
      "ctrl(3) @ x q[0], q[1], q[2], q[3];\n"
      "ctrl(3) @ negctrl @ x q[0], q[1], q[2], q[3], q[4];\n"
      "ctrl @ p(0.5) q[0], q[1];\n"
      "ctrl @ rx(pi) q[0], q[1];\n"
      "ctrl @ y q[0], q[1];\n"
      "ctrl @ ry(pi) q[0], q[1];\n"
      "ctrl @ z q[0], q[1];\n"
      "ctrl @ rz(pi) q[0], q[1];\n"
      "ctrl @ h q[0], q[1];\n"
      "ctrl @ swap q[0], q[1], q[2];\n"
      "ctrl @ rxx(pi) q[0], q[1], q[2];\n"
      "ctrl @ negctrl @ x q[0], q[1], q[2];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected =
      "// i 0 1 2 3 4\n"
      "// o 0 1 2 3 4\n"
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "qubit[5] q;\n"
      "x q[0];\n"
      "cx q[0], q[1];\n"
      "ccx q[0], q[1], q[2];\n"
      "ctrl(3) @ x q[0], q[1], q[2], q[3];\n"
      "ctrl(3) @ negctrl @ x q[0], q[1], q[2], q[3], q[4];\n"
      "cp(0.5) q[0], q[1];\n"
      "crx(3.14159265358979) q[0], q[1];\n"
      "cy q[0], q[1];\n"
      "cry(3.14159265358979) q[0], q[1];\n"
      "cz q[0], q[1];\n"
      "crz(3.14159265358979) q[0], q[1];\n"
      "ch q[0], q[1];\n"
      "cswap q[0], q[1], q[2];\n"
      "ctrl @ rxx(3.14159265358979) q[0], q[1], q[2];\n"
      "ctrl @ negctrl @ x q[0], q[1], q[2];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3InvModifier) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "inv @ s q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "sdg q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3CompoundGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "gate my_compound_gate q {\n"
                               " // comment\n"
                               "  x /* nested comment */ q;\n"
                               "  h q;\n"
                               "}\n"
                               "my_compound_gate q;";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "x q[0];\n"
                               "h q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ControlledCompoundGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_compound_gate q {\n"
                               "  x q;\n"
                               "}\n"
                               "ctrl @ my_compound_gate q[0], q[1];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "cx q[0], q[1];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ParamCompoundGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_compound_gate(a) q {\n"
                               "  rz(a) q;\n"
                               "}\n"
                               "my_compound_gate(1.0 * pi) q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "rz(3.14159265358979) q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3Measure) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit r1;\n"
                               "bit[2] r2;\n"
                               "h q;\n"
                               "r1[0] = measure q[0];\n"
                               "r1 = measure q[0];\n"
                               "r2 = measure q;\n"
                               "measure q[1] -> r1;\n"
                               "measure q[1] -> r2[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] r1;\n"
                               "bit[2] r2;\n"
                               "h q[0];\n"
                               "h q[1];\n"
                               "r1[0] = measure q[0];\n"
                               "r1[0] = measure q[0];\n"
                               "r2 = measure q;\n"
                               "r1[0] = measure q[1];\n"
                               "r2[0] = measure q[1];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3InitialLayout) {
  const std::string testfile = "// i 1 0\n"
                               "// o 1 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 1 0\n"
                               "// o 1 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n";

  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ConstEval) {
  const std::string testfile =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "const uint N = (0x4 + 8 - 0b10 - (0o10 / 4)) / 2;\n"
      "qubit[N * 2] q;\n"
      "ctrl @ x q[0], q[N * 2 - 1];\n"
      "x q;";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2 3 4 5 6 7\n"
                               "// o 0 1 2 3 4 5 6 7\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[8] q;\n"
                               "bit[32] N;\n"
                               "cx q[0], q[7];\n"
                               "x q[0];\n"
                               "x q[1];\n"
                               "x q[2];\n"
                               "x q[3];\n"
                               "x q[4];\n"
                               "x q[5];\n"
                               "x q[6];\n"
                               "x q[7];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3NonUnitary) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "reset q1[0];\n"
                               "barrier q1, q2;\n"
                               "reset q1;\n"
                               "bit c = measure q1[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2 3\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "bit[1] c;\n"
                               "reset q1[0];\n"
                               "barrier q1[0], q1[1], q2[0], q2[1];\n"
                               "reset q1;\n"
                               "c[0] = measure q1[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3IfStatement) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (c) {\n"
                               "  x q[1];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c[0] = measure q[0];\n"
                               "if (c[0]) {\n"
                               "  x q[1];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3SingleBitIfStatement) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (c[0]) {\n"
                               "  x q[0];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q;\n"
                               "if (c[0]) {\n"
                               "  x q[0];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3InvertedSingleBitIfStatement) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (!c[0]) {\n"
                               "  x q[0];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q;\n"
                               "if (!c[0]) {\n"
                               "  x q[0];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3SingleBitIfStatementRegister) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q[0];\n"
                               "if (c == 1) {\n"
                               "  x q[0];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q;\n"
                               "if (c == 1) {\n"
                               "  x q[0];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3SingleBitIfStatementRegisterFlipped) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q[0];\n"
                               "if (1 == c) {\n"
                               "  x q[0];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q;\n"
                               "if (c == 1) {\n"
                               "  x q[0];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3IfElseStatement) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (c) {\n"
                               "  x q[1];\n"
                               "} else {\n"
                               "  x q[0];\n"
                               "  x q[1];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c[0] = measure q[0];\n"
                               "if (c[0]) {\n"
                               "  x q[1];\n"
                               "} else {\n"
                               "  x q[0];\n"
                               "  x q[1];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3IfElseStatementRegister) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c = measure q[0];\n"
                               "if (c == 1) {\n"
                               "  x q[1];\n"
                               "} else {\n"
                               "  x q[0];\n"
                               "  x q[1];\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c[0] = measure q[0];\n"
                               "if (c == 1) {\n"
                               "  x q[1];\n"
                               "} else {\n"
                               "  x q[0];\n"
                               "  x q[1];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3EmptyIfElse) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (c) {\n"
                               "} else {\n"
                               "}";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c[0] = measure q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3UnsupportedSingleBitIfStatement) {
  const auto comparisonKinds = {ComparisonKind::Lt, ComparisonKind::Leq,
                                ComparisonKind::Gt, ComparisonKind::Geq};

  for (const auto comparisonKind : comparisonKinds) {
    const std::string testfile = "OPENQASM 3.0;\n"
                                 "include \"stdgates.inc\";\n"
                                 "qubit[1] q;\n"
                                 "h q[0];\n"
                                 "bit c = measure q[0];\n"
                                 "if (c " +
                                 toString(comparisonKind) +
                                 " true) {\n"
                                 "  x q[0];\n"
                                 "}";
    EXPECT_THROW(
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Type Check Error: Cannot compare boolean types.");
          throw;
        },
        qasm3::CompilerError);
  }
}

TEST_F(Qasm3ParserTest, ImportQasm3OutputPerm) {
  const std::string testfile = "// i 0 2 1 3\n"
                               "// o 3 0\n"
                               "qubit[4] q;\n";
  const auto qc = qasm3::Importer::imports(testfile);

  std::stringstream out{};
  QuantumComputation::printPermutation(qc.outputPermutation, out);

  const std::string expected = "\t0: 1\n"
                               "\t3: 0\n";
  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3OutputPermDefault) {
  const std::string testfile = "// i 0 2 1 3\n"
                               "qubit[4] q;\n";
  const auto qc = qasm3::Importer::imports(testfile);

  std::stringstream out{};
  QuantumComputation::printPermutation(qc.outputPermutation, out);

  const std::string expected = "\t0: 0\n"
                               "\t1: 1\n"
                               "\t2: 2\n"
                               "\t3: 3\n";
  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3IfElseNoBlock) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (c) {} else \n"
                               "  x q[1];\n"
                               "x q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c[0] = measure q[0];\n"
                               "if (!c[0]) {\n"
                               "  x q[1];\n"
                               "}\n"
                               "x q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3InvalidStatementInBlock) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "bit c = measure q;\n"
                               "if (c) {\n"
                               "  qubit invalid;\n"
                               "}";
  EXPECT_THROW(
      try {
        const auto qc = qasm3::Importer::imports(testfile);
      } catch (const qasm3::CompilerError& e) {
        EXPECT_EQ(e.message,
                  "Only quantum statements are supported in blocks.");
        throw;
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasm3ImplicitInclude) {
  const std::string testfile = "qubit q;\n"
                               "h q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "h q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3Qelib1) {
  const std::string testfile = "OPENQASM 2.0;\n"
                               "include \"qelib1.inc\";\n"
                               "qubit q;\n"
                               "h q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "h q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3NestedGates) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_x q { x q; }\n"
                               "gate my_x2 q1 { x q1; }\n"
                               "qubit[1] q;\n"
                               "my_x2 q[0];\n";
  const auto qc = qasm3::Importer::imports(testfile);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.at(0)->getType(), OpType::X);
}

TEST_F(Qasm3ParserTest, ImportQasm3AlternatingControl) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[7] q;\n"
                               "ctrl @ negctrl(2) @ negctrl @ ctrl @ ctrl @ x "
                               "q[0], q[1], q[2], q[3], q[4], q[5], q[6];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2 3 4 5 6\n"
                               "// o 0 1 2 3 4 5 6\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[7] q;\n"
                               "ctrl @ negctrl(3) @ ctrl(2) @ x q[0], q[1], "
                               "q[2], q[3], q[4], q[5], q[6];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasmConstEval) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "const uint N_1 = 0xa;\n"
                               "const uint N_2 = 8;\n"
                               "qubit[N_1 - N_2] q;\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[32] N_1;\n"
                               "bit[32] N_2;\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasmBroadcasting) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "h q1;\n"
                               "reset q2;\n"
                               "cx q1, q2;\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2 3\n"
                               "// o 0 1 2 3\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "h q1[0];\n"
                               "h q1[1];\n"
                               "reset q2;\n"
                               "cx q1[0], q2[0];\n"
                               "cx q1[1], q2[1];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasmComparison) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "h q;\n"
                               "c[0] = measure q[0];\n"
                               "if (c < 0) { x q[0]; }\n"
                               "if (c <= 0) { x q[0]; }\n"
                               "if (c > 0) { x q[0]; }\n"
                               "if (c >= 0) { x q[0]; }\n"
                               "if (c == 0) { x q[0]; }\n"
                               "if (c != 0) { x q[0]; }\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "h q[0];\n"
                               "h q[1];\n"
                               "c[0] = measure q[0];\n"
                               "if (c < 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c <= 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c > 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c >= 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c == 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c != 0) {\n"
                               "  x q[0];\n"
                               "}\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasmNativeRedeclaration) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "bit c1;\n"
                               "gate h q { U(pi/2, 0, pi) q; }\n"
                               "h q;\n"
                               "c1 = measure q;\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c1;\n"
                               "h q[0];\n"
                               "c1 = measure q;\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm2CPrefix) {
  const std::string testfile = "OPENQASM 2.0;\n"
                               "qubit[5] q;\n"
                               "// nothing in the declaration on purpose\n"
                               "gate ccccx q1, q2, q3, q4, q5 {\n"
                               "}\n"
                               "ccccx q[0], q[1], q[2], q[3], q[4];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2 3 4\n"
                               "// o 0 1 2 3 4\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[5] q;\n"
                               "ctrl(4) @ x q[0], q[1], q[2], q[3], q[4];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportMCXGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "qubit[4] q;\n"
                               "mcx q[0], q[1], q[2], q[3];\n";
  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2 3\n"
                               "// o 0 1 2 3\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[4] q;\n"
                               "ctrl(3) @ x q[0], q[1], q[2], q[3];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportMQTBenchCircuit) {
  const std::string qasm = R"(
    // Benchmark was created by MQT Bench on 2024-03-17
    // For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
    // MQT Bench version: 1.1.0
    // Qiskit version: 1.0.2

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg eval[1];
    qreg q[1];
    creg meas[2];
    u2(0,-pi) eval[0];
    u3(0.9272952180016122,0,0) q[0];
    cx eval[0],q[0];
    u(-0.9272952180016122,0,0) q[0];
    cx eval[0],q[0];
    h eval[0];
    u(0.9272952180016122,0,0) q[0];
    barrier eval[0],q[0];
    measure eval[0] -> meas[0];
    measure q[0] -> meas[1];
  )";
  auto qc = qasm3::Importer::imports(qasm);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] eval;\n"
                               "qubit[1] q;\n"
                               "bit[2] meas;\n"
                               "h eval[0];\n"
                               "ry(0.927295218001612) q[0];\n"
                               "cx eval[0], q[0];\n"
                               "ry(-0.927295218001612) q[0];\n"
                               "cx eval[0], q[0];\n"
                               "h eval[0];\n"
                               "ry(0.927295218001612) q[0];\n"
                               "barrier eval[0], q[0];\n"
                               "meas[0] = measure eval[0];\n"
                               "meas[1] = measure q[0];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportMSGate) {
  const std::string testfile = "OPENQASM 3.0;"
                               "qubit[3] q;"
                               "bit[3] c;"
                               "gate ms(p0) q0, q1, q2 {"
                               "  rxx(p0) q0, q1;"
                               "  rxx(p0) q0, q2;"
                               "  rxx(p0) q1, q2;"
                               "}"
                               "ms(0.844396) q[0], q[1], q[2];"
                               "c = measure q;";

  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2\n"
                               "// o 0 1 2\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "bit[3] c;\n"
                               "rxx(0.844396) q[0], q[1];\n"
                               "rxx(0.844396) q[0], q[2];\n"
                               "rxx(0.844396) q[1], q[2];\n"
                               "c = measure q;\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, HardwareQubitsInGates) {
  const std::string testfile = "OPENQASM 3.0;"
                               "h $0;"
                               "cx $0, $1;";

  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "h q[0];\n"
                               "cx q[0], q[1];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, HardwareQubitInMeasurement) {
  const std::string testfile = "OPENQASM 3.0;"
                               "bit c = measure $0;";

  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "bit[1] c;\n"
                               "c = measure q;\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, HardwareQubitsNonConsecutive) {
  const std::string testfile = "OPENQASM 3.0;"
                               "h $0;"
                               "cx $0, $2;";

  const auto qc = qasm3::Importer::imports(testfile);

  const std::string out = qc.toQASM();
  const std::string expected = "// i 0 1 2\n"
                               "// o 0 1 2\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "h q[0];\n"
                               "cx q[0], q[2];\n";
  EXPECT_EQ(out, expected);
}

TEST_F(Qasm3ParserTest, ImportQasm2CPrefixInvalidGate) {
  const std::string testfile = "OPENQASM 2.0;\n"
                               "qubit[5] q;\n"
                               "cccck q[0], q[1], q[2], q[3], q[4];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Usage of unknown gate 'cccck'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasm3CPrefix) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "qubit[5] q;\n"
                               "ccccx q[0], q[1], q[2], q[3], q[4];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Usage of unknown gate 'ccccx'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmScanner) {
  std::stringstream ss{};
  const std::string testfile =
      "$1 : . .5 -1. 1.25e-3 1e3 -= += ++ *= **= ** /= % %= |= || | &= "
      "&& & ^= ^ ~= ~ ! <= <<= << < >= >>= >> >";
  const auto tokens = std::vector{
      qasm3::Token::Kind::HardwareQubit,
      qasm3::Token::Kind::Colon,
      qasm3::Token::Kind::Dot,
      qasm3::Token::Kind::FloatLiteral,
      qasm3::Token::Kind::FloatLiteral,
      qasm3::Token::Kind::FloatLiteral,
      qasm3::Token::Kind::FloatLiteral,
      qasm3::Token::Kind::MinusEquals,
      qasm3::Token::Kind::PlusEquals,
      qasm3::Token::Kind::DoublePlus,
      qasm3::Token::Kind::AsteriskEquals,
      qasm3::Token::Kind::DoubleAsteriskEquals,
      qasm3::Token::Kind::DoubleAsterisk,
      qasm3::Token::Kind::SlashEquals,
      qasm3::Token::Kind::Percent,
      qasm3::Token::Kind::PercentEquals,
      qasm3::Token::Kind::PipeEquals,
      qasm3::Token::Kind::DoublePipe,
      qasm3::Token::Kind::Pipe,
      qasm3::Token::Kind::AmpersandEquals,
      qasm3::Token::Kind::DoubleAmpersand,
      qasm3::Token::Kind::Ampersand,
      qasm3::Token::Kind::CaretEquals,
      qasm3::Token::Kind::Caret,
      qasm3::Token::Kind::TildeEquals,
      qasm3::Token::Kind::Tilde,
      qasm3::Token::Kind::ExclamationPoint,
      qasm3::Token::Kind::LessThanEquals,
      qasm3::Token::Kind::LeftShitEquals,
      qasm3::Token::Kind::LeftShift,
      qasm3::Token::Kind::LessThan,
      qasm3::Token::Kind::GreaterThanEquals,
      qasm3::Token::Kind::RightShiftEquals,
      qasm3::Token::Kind::RightShift,
      qasm3::Token::Kind::GreaterThan,
      qasm3::Token::Kind::Eof,
  };

  ss << testfile;
  qasm3::Scanner scanner(&ss);

  for (const auto& expected : tokens) {
    auto token = scanner.next();
    EXPECT_EQ(token.kind, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmParseOperators) {
  std::stringstream ss{};
  const std::string testfile = "x += 1;\n"
                               "x -= 1;\n"
                               "x *= 1;\n"
                               "x /= 1;\n"
                               "x &= 1;\n"
                               "x |= 1;\n"
                               "x ~= 1;\n"
                               "x ^= 1;\n"
                               "x <<= 1;\n"
                               "x >>= 1;\n"
                               "x %= 1;\n"
                               "x **= 1;\n";

  ss << testfile;
  qasm3::Parser parser(ss, false);

  const auto expectedTypes = std::vector{
      qasm3::AssignmentStatement::Type::PlusAssignment,
      qasm3::AssignmentStatement::Type::MinusAssignment,
      qasm3::AssignmentStatement::Type::TimesAssignment,
      qasm3::AssignmentStatement::Type::DivAssignment,
      qasm3::AssignmentStatement::Type::BitwiseAndAssignment,
      qasm3::AssignmentStatement::Type::BitwiseOrAssignment,
      qasm3::AssignmentStatement::Type::BitwiseNotAssignment,
      qasm3::AssignmentStatement::Type::BitwiseXorAssignment,
      qasm3::AssignmentStatement::Type::LeftShiftAssignment,
      qasm3::AssignmentStatement::Type::RightShiftAssignment,
      qasm3::AssignmentStatement::Type::ModuloAssignment,
      qasm3::AssignmentStatement::Type::PowerAssignment,
  };

  for (const auto& expected : expectedTypes) {
    const auto stmt = parser.parseAssignmentStatement();
    EXPECT_EQ(stmt->type, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmParseUnaryExpressions) {
  std::stringstream ss{};
  const std::string testfile = "sin(x)\n"
                               "cos(x)\n"
                               "tan(x)\n"
                               "exp(x)\n"
                               "ln(x)\n"
                               "sqrt(x)\n";

  ss << testfile;
  qasm3::Parser parser(ss, false);

  const auto expectedTypes = std::vector{
      qasm3::UnaryExpression::Op::Sin, qasm3::UnaryExpression::Op::Cos,
      qasm3::UnaryExpression::Op::Tan, qasm3::UnaryExpression::Op::Exp,
      qasm3::UnaryExpression::Op::Ln,  qasm3::UnaryExpression::Op::Sqrt,
  };

  for (const auto& expected : expectedTypes) {
    const auto expr = parser.parseExpression();
    const auto unaryExpr =
        std::dynamic_pointer_cast<qasm3::UnaryExpression>(expr);
    EXPECT_NE(unaryExpr, nullptr);
    EXPECT_EQ(unaryExpr->op, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmParseBinaryExpressions) {
  std::stringstream ss{};
  const std::string testfile = "x^5\n"
                               "x == 5\n"
                               "x != 5\n"
                               "x <= 5\n"
                               "x < 5\n"
                               "x >= 5\n"
                               "x > 5\n";

  ss << testfile;
  qasm3::Parser parser(ss, false);

  const auto expectedTypes = std::vector{
      qasm3::BinaryExpression::Op::Power,
      qasm3::BinaryExpression::Op::Equal,
      qasm3::BinaryExpression::Op::NotEqual,
      qasm3::BinaryExpression::Op::LessThanOrEqual,
      qasm3::BinaryExpression::Op::LessThan,
      qasm3::BinaryExpression::Op::GreaterThanOrEqual,
      qasm3::BinaryExpression::Op::GreaterThan,
  };

  for (const auto& expected : expectedTypes) {
    const auto expr = parser.parseExpression();
    const auto binaryExpr =
        std::dynamic_pointer_cast<qasm3::BinaryExpression>(expr);
    EXPECT_NE(binaryExpr, nullptr);
    EXPECT_EQ(binaryExpr->op, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmUnknownQreg) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "x q;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Usage of unknown quantum register.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmIndexOutOfBounds) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "x q[2];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Index expression must be smaller than the "
                               "width of the quantum register.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmIndexOutOfBoundsClassical) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "c[2] = measure q[0];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Index expression must be smaller than the "
                               "width of the classical register.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateDeclaration) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "qubit[2] q;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Identifier 'q' already declared.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInitConstRegWithMeasure) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "const bit c = measure q;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Constant Evaluation: Constant declaration "
                               "initialization expression must be const.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmAssignmentUnknownIdentifier) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "c = measure q;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Unknown identifier 'c'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmAssignmentConstVar) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "const bit c = 0;\n"
                               "c = measure q;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Type Check Error: Type mismatch in declaration statement: "
                    "Expected 'bit[1]', found 'uint[32]'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmMultipleInputPermutations) {
  const std::string testfile = "// i 0\n"
                               "// i 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Multiple initial layout specifications found.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmMultipleOutputPermutations) {
  const std::string testfile = "// o 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Multiple output permutation specifications found.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidOpaqueGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "opaque asdf q;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Unsupported opaque gate 'asdf'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateGateDecl) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_x q { x q; }\n"
                               "gate my_x q { x q; }\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Gate 'my_x' already declared.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateQubitArgGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_gate q, q { }\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Qubit 'q' already declared.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUndeclaredGate) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "my_x q;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Usage of unknown gate 'my_x'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidGateTargets) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_x q { x q; }\n"
                               "my_x q[0], q[1];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate 'my_x' takes 1 targets, but 2 were supplied.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidGateControls) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "cx q[0];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate 'cx' takes 1 targets, but 0 were supplied.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidGateModifiers) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "ctrl(2) @ x q[0];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate 'x' takes 2 controls, but only 1 were supplied.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateCallNonConst) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c = measure q;\n"
                               "rz(c) q[0];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Only const expressions are supported as gate "
                    "parameters, but found 'IndexedIdentifier (c)'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateCallBroadcastingInvalidWidth) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[3] q2;\n"
                               "cx q1, q2;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(
              e.message,
              "When broadcasting, all registers must be of the same width.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateCallIndexingGateBody) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_x q { x q[0]; }\n"
                               "qubit q;\n"
                               "my_x q;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate arguments cannot be indexed within gate body.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateMeasureInvalidSizes) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[3] c = measure q;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(
              e.message,
              "Classical and quantum register must have the same width "
              "in measure statement. Classical register 'c' has 3 bits, "
              "but quantum register 'IndexedIdentifier (q)' has 2 qubits.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateOldStyleDesignator) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q[2];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "In OpenQASM 3.0, the designator has been "
                               "changed to `type[designator] identifier;`");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateExpectStatement) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "+\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Expected quantum statement, got '+'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateVersionDeclaration) {
  const std::string testfile = "qubit q;\n"
                               "OPENQASM 3.0;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(
              e.message,
              "Version declaration must be at the beginning of the file.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidExpected) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "qubit[2] q;\n"
                               "cx q[0] q[1];"; // missing comma
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Expected ',', got 'Identifier'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmTypeMismatchAssignment) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit x;\n"
                               "x = 10;";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Type mismatch in assignment. "
                               "Expected 'bit[1]', found 'uint[32]'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmTypeMismatchBinaryExpr) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "const bit x = 0;\n"
                               "const int y = 10 + x;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Type Check Error: Type mismatch in declaration statement: "
                    "Expected 'bit[1]', found 'uint[32]'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmConstNotInitialized) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "const bit x;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Constant Evaluation: Constant declaration initialization "
                    "expression must be initialized.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnaryTypeMismatchLogicalNot) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "int x = !0;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Cannot apply logical not to "
                               "non-boolean type.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnaryTypeMismatchBitwiseNot) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bool x = ~false;\n"
                               "bool y = !true;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Cannot apply bitwise not to "
                               "non-numeric type.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmBinaryTypeMismatch) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "int x = 1 + false;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Type mismatch in binary "
                               "expression: uint[32], bool.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmAssignmentIndexType) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit[16] x;\n"
                               "x[-1] = 0;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Type mismatch in assignment. "
                               "Expected 'bit[16]', found 'uint[32]'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnknownIdentifier) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "int x = y;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Unknown identifier 'y'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnknownQubit) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit x = measure q;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type Check Error: Unknown identifier 'q'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmNegativeTypeDesignator) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit[-1] c;\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(
              e.message,
              "Type Check Error: Designator expression type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateQubitBroadcast) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "qubit[2] q;\n"
                               "cx q, q[1];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Duplicate qubit in target list.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateQubitBroadcastInControls) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "qubit[2] q;\n"
                               "ccx q, q[0], q[1];\n";
  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Duplicate qubit in control list.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmRegisterDeclarationInDefinition) {
  const std::string testfile = "qubit[1] q;"
                               "gate test a {"
                               "qubit[2] crash;"
                               "x a;}"
                               "test q[0];";

  EXPECT_THROW(
      {
        try {
          const auto qc = qasm3::Importer::imports(testfile);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Expected quantum statement, got 'qubit'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, TestPrintTokens) {
  // This test is to print all tokens and make the coverage report happy.
  const auto tokens = std::vector{
      qasm3::Token(qasm3::Token::Kind::None, 0, 0),
      qasm3::Token(qasm3::Token::Kind::OpenQasm, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Include, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DefCalGrammar, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Def, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Cal, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DefCal, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Gate, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Opaque, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Extern, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Box, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Let, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Break, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Continue, 0, 0),
      qasm3::Token(qasm3::Token::Kind::If, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Else, 0, 0),
      qasm3::Token(qasm3::Token::Kind::End, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Return, 0, 0),
      qasm3::Token(qasm3::Token::Kind::For, 0, 0),
      qasm3::Token(qasm3::Token::Kind::While, 0, 0),
      qasm3::Token(qasm3::Token::Kind::In, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Pragma, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Input, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Output, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Const, 0, 0),
      qasm3::Token(qasm3::Token::Kind::ReadOnly, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Mutable, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Qreg, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Qubit, 0, 0),
      qasm3::Token(qasm3::Token::Kind::CReg, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Bool, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Bit, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Int, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Uint, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Float, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Angle, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Complex, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Array, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Void, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Duration, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Stretch, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Gphase, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Inv, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Pow, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Ctrl, 0, 0),
      qasm3::Token(qasm3::Token::Kind::NegCtrl, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Dim, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DurationOf, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Delay, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Reset, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Measure, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Barrier, 0, 0),
      qasm3::Token(qasm3::Token::Kind::True, 0, 0),
      qasm3::Token(qasm3::Token::Kind::False, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LBracket, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RBracket, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LBrace, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RBrace, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LParen, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RParen, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Colon, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Semicolon, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Eof, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Dot, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Comma, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Equals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Arrow, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Plus, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoublePlus, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Minus, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Asterisk, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleAsterisk, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Slash, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Percent, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Pipe, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoublePipe, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Ampersand, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleAmpersand, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Caret, 0, 0),
      qasm3::Token(qasm3::Token::Kind::At, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Tilde, 0, 0),
      qasm3::Token(qasm3::Token::Kind::ExclamationPoint, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::NotEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::PlusEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::MinusEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::AsteriskEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::SlashEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::AmpersandEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::PipeEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TildeEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::CaretEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LeftShitEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RightShiftEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::PercentEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleAsteriskEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LessThan, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LessThanEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::GreaterThan, 0, 0),
      qasm3::Token(qasm3::Token::Kind::GreaterThanEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LeftShift, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RightShift, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Imag, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Underscore, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleQuote, 0, 0),
      qasm3::Token(qasm3::Token::Kind::SingleQuote, 0, 0),
      qasm3::Token(qasm3::Token::Kind::BackSlash, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Identifier, 0, 0, "qubit"),
      qasm3::Token(qasm3::Token::Kind::HardwareQubit, 0, 0),
      qasm3::Token(qasm3::Token::Kind::StringLiteral, 0, 0, "hello, world"),
      qasm3::Token(qasm3::Token::Kind::IntegerLiteral, 0, 0),
      qasm3::Token(qasm3::Token::Kind::FloatLiteral, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TimingLiteral, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Sin, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Cos, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Tan, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Exp, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Ln, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Sqrt, 0, 0),
      qasm3::Token(qasm3::Token::Kind::InitialLayout, 0, 0, "i 0 1 2 3"),
      qasm3::Token(qasm3::Token::Kind::OutputPermutation, 0, 0, "o 0 1 2 3"),
  };

  // Print all tokens.
  std::stringstream ss{};
  for (const auto& token : tokens) {
    ss << token << "\n";
  }

  // We expect all tokens to look like this.
  // These are just all tokens joined by a newline.
  const std::string expected =
      "None\n"
      "OPENQASM\n"
      "include\n"
      "DefCalGrammar\n"
      "Def\n"
      "Cal\n"
      "DefCal\n"
      "gate\n"
      "opaque\n"
      "extern\n"
      "box\n"
      "let\n"
      "break\n"
      "continue\n"
      "if\n"
      "else\n"
      "end\n"
      "return\n"
      "for\n"
      "while\n"
      "in\n"
      "pragma\n"
      "input\n"
      "output\n"
      "const\n"
      "readOnly\n"
      "mutable\n"
      "qreg\n"
      "qubit\n"
      "cReg\n"
      "bool\n"
      "bit\n"
      "int\n"
      "uint\n"
      "float\n"
      "angle\n"
      "complex\n"
      "array\n"
      "void\n"
      "duration\n"
      "stretch\n"
      "gphase\n"
      "inv\n"
      "pow\n"
      "ctrl\n"
      "negCtrl\n"
      "#dim\n"
      "durationof\n"
      "delay\n"
      "reset\n"
      "measure\n"
      "barrier\n"
      "true\n"
      "false\n"
      "[\n"
      "]\n"
      "{\n"
      "}\n"
      "(\n"
      ")\n"
      ":\n"
      ";\n"
      "Eof\n"
      ".\n"
      ",\n"
      "=\n"
      "->\n"
      "+\n"
      "++\n"
      "-\n"
      "*\n"
      "**\n"
      "/\n"
      "%\n"
      "|\n"
      "||\n"
      "&\n"
      "&&\n"
      "^\n"
      "@\n"
      "~\n"
      "!\n"
      "==\n"
      "!=\n"
      "+=\n"
      "-=\n"
      "*=\n"
      "/=\n"
      "&=\n"
      "|=\n"
      "~=\n"
      "^=\n"
      "<<=\n"
      ">>=\n"
      "%=\n"
      "**=\n"
      "<\n"
      "<=\n"
      ">\n"
      ">=\n"
      "<<\n"
      ">>\n"
      "imag\n"
      "underscore\n"
      "\"\n"
      "'\n"
      "\\\n"
      // These tokens are not keywords, but have a value associated
      "Identifier (qubit)\n"
      "HardwareQubit\n"
      "StringLiteral (\"hello, world\")\n"
      "IntegerLiteral (0)\n"
      "FloatLiteral (0)\n"
      "TimingLiteral (0 [s])\n"
      "sin\n"
      "cos\n"
      "tan\n"
      "exp\n"
      "ln\n"
      "sqrt\n"
      "InitialLayout (i 0 1 2 3)\n"
      "OutputPermutation (o 0 1 2 3)\n";

  // Now we check if they are the same.
  EXPECT_EQ(ss.str(), expected);
}

TEST_F(Qasm3ParserTest, TestConstEval) {
  qasm3::const_eval::ConstEvalPass constEvalPass{};

  // Test constant eval.
  // The first element of the pair is the expression to be evaluated, the second
  // element is the expected result.
  const auto inputs = std::vector<std::pair<std::shared_ptr<qasm3::Expression>,
                                            qasm3::const_eval::ConstEvalValue>>{
      // integer unsigned
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Power,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(4, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(5, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Multiply,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(2, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Divide,
                    std::make_shared<qasm3::Constant>(6, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Modulo,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(1, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LeftShift,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(1, false)),
                qasm3::const_eval::ConstEvalValue(4, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::RightShift,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(1, false)),
                qasm3::const_eval::ConstEvalValue(1, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThan,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThan,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseAnd,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(3, false)),
                qasm3::const_eval::ConstEvalValue(1, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseXor,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseOr,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},

      // integer signed
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Power,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(4, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(5, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Multiply,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(2, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Divide,
                    std::make_shared<qasm3::Constant>(6, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Modulo,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(1, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LeftShift,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(1, true)),
                qasm3::const_eval::ConstEvalValue(4, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::RightShift,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(1, true)),
                qasm3::const_eval::ConstEvalValue(1, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThan,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThan,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseAnd,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(3, true)),
                qasm3::const_eval::ConstEvalValue(1, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseXor,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseOr,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},

      // float
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Power,
                    std::make_shared<qasm3::Constant>(2.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(4.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(3.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(5.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(3.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Multiply,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(2.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Divide,
                    std::make_shared<qasm3::Constant>(6.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(3.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Modulo,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(1.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThan,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThanOrEqual,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThan,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThanOrEqual,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(true)},

      // bool
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseAnd,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseXor,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseOr,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LogicalAnd,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LogicalOr,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},

      // coercion
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(3.0)),
                qasm3::const_eval::ConstEvalValue(5.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(3.0)),
                qasm3::const_eval::ConstEvalValue(5.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(6U, false),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(4, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(3.0),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(5.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(3.0),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(5.0)},

      // unary expr
      std::pair{
          std::make_shared<qasm3::UnaryExpression>(
              qasm3::UnaryExpression::Op::BitwiseNot,
              std::make_shared<qasm3::Constant>(0xFFFF'FFFF'FFFF'FFFF, false)),
          qasm3::const_eval::ConstEvalValue(0, false)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::LogicalNot,
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Negate,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(-1.0)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Negate,
                    std::make_shared<qasm3::Constant>(1, true)),
                qasm3::const_eval::ConstEvalValue(-1, true)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Sin,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::sin(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Cos,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::cos(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Tan,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::tan(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Exp,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::numbers::e)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Ln,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::log(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Sqrt,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::sqrt(1.0))},

  };

  for (const auto& [expr, expected] : inputs) {
    auto result = constEvalPass.visit(expr);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result, expected);
  }
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralSeconds) {
  qasm3::Scanner scanner(new std::istringstream("1.0s"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralMilliseconds) {
  qasm3::Scanner scanner(new std::istringstream("1.0ms"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0e-3);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralMicroseconds) {
  qasm3::Scanner scanner(new std::istringstream("1.0us"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0e-6);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralNanoseconds) {
  qasm3::Scanner scanner(new std::istringstream("1.0ns"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0e-9);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralPicoseconds) {
  qasm3::Scanner scanner(new std::istringstream("1.0ps"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0e-12);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralDoubleSuffix) {
  qasm3::Scanner scanner(new std::istringstream("1.0dt"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralInvalidSuffix) {
  qasm3::Scanner scanner(new std::istringstream("1.0xs"));
  const auto token = scanner.next();
  EXPECT_NE(token.kind, qasm3::Token::Kind::TimingLiteral);
}

TEST_F(Qasm3ParserTest, TokenKindTimingLiteralMicrosecondsInteger) {
  qasm3::Scanner scanner(new std::istringstream("1us"));
  const auto token = scanner.next();
  EXPECT_EQ(token.kind, qasm3::Token::Kind::TimingLiteral);
  EXPECT_DOUBLE_EQ(token.valReal, 1.0e-6);
}
