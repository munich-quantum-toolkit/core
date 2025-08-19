/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qasm3/Exception.hpp"
#include "qasm3/Importer.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class IO : public testing::Test {
protected:
  void TearDown() override {}

  void SetUp() override {}

  std::size_t nqubits = 0;
  std::size_t seed = 0;
  std::string output3 = "tmp";
  std::string output4 = "tmp.tmp.qasm";
  std::string output5 = "./tmpdir/circuit.qasm";
  std::string output5dir = "tmpdir";
  qc::QuantumComputation qc;
};

namespace {
void compareFiles(const std::string& file1, const std::string& file2) {
  std::ifstream fstream1(file1);
  std::string str1((std::istreambuf_iterator<char>(fstream1)),
                   std::istreambuf_iterator<char>());
  std::ifstream fstream2(file2);
  std::string str2((std::istreambuf_iterator<char>(fstream2)),
                   std::istreambuf_iterator<char>());
  std::erase_if(str1, isspace);
  std::erase_if(str2, isspace);
  ASSERT_EQ(str1, str2);
}
} // namespace

TEST_F(IO, importAndDumpQASM) {
  // Create temporary filenames
  const auto tempDir = std::filesystem::temp_directory_path();
  const auto outputPath = tempDir / "tmp_importAndDumpQASM.txt";
  const auto output2Path = tempDir / "tmp2_importAndDumpQASM.txt";

  constexpr auto input = "../circuits/test.qasm";
  constexpr auto format = qc::Format::OpenQASM2;
  std::cout << "FILE: " << input << "\n";

  qc = qasm3::Importer::importf(input);
  qc.dump(outputPath.string(), format);
  qc.reset();
  qc = qasm3::Importer::importf(outputPath.string());
  qc.dump(output2Path.string(), format);

  compareFiles(outputPath.string(), output2Path.string());
  std::filesystem::remove(outputPath);
  std::filesystem::remove(output2Path);
}

TEST_F(IO, importAndDumpQASMFromConstructor) {
  // Create temporary filenames
  const auto tempDir = std::filesystem::temp_directory_path();
  const auto outputPath = tempDir / "tmp_importAndDumpQASMFromConstructor.txt";
  const auto output2Path =
      tempDir / "tmp2_importAndDumpQASMFromConstructor.txt";

  constexpr auto input = "../circuits/test.qasm";
  constexpr auto format = qc::Format::OpenQASM2;
  std::cout << "FILE: " << input << "\n";

  qc = qasm3::Importer::importf(input);
  qc.dump(outputPath.string(), format);
  qc.reset();
  qc = qasm3::Importer::importf(outputPath.string());
  qc.dump(output2Path.string(), format);

  compareFiles(outputPath.string(), output2Path.string());
  std::filesystem::remove(outputPath);
  std::filesystem::remove(output2Path);
}

TEST_F(IO, dumpValidFilenames) {
  qc.dump(output3, qc::Format::OpenQASM2);
  qc.dump(output4, qc::Format::OpenQASM2);
  qc.dump(output4);

  std::filesystem::create_directory(output5dir);
  qc.dump(output5, qc::Format::OpenQASM2);
  qc.dump(output5);

  std::filesystem::remove(output3);
  std::filesystem::remove(output4);
  std::filesystem::remove(output5);
  std::filesystem::remove(output5dir);
}

TEST_F(IO, importFromStringQASM) {
  qc = qasm3::Importer::imports("qreg q[2];"
                                "U(pi/2,0,pi) q[0];"
                                "CX q[0],q[1];");
  std::cout << qc << "\n";
}

TEST_F(IO, insufficientRegistersQelib) {
  EXPECT_THROW(qc = qasm3::Importer::imports("qreg q[2];"
                                             "cx q[0];"),
               qasm3::CompilerError);
}

TEST_F(IO, insufficientRegistersEnhancedQelib) {
  EXPECT_THROW(qc = qasm3::Importer::imports("qreg q[4];"
                                             "ctrl(3) @ z q[0], q[1], q[2];"),
               qasm3::CompilerError);
}

TEST_F(IO, superfluousRegistersQelib) {
  EXPECT_THROW(qc = qasm3::Importer::imports("qreg q[3];"
                                             "cx q[0], q[1], q[2];"),
               qasm3::CompilerError);
}

TEST_F(IO, superfluousRegistersEnhancedQelib) {
  EXPECT_THROW(
      qc = qasm3::Importer::imports("qreg q[5];"
                                    "ctrl(3) z q[0], q[1], q[2], q[3], q[4];"),
      qasm3::CompilerError);
}

TEST_F(IO, qiskitMcxGray) {
  qc = qasm3::Importer::imports("qreg q[4];"
                                "mcx_gray q[0], q[1], q[2], q[3];");
  std::cout << qc << "\n";
  const auto& gate = qc.front();
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxSkipGateDefinition) {

  qc = qasm3::Importer::imports(
      "qreg q[4];"
      "gate mcx q0,q1,q2,q3 { ctrl(3) @ x q0,q1,q2,q3; }"
      "mcx q[0], q[1], q[2], q[3];");
  std::cout << qc << "\n";
  const auto& gate = qc.front();
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcphase) {
  qc = qasm3::Importer::imports("qreg q[4];"
                                "mcphase(pi) q[0], q[1], q[2], q[3];");
  std::cout << qc << "\n";
  const auto& gate = qc.front();
  EXPECT_EQ(gate->getType(), qc::Z);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcphaseInDeclaration) {
  qc = qasm3::Importer::imports(
      "qreg q[4];"
      "gate foo q0, q1, q2, q3 { mcphase(pi) q0, q1, q2, q3; }"
      "foo q[0], q[1], q[2], q[3];");
  std::cout << qc << "\n";
  const auto& op = qc.front();
  EXPECT_EQ(op->getType(), qc::Z);
  EXPECT_EQ(op->getNcontrols(), 3);
  EXPECT_EQ(op->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxRecursive) {
  qc = qasm3::Importer::imports(
      "qreg q[6];"
      "qreg anc[1];"
      "mcx_recursive q[0], q[1], q[2], q[3], q[4];"
      "mcx_recursive q[0], q[1], q[2], q[3], q[4], q[5], anc[0];");
  std::cout << qc << "\n";
  const auto& gate = qc.at(0);
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 4);
  EXPECT_EQ(gate->getTargets().at(0), 4);
  const auto& second = qc.at(1);
  EXPECT_EQ(second->getType(), qc::X);
  EXPECT_EQ(second->getNcontrols(), 5);
  EXPECT_EQ(second->getTargets().at(0), 5);
}

TEST_F(IO, qiskitMcxVchain) {
  qc = qasm3::Importer::imports("qreg q[4];"
                                "qreg anc[1];"
                                "mcx_vchain q[0], q[1], q[2], q[3], anc[0];");
  std::cout << qc << "\n";
  const auto& gate = qc.front();
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxRecursiveInDeclaration) {
  qc = qasm3::Importer::imports(
      "qreg q[7];"
      "gate foo q0, q1, q2, q3, q4 { mcx_recursive q0, q1, q2, q3, q4; }"
      "gate bar q0, q1, q2, q3, q4, q5, anc { mcx_recursive q0, q1, q2, q3, "
      "q4, q5, anc; }"
      "foo q[0], q[1], q[2], q[3], q[4];"
      "bar q[0], q[1], q[2], q[3], q[4], q[5], q[6];");
  std::cout << qc << "\n";
  const auto& op = qc.at(0);
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getNcontrols(), 4);
  EXPECT_EQ(op->getTargets().at(0), 4);
  const auto& second = qc.at(1);
  EXPECT_EQ(second->getType(), qc::X);
  EXPECT_EQ(second->getNcontrols(), 5);
  EXPECT_EQ(second->getTargets().at(0), 5);
}

TEST_F(IO, qiskitMcxVchainInDeclaration) {
  qc = qasm3::Importer::imports(
      "qreg q[5];"
      "gate foo q0, q1, q2, q3, anc { mcx_vchain q0, q1, q2, q3, anc; }"
      "foo q[0], q[1], q[2], q[3], q[4];");
  std::cout << qc << "\n";
  const auto& op = qc.front();
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getNcontrols(), 3);
  EXPECT_EQ(op->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxDuplicateQubit) {
  EXPECT_THROW(qc = qasm3::Importer::imports(
                   "qreg q[4];"
                   "qreg anc[1];"
                   "mcx_vchain q[0], q[0], q[2], q[3], anc[0];"),
               qasm3::CompilerError);
}

TEST_F(IO, qiskitMcxQubitRegister) {
  EXPECT_THROW(
      qc = qasm3::Importer::imports("qreg q[4];"
                                    "qreg anc[1];"
                                    "mcx_vchain q, q[0], q[2], q[3], anc[0];"),
      qasm3::CompilerError);
}

TEST_F(IO, barrierInDeclaration) {
  qc = qasm3::Importer::imports("qreg q[1];"
                                "gate foo q0 { h q0; barrier q0; h q0; }"
                                "foo q[0];");
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 1);
  const auto& op = qc.at(0);
  EXPECT_EQ(op->getType(), qc::Compound);
  const auto* comp = dynamic_cast<const qc::CompoundOperation*>(op.get());
  ASSERT_NE(comp, nullptr);
  EXPECT_EQ(comp->size(), 3);
  EXPECT_EQ(comp->at(0)->getType(), qc::H);
  EXPECT_EQ(comp->at(1)->getType(), qc::Barrier);
  EXPECT_EQ(comp->at(2)->getType(), qc::H);
}

TEST_F(IO, CommentInDeclaration) {
  qc = qasm3::Importer::imports("qreg q[1];gate foo q0 {"
                                "h q0;"
                                "//x q0;\n"
                                "h q0;"
                                "}"
                                "foo q[0];");
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 1);
  const auto& op = qc.at(0);
  EXPECT_EQ(op->getType(), qc::Compound);
  const auto* comp = dynamic_cast<const qc::CompoundOperation*>(op.get());
  ASSERT_NE(comp, nullptr);
  EXPECT_EQ(comp->size(), 2);
  EXPECT_EQ(comp->at(0)->getType(), qc::H);
  EXPECT_EQ(comp->at(1)->getType(), qc::H);
}

TEST_F(IO, iSWAPDumpIsValid) {
  qc.addQubitRegister(2);
  qc.iswap(0, 1);
  std::cout << qc << "\n";
  const auto qasm = qc.toQASM();
  std::cout << qasm << "\n";
  EXPECT_NO_THROW(qc = qasm3::Importer::imports(qasm));
  std::cout << qc << "\n";
}

TEST_F(IO, iSWAPdagDumpIsValid) {
  qc.addQubitRegister(2);
  qc.iswapdg(0, 1);
  std::cout << qc << "\n";
  const auto qasm = qc.toQASM();
  std::cout << qasm << "\n";
  EXPECT_NO_THROW(qc = qasm3::Importer::imports(qasm));
  std::cout << qc << "\n";
}

TEST_F(IO, PeresDumpIsValid) {
  qc.addQubitRegister(2);
  qc.peres(0, 1);
  std::cout << qc << "\n";
  const auto qasm = qc.toQASM();
  std::cout << qasm << "\n";
  EXPECT_NO_THROW(qc = qasm3::Importer::imports(qasm));
  std::cout << qc << "\n";
}

TEST_F(IO, PeresdagDumpIsValid) {
  qc.addQubitRegister(2);
  qc.peresdg(0, 1);
  std::cout << qc << "\n";
  const auto qasm = qc.toQASM();
  std::cout << qasm << "\n";
  EXPECT_NO_THROW(qc = qasm3::Importer::imports(qasm));
  std::cout << qc << "\n";
}

TEST_F(IO, printingNonUnitary) {
  qc = qasm3::Importer::imports("qreg q[2];"
                                "creg c[2];"
                                "h q[0];"
                                "reset q[0];"
                                "h q[0];"
                                "barrier q;"
                                "measure q -> c;");
  std::cout << qc << "\n";
  for (const auto& op : qc) {
    op->print(std::cout, qc.getNqubits());
    std::cout << "\n";
  }
}

TEST_F(IO, sxAndSxdag) {
  qc = qasm3::Importer::imports("qreg q[1];"
                                "creg c[1];"
                                "gate test q0 { sx q0; sxdg q0;}"
                                "sx q[0];"
                                "sxdg q[0];"
                                "test q[0];");
  std::cout << qc << "\n";
  const auto& op1 = qc.at(0);
  EXPECT_EQ(op1->getType(), qc::OpType::SX);
  const auto& op2 = qc.at(1);
  EXPECT_EQ(op2->getType(), qc::OpType::SXdg);
  const auto& op3 = qc.at(2);
  ASSERT_TRUE(op3->isCompoundOperation());
  auto* compOp = dynamic_cast<qc::CompoundOperation*>(op3.get());
  ASSERT_NE(compOp, nullptr);
  const auto& compOp1 = compOp->at(0);
  EXPECT_EQ(compOp1->getType(), qc::OpType::SX);
  const auto& compOp2 = compOp->at(1);
  EXPECT_EQ(compOp2->getType(), qc::OpType::SXdg);
}

TEST_F(IO, unifyRegisters) {
  qc = qasm3::Importer::imports("qreg q[1];"
                                "qreg r[1];"
                                "x q[0];"
                                "x r[0];");
  std::cout << qc << "\n";
  qc.unifyQuantumRegisters();
  std::cout << qc << "\n";
  const auto qasm = qc.toQASM(false);
  EXPECT_EQ(qasm, "// i 0 1\n"
                  "// o 0 1\n"
                  "OPENQASM 2.0;\n"
                  "include \"qelib1.inc\";\n"
                  "qreg q[2];\n"
                  "x q[0];\n"
                  "x q[1];\n");
}

TEST_F(IO, appendMeasurementsAccordingToOutputPermutation) {
  qc = qasm3::Importer::imports("// o 1\n"
                                "qreg q[2];"
                                "x q[1];");
  qc.appendMeasurementsAccordingToOutputPermutation();
  std::cout << qc << "\n";
  const auto& op = qc.back();
  ASSERT_EQ(op->getType(), qc::OpType::Measure);
  const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
  ASSERT_NE(meas, nullptr);
  EXPECT_EQ(meas->getTargets().size(), 1U);
  EXPECT_EQ(meas->getTargets().front(), 1U);
  EXPECT_EQ(meas->getClassics().size(), 1U);
  EXPECT_EQ(meas->getClassics().front(), 0U);
}

TEST_F(IO, appendMeasurementsAccordingToOutputPermutationAddRegister) {
  qc = qasm3::Importer::imports("// o 0 1\n"
                                "qreg q[2];"
                                "creg d[1];"
                                "x q;");
  qc.appendMeasurementsAccordingToOutputPermutation();
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNcbits(), 2U);
  const auto& op = qc.back();
  ASSERT_EQ(op->getType(), qc::OpType::Measure);
  const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
  ASSERT_NE(meas, nullptr);
  EXPECT_EQ(meas->getTargets().size(), 1U);
  EXPECT_EQ(meas->getTargets().front(), 1U);
  EXPECT_EQ(meas->getClassics().size(), 1U);
  EXPECT_EQ(meas->getClassics().front(), 1U);
  const auto& op2 = *(++qc.rbegin());
  ASSERT_EQ(op2->getType(), qc::OpType::Measure);
  const auto& meas2 = dynamic_cast<const qc::NonUnitaryOperation*>(op2.get());
  ASSERT_NE(meas2, nullptr);
  EXPECT_EQ(meas2->getTargets().size(), 1U);
  EXPECT_EQ(meas2->getTargets().front(), 0U);
  EXPECT_EQ(meas2->getClassics().size(), 1U);
  EXPECT_EQ(meas2->getClassics().front(), 0U);
  const auto qasm = qc.toQASM(false);
  std::cout << qasm << "\n";
  EXPECT_EQ(qasm, "// i 0 1\n"
                  "// o 0 1\n"
                  "OPENQASM 2.0;\n"
                  "include \"qelib1.inc\";\n"
                  "qreg q[2];\n"
                  "creg d[1];\n"
                  "creg c[1];\n"
                  "x q[0];\n"
                  "x q[1];\n"
                  "barrier q;\n"
                  "measure q[0] -> d[0];\n"
                  "measure q[1] -> c[0];\n");
}

TEST_F(IO, NativeTwoQubitGateImportAndExport) {
  const auto gates = std::vector<std::string>{"dcx",
                                              "ecr",
                                              "rxx(0.5)",
                                              "ryy(0.5)",
                                              "rzz(0.5)",
                                              "rzx(0.5)",
                                              "xx_minus_yy(0.5,0.5)",
                                              "xx_plus_yy(0.5,0.5)"};

  const std::string header = "// i 0 1\n"
                             "// o 0 1\n"
                             "OPENQASM 2.0;\n"
                             "include \"qelib1.inc\";\n"
                             "qreg q[2];\n";
  for (const auto& gate : gates) {
    std::stringstream ss{};
    ss << header << gate << " q[0], q[1];\n";
    const auto target = ss.str();
    qc = qasm3::Importer::imports(target);
    std::cout << qc << "\n";
    std::ostringstream oss{};
    qc.dumpOpenQASM(oss, false);
    std::cout << oss.str() << "\n";
    EXPECT_STREQ(oss.str().c_str(), target.c_str());
    qc.reset();
    std::cout << "---\n";
  }
}

TEST_F(IO, UseQelib1Gate) {
  qc = qasm3::Importer::imports("include \"qelib1.inc\";"
                                "qreg q[3];"
                                "rccx q[0], q[1], q[2];");
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNqubits(), 3U);
  EXPECT_EQ(qc.getNops(), 1U);
  EXPECT_EQ(qc.front()->getType(), qc::Compound);
  const auto& op = dynamic_cast<const qc::CompoundOperation*>(qc.front().get());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->size(), 9U);
}

TEST_F(IO, ParameterizedGateDefinition) {
  qc = qasm3::Importer::imports(
      "qreg q[1];"
      "gate foo(theta, beta) q { rz(theta) q; rx(beta) q; }"
      "foo(2*cos(pi/4), 0.5*sin(pi/2)) q[0];");
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNqubits(), 1U);
  EXPECT_EQ(qc.getNops(), 1U);
  EXPECT_EQ(qc.at(0)->getType(), qc::Compound);
  const auto& op = dynamic_cast<const qc::CompoundOperation*>(qc.at(0).get());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->size(), 2U);
  EXPECT_EQ(op->at(0)->getType(), qc::RZ);
  EXPECT_EQ(op->at(1)->getType(), qc::RX);
  const auto& rz = dynamic_cast<const qc::StandardOperation*>(op->at(0).get());
  ASSERT_NE(rz, nullptr);
  const auto& rx = dynamic_cast<const qc::StandardOperation*>(op->at(1).get());
  ASSERT_NE(rx, nullptr);
  EXPECT_EQ(rz->getParameter().at(0), 2 * std::cos(qc::PI_4));
  EXPECT_EQ(rx->getParameter().at(0), 0.5 * std::sin(qc::PI_2));
}

TEST_F(IO, NonExistingInclude) {
  EXPECT_THROW(qc = qasm3::Importer::imports("include \"nonexisting.inc\";"),
               qasm3::CompilerError);
}

TEST_F(IO, NonStandardInclude) {
  std::ofstream ofs{"defs.inc"};
  ofs << "gate foo q { h q; }\n";
  ofs.close();
  qc = qasm3::Importer::imports("include \"defs.inc\";"
                                "qreg q[1];"
                                "foo q[0];");
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNqubits(), 1U);
  EXPECT_EQ(qc.getNops(), 1U);
  EXPECT_EQ(qc.front()->getType(), qc::H);
  std::filesystem::remove("defs.inc");
}

TEST_F(IO, SingleRegistersDoubleCreg) {
  const auto* const qasmIn = "qreg p[1];\n"
                             "qreg q[1];\n"
                             "creg c[2];\n"
                             "measure p[0] -> c[0];";
  qc = qasm3::Importer::imports(qasmIn);
  std::cout << qc << "\n";
  const auto qasmOut = qc.toQASM(false);
  std::cout << qasmOut << "\n";
  EXPECT_NE(qasmOut.find(qasmIn), std::string::npos);
}

TEST_F(IO, MarkAncillaryAndDump) {
  qc = qasm3::Importer::imports("qreg q[2];"
                                "x q[0];"
                                "x q[1];");
  std::cout << qc << "\n";
  qc.setLogicalQubitAncillary(0U);
  EXPECT_EQ(qc.getNancillae(), 1U);
  EXPECT_TRUE(qc.logicalQubitIsAncillary(0U));
  std::cout << qc << "\n";
  const auto qasm = qc.toQASM(false);
  std::cout << qasm << "\n";
  const auto* const expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 2.0;\n"
                               "include \"qelib1.inc\";\n"
                               "qreg q[2];\n"
                               "x q[0];\n"
                               "x q[1];\n";
  EXPECT_EQ(qasm, expected);
}

TEST_F(IO, dumpEmptyOpenQASM) {
  qc = qasm3::Importer::imports("");

  std::string const openQASM2 =
      "// i\n// o\nOPENQASM 2.0;\ninclude \"qelib1.inc\";\n";
  std::string const openQASM3 =
      "// i\n// o\nOPENQASM 3.0;\ninclude \"stdgates.inc\";\n";

  EXPECT_EQ(openQASM2, qc.toQASM(false));
  EXPECT_EQ(openQASM3, qc.toQASM(true));
}

TEST_F(IO, fromCompoundOperation) {
  qc.addQubitRegister(2);
  qc.addClassicalRegister(2);
  qc.x(1);
  qc.measure(1, 1);
  const auto compound = qc.asCompoundOperation();
  const auto qc2 = qc::QuantumComputation::fromCompoundOperation(*compound);

  EXPECT_EQ(qc2.getNqubits(), 2);
  EXPECT_EQ(qc2.getNcbits(), 2);
  EXPECT_EQ(qc2.getNops(), 2);
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "x q[1];\n"
                               "c[1] = measure q[1];\n";
  const auto actual = qc2.toQASM();
  EXPECT_EQ(expected, actual);
}

TEST_F(IO, ifElseOperationOperationToOpenQASM3) {
  qc.addQubitRegister(2);
  const auto& creg = qc.addClassicalRegister(2);
  qc.if_(qc::X, 0, 0);
  qc.if_(qc::X, 1, creg, 1U);
  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "if (c[0]) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c == 1) {\n"
                               "  x q[1];\n"
                               "}\n";

  const auto actual = qc.toQASM();
  EXPECT_EQ(expected, actual);
}

TEST_F(IO, ifElseOperationInvalidBitComparison) {
  qc.addQubitRegister(1);
  qc.addClassicalRegister(1);
  try {
    qc.if_(qc::X, 0, 0, true, qc::Lt);
    FAIL() << "Expected an exception for invalid expected value.";
  } catch (const std::invalid_argument& e) {
    EXPECT_STREQ(e.what(),
                 "Inequality comparisons on a single bit are not supported.");
    SUCCEED();
  } catch (...) {
    FAIL() << "Expected an invalid_argument exception.";
  }
}

TEST_F(IO, dumpingIncompleteOutputPermutationNotStartingAtZero) {
  qc.addQubitRegister(2);
  qc.addClassicalRegister(1);
  qc.measure(1, 0);
  qc.initializeIOMapping();
  const auto qasm = qc.toQASM();
  std::cout << qasm << "\n";
  const auto qc2 = qasm3::Importer::imports(qasm);
  EXPECT_EQ(qc, qc2);
}

TEST_F(IO, indexedRegisterOperands) {
  const auto& q = qc.addQubitRegister(2);
  const auto& c = qc.addClassicalRegister(2);

  qc.h(q[0]);
  qc.cx(q[0], q[1]);
  qc.measure(q[0], c[0]);
  qc.measure(q[1], c[1]);

  const auto qasm = qc.toQASM();
  const auto* const expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "h q[0];\n"
                               "cx q[0], q[1];\n"
                               "c[0] = measure q[0];\n"
                               "c[1] = measure q[1];\n";
  EXPECT_EQ(qasm, expected);
}
