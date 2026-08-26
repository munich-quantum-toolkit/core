/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/OpenQASMSerializer.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/Register.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <ostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace qc {

namespace {

constexpr std::size_t OUTPUT_INDENT_SIZE = 2U;

template <class RegisterType>
void printSortedRegisters(
    const std::unordered_map<std::string, RegisterType>& registers,
    const std::string& identifier, std::ostream& output, const bool openQASM3) {
  std::map<std::size_t, RegisterType> sortedRegs{};
  for (const auto& [name, reg] : registers) {
    sortedRegs.emplace(reg.getStartIndex(), reg);
  }

  for (const auto& [_, reg] : sortedRegs) {
    if (openQASM3) {
      output << identifier << "[" << reg.getSize() << "] " << reg.getName()
             << ";\n";
    } else {
      output << identifier << " " << reg.getName() << "[" << reg.getSize()
             << "];\n";
    }
  }
}

template <class RegisterMap, class Index>
bool isWholeRegister(const RegisterMap& regMap, const Index start,
                     const Index end) {
  const auto& startReg = regMap.at(start).first;
  const auto& endReg = regMap.at(end).first;
  return startReg == endReg && startReg.getStartIndex() == start &&
         endReg.getEndIndex() == end;
}

void dumpControls(std::ostringstream& serialized,
                  const StandardOperation& operation) {
  const auto& controls = operation.getControls();
  if (controls.empty()) {
    return;
  }

  // If an operation is in stdgates.inc, use a c prefix instead of ctrl @.
  if (bool printBuiltin = std::ranges::none_of(controls,
                                               [](const Control& control) {
                                                 return control.type ==
                                                        Control::Type::Neg;
                                               });
      printBuiltin) {
    const auto numControls = controls.size();
    switch (operation.getType()) {
    case P:
    case RX:
    case Y:
    case RY:
    case Z:
    case RZ:
    case H:
    case SWAP:
      printBuiltin = numControls == 1U;
      break;
    case X:
      printBuiltin = numControls == 1U || numControls == 2U;
      break;
    default:
      printBuiltin = false;
    }
    if (printBuiltin) {
      serialized << std::string(numControls, 'c');
      return;
    }
  }

  auto currentType = controls.begin()->type;
  auto count = 0;

  for (const auto& control : controls) {
    if (control.type == currentType) {
      ++count;
    } else {
      serialized << (currentType == Control::Type::Neg ? "negctrl" : "ctrl");
      if (count > 1) {
        serialized << "(" << count << ")";
      }
      serialized << " @ ";
      currentType = control.type;
      count = 1;
    }
  }

  serialized << (currentType == Control::Type::Neg ? "negctrl" : "ctrl");
  if (count > 1) {
    serialized << "(" << count << ")";
  }
  serialized << " @ ";
}

void dumpGateType(std::ostream& output, std::ostringstream& serialized,
                  const StandardOperation& operation,
                  const QubitIndexToRegisterMap& qubitMap) {
  const auto type = operation.getType();
  const auto& controls = operation.getControls();
  const auto& targets = operation.getTargets();
  const auto& parameter = operation.getParameter();

  switch (type) {
  case GPhase:
    serialized << "gphase(" << parameter.at(0) << ")";
    break;
  case I:
    serialized << "id";
    break;
  case Barrier:
    assert(controls.empty());
    serialized << "barrier";
    break;
  case H:
    serialized << "h";
    break;
  case X:
    serialized << "x";
    break;
  case Y:
    serialized << "y";
    break;
  case Z:
    serialized << "z";
    break;
  case S:
    serialized << (controls.empty() ? "s" : "p(pi/2)");
    break;
  case Sdg:
    serialized << (controls.empty() ? "sdg" : "p(-pi/2)");
    break;
  case T:
    serialized << (controls.empty() ? "t" : "p(pi/4)");
    break;
  case Tdg:
    serialized << (controls.empty() ? "tdg" : "p(-pi/4)");
    break;
  case V:
    serialized << "U(pi/2,-pi/2,pi/2)";
    break;
  case Vdg:
    serialized << "U(pi/2,pi/2,-pi/2)";
    break;
  case U:
    serialized << "U(" << parameter[0] << "," << parameter[1] << ","
               << parameter[2] << ")";
    break;
  case U2:
    serialized << "U(pi/2," << parameter[0] << "," << parameter[1] << ")";
    break;
  case P:
    serialized << "p(" << parameter[0] << ")";
    break;
  case SX:
    serialized << "sx";
    break;
  case SXdg:
    serialized << "sxdg";
    break;
  case RX:
    serialized << "rx(" << parameter[0] << ")";
    break;
  case RY:
    serialized << "ry(" << parameter[0] << ")";
    break;
  case RZ:
    serialized << "rz(" << parameter[0] << ")";
    break;
  case R:
    serialized << "r(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case DCX:
    serialized << "dcx";
    break;
  case ECR:
    serialized << "ecr";
    break;
  case RXX:
    serialized << "rxx(" << parameter[0] << ")";
    break;
  case RYY:
    serialized << "ryy(" << parameter[0] << ")";
    break;
  case RZZ:
    serialized << "rzz(" << parameter[0] << ")";
    break;
  case RZX:
    serialized << "rzx(" << parameter[0] << ")";
    break;
  case XXminusYY:
    serialized << "xx_minus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case XXplusYY:
    serialized << "xx_plus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case RCCX:
    serialized << "rccx";
    break;
  case SWAP:
    serialized << "swap";
    break;
  case iSWAP:
    serialized << "iswap";
    break;
  case iSWAPdg:
    serialized << "iswapdg";
    break;
  case Peres:
    output << serialized.str() << "cx";
    for (const auto& control : controls) {
      output << " " << qubitMap.at(control.qubit).second << ",";
    }
    output << " " << qubitMap.at(targets[1]).second << ", "
           << qubitMap.at(targets[0]).second << ";\n";

    output << serialized.str() << "x";
    for (const auto& control : controls) {
      output << " " << qubitMap.at(control.qubit).second << ",";
    }
    output << " " << qubitMap.at(targets[1]).second << ";\n";
    return;
  case Peresdg:
    output << serialized.str() << "x";
    for (const auto& control : controls) {
      output << " " << qubitMap.at(control.qubit).second << ",";
    }
    output << " " << qubitMap.at(targets[1]).second << ";\n";

    output << serialized.str() << "cx";
    for (const auto& control : controls) {
      output << " " << qubitMap.at(control.qubit).second << ",";
    }
    output << " " << qubitMap.at(targets[1]).second << ", "
           << qubitMap.at(targets[0]).second << ";\n";
    return;
  default:
    std::cerr << "gate type " << toString(type)
              << " could not be converted to OpenQASM\n.";
  }

  output << serialized.str();

  for (auto it = controls.begin(); it != controls.end();) {
    output << " " << qubitMap.at(it->qubit).second;
    if (++it != controls.end() || !targets.empty()) {
      output << ",";
    }
  }

  if (!targets.empty() && type == Barrier &&
      isWholeRegister(qubitMap, targets.front(), targets.back())) {
    output << " " << qubitMap.at(targets.front()).first.getName();
  } else {
    for (auto it = targets.begin(); it != targets.end();) {
      output << " " << qubitMap.at(*it).second;
      if (++it != targets.end()) {
        output << ",";
      }
    }
  }
  output << ";\n";
}

void dumpStandardOperation(std::ostream& output,
                           const StandardOperation& operation,
                           const QubitIndexToRegisterMap& qubitMap,
                           const std::size_t indent, const bool openQASM3) {
  std::ostringstream serialized;
  serialized << std::setprecision(std::numeric_limits<fp>::digits10);
  serialized << std::string(indent * OUTPUT_INDENT_SIZE, ' ');

  const auto& controls = operation.getControls();
  if (openQASM3) {
    dumpControls(serialized, operation);
    dumpGateType(output, serialized, operation, qubitMap);
    return;
  }

  const auto type = operation.getType();
  if ((controls.size() > 1U && type != X) || controls.size() > 2U) {
    std::cout << "[WARNING] Multiple controlled gates are not natively "
                 "supported by OpenQASM. "
              << "However, this library can parse .qasm files with multiple "
                 "controlled gates (e.g., cccx) correctly. "
              << "Thus, while not valid vanilla OpenQASM, the dumped file will "
                 "work with this library.\n";
  }

  serialized << std::string(controls.size(), 'c');
  const bool isSpecialGate = type == Peres || type == Peresdg;
  if (!isSpecialGate) {
    for (const auto& control : controls) {
      if (control.type == Control::Type::Neg) {
        output << "x " << qubitMap.at(control.qubit).second << ";\n";
      }
    }
  }

  dumpGateType(output, serialized, operation, qubitMap);

  if (!isSpecialGate) {
    for (const auto& control : controls) {
      if (control.type == Control::Type::Neg) {
        output << "x " << qubitMap.at(control.qubit).second << ";\n";
      }
    }
  }
}

void dumpNonUnitaryOperation(std::ostream& output,
                             const NonUnitaryOperation& operation,
                             const QubitIndexToRegisterMap& qubitMap,
                             const BitIndexToRegisterMap& bitMap,
                             const std::size_t indent, const bool openQASM3) {
  output << std::string(indent * OUTPUT_INDENT_SIZE, ' ');

  const auto& targets = operation.getTargets();
  const auto& classics = operation.getClassics();
  const auto type = operation.getType();
  if (isWholeRegister(qubitMap, targets.front(), targets.back()) &&
      (type != Measure ||
       isWholeRegister(bitMap, classics.front(), classics.back()))) {
    if (type == Measure && openQASM3) {
      output << bitMap.at(classics.front()).first.getName() << " = ";
    }
    output << toString(type) << " "
           << qubitMap.at(targets.front()).first.getName();
    if (type == Measure && !openQASM3) {
      output << " -> " << bitMap.at(classics.front()).first.getName();
    }
    output << ";\n";
    return;
  }

  auto classicsIt = classics.cbegin();
  for (const auto& target : targets) {
    const auto& qreg = qubitMap.at(target);
    if (type == Measure && openQASM3) {
      const auto& creg = bitMap.at(*classicsIt);
      output << creg.second << " = ";
    }
    output << toString(type) << " " << qreg.second;
    if (type == Measure && !openQASM3) {
      const auto& creg = bitMap.at(*classicsIt);
      output << " -> " << creg.second;
      ++classicsIt;
    }
    output << ";\n";
  }
}

void dumpOperation(std::ostream& output, const Operation& operation,
                   const QubitIndexToRegisterMap& qubitMap,
                   const BitIndexToRegisterMap& bitMap, std::size_t indent,
                   bool openQASM3);

void dumpIfElseOperation(std::ostream& output, const IfElseOperation& operation,
                         const QubitIndexToRegisterMap& qubitMap,
                         const BitIndexToRegisterMap& bitMap,
                         const std::size_t indent, const bool openQASM3) {
  output << std::string(indent * OUTPUT_INDENT_SIZE, ' ') << "if (";
  if (const auto& controlRegister = operation.getControlRegister();
      controlRegister.has_value()) {
    assert(!operation.getControlBit().has_value());
    output << controlRegister->getName() << ' ' << operation.getComparisonKind()
           << ' ' << operation.getExpectedValueRegister();
  } else if (const auto& controlBit = operation.getControlBit();
             controlBit.has_value()) {
    output << (!operation.getExpectedValueBit() ? "!" : "")
           << bitMap.at(*controlBit).second;
  }
  output << ") {\n";

  if (const auto* thenOperation = operation.getThenOp();
      thenOperation != nullptr) {
    dumpOperation(output, *thenOperation, qubitMap, bitMap, indent + 1U,
                  openQASM3);
  }

  const auto* elseOperation = operation.getElseOp();
  if (elseOperation == nullptr) {
    output << "}\n";
    return;
  }

  output << "}";
  if (openQASM3) {
    output << " else {\n";
    dumpOperation(output, *elseOperation, qubitMap, bitMap, indent + 1U,
                  openQASM3);
  } else {
    output << '\n' << "if (";
    if (const auto& controlRegister = operation.getControlRegister();
        controlRegister.has_value()) {
      assert(!operation.getControlBit().has_value());
      output << controlRegister->getName() << ' '
             << getInvertedComparisonKind(operation.getComparisonKind()) << ' '
             << operation.getExpectedValueRegister();
    }
    if (const auto& controlBit = operation.getControlBit();
        controlBit.has_value()) {
      assert(!operation.getControlRegister().has_value());
      output << (operation.getExpectedValueBit() ? "!" : "")
             << bitMap.at(*controlBit).second;
    }
    output << ") {\n";
    dumpOperation(output, *elseOperation, qubitMap, bitMap, indent + 1U,
                  openQASM3);
  }
  output << "}\n";
}

void dumpOperation(std::ostream& output, const Operation& operation,
                   const QubitIndexToRegisterMap& qubitMap,
                   const BitIndexToRegisterMap& bitMap,
                   const std::size_t indent, const bool openQASM3) {
  if (dynamic_cast<const SymbolicOperation*>(&operation) != nullptr) {
    if (openQASM3) {
      throw std::runtime_error(
          "Printing OpenQASM 3.0 parameterized gates is not supported yet!");
    }
    throw std::runtime_error(
        "OpenQASM 2.0 doesn't support parameterized gates!");
  }
  if (const auto* ifElse = dynamic_cast<const IfElseOperation*>(&operation);
      ifElse != nullptr) {
    dumpIfElseOperation(output, *ifElse, qubitMap, bitMap, indent, openQASM3);
    return;
  }
  if (const auto* compound = dynamic_cast<const CompoundOperation*>(&operation);
      compound != nullptr) {
    for (const auto& nestedOperation : *compound) {
      dumpOperation(output, *nestedOperation, qubitMap, bitMap, indent,
                    openQASM3);
    }
    return;
  }
  if (const auto* nonUnitary =
          dynamic_cast<const NonUnitaryOperation*>(&operation);
      nonUnitary != nullptr) {
    dumpNonUnitaryOperation(output, *nonUnitary, qubitMap, bitMap, indent,
                            openQASM3);
    return;
  }
  if (const auto* standard = dynamic_cast<const StandardOperation*>(&operation);
      standard != nullptr) {
    dumpStandardOperation(output, *standard, qubitMap, indent, openQASM3);
    return;
  }
  throw std::invalid_argument(
      "Operation type is not supported by the OpenQASM serializer.");
}

} // namespace

void OpenQASMSerializer::serialize(
    const QuantumComputation& computation) const {
  const auto openQASM3 = format == Format::OpenQASM3;

  Permutation qubitToIndex{};
  Permutation inverseInitialLayout{};
  Qubit index = 0U;
  for (const auto& [physical, logical] : computation.initialLayout) {
    inverseInitialLayout.emplace(logical, index);
    qubitToIndex[physical] = index;
    ++index;
  }
  output << "// i";
  for (const auto& [logical, physical] : inverseInitialLayout) {
    output << " " << static_cast<std::size_t>(physical);
  }
  output << "\n";

  Permutation inverseOutputPermutation{};
  for (const auto& [physical, logical] : computation.outputPermutation) {
    inverseOutputPermutation.emplace(logical, qubitToIndex[physical]);
  }
  output << "// o";
  for (const auto& [logical, physical] : inverseOutputPermutation) {
    output << " " << physical;
  }
  output << "\n";

  if (openQASM3) {
    output << "OPENQASM 3.0;\n";
    output << "include \"stdgates.inc\";\n";
  } else {
    output << "OPENQASM 2.0;\n";
    output << "include \"qelib1.inc\";\n";
  }

  auto combinedRegs = computation.getQuantumRegisters();
  for (const auto& reg : computation.getAncillaRegisters()) {
    combinedRegs.emplace(reg);
  }
  printSortedRegisters(combinedRegs, openQASM3 ? "qubit" : "qreg", output,
                       openQASM3);

  const auto& classicalRegisters = computation.getClassicalRegisters();
  printSortedRegisters(classicalRegisters, openQASM3 ? "bit" : "creg", output,
                       openQASM3);

  QubitIndexToRegisterMap qubitMap{};
  for (const auto& [_, reg] : combinedRegs) {
    const auto bound = reg.getStartIndex() + reg.getSize();
    for (Qubit i = reg.getStartIndex(); i < bound; ++i) {
      qubitMap.try_emplace(i, reg, reg.toString(i));
    }
  }

  BitIndexToRegisterMap bitMap{};
  for (const auto& [_, reg] : classicalRegisters) {
    const auto bound = reg.getStartIndex() + reg.getSize();
    for (Bit i = reg.getStartIndex(); i < bound; ++i) {
      bitMap.try_emplace(i, reg, reg.toString(i));
    }
  }

  for (const auto& operation : computation) {
    serialize(*operation, qubitMap, bitMap);
  }
}

void OpenQASMSerializer::serialize(const Operation& operation,
                                   const QubitIndexToRegisterMap& qubitMap,
                                   const BitIndexToRegisterMap& bitMap,
                                   const std::size_t indent) const {
  dumpOperation(output, operation, qubitMap, bitMap, indent,
                format == Format::OpenQASM3);
}

} // namespace qc
