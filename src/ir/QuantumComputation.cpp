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

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "ir/operations/SymbolicOperation.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace qc {

namespace {
template <class RegisterType>
void printSortedRegisters(
    const std::unordered_map<std::string, RegisterType>& registers,
    const std::string& identifier, std::ostream& of, const bool openQASM3) {
  // sort regs by start index
  std::map<size_t, RegisterType> sortedRegs{};
  for (const auto& [name, reg] : registers) {
    sortedRegs.emplace(reg.getStartIndex(), reg);
  }

  for (const auto& r : sortedRegs) {
    const auto& reg = r.second;
    if (openQASM3) {
      of << identifier << "[" << reg.getSize() << "] " << reg.getName()
         << ";\n";
    } else {
      of << identifier << " " << reg.getName() << "[" << reg.getSize()
         << "];\n";
    }
  }
}

void consolidateRegister(QuantumRegisterMap& regs) {
  bool finished = regs.empty();
  while (!finished) {
    for (const auto& [name, qreg] : regs) {
      finished = true;
      // check if lower part of register
      if (name.length() > 2 && name.compare(name.size() - 2, 2, "_l") == 0) {
        auto lowidx = qreg.getStartIndex();
        auto lownum = qreg.getSize();
        // search for higher part of register
        auto highname = name.substr(0, name.size() - 1) + 'h';
        if (const auto it = regs.find(highname); it != regs.end()) {
          auto& highReg = it->second;
          auto highidx = highReg.getStartIndex();
          auto highnum = highReg.getSize();
          // fusion of registers possible
          if (lowidx + lownum == highidx) {
            finished = false;
            auto targetname = name.substr(0, name.size() - 2);
            auto targetidx = lowidx;
            auto targetnum = lownum + highnum;
            regs.erase(name);
            regs.erase(highname);
            regs.try_emplace(targetname, targetidx, targetnum, targetname);
          }
        }
        break;
      }
    }
  }
}

/**
 * @brief Removes a certain qubit in a register from the register map
 * @details If this was the last qubit in the register, the register is
 * deleted. Removals at the beginning or the end of a register just modify the
 * existing register. Removals in the middle of a register split the register
 * into two new registers. The new registers are named by appending "_l" and
 * "_h" to the original register name.
 * @param regs A collection of all the registers
 * @param reg The name of the register containing the qubit to be removed
 * @param idx The index of the qubit in the register to be removed
 */
void removeQubitfromQubitRegister(QuantumRegisterMap& regs,
                                  QuantumRegister& reg, const Qubit idx) {
  if (idx == 0) {
    // last remaining qubit of register
    if (reg.getSize() == 1) {
      // delete register
      regs.erase(reg.getName());
    }
    // first qubit of register
    else {
      reg.getStartIndex()++;
      reg.getSize()--;
    }
    // last index
  } else if (idx == reg.getSize() - 1) {
    // reduce count of register
    reg.getSize()--;
  } else {
    const auto startIndex = reg.getStartIndex();
    const auto count = reg.getSize();
    const auto lowPart = reg.getName() + "_l";
    const auto lowIndex = startIndex;
    const auto lowCount = idx;
    const auto highPart = reg.getName() + "_h";
    const auto highIndex = startIndex + idx + 1;
    const auto highCount = count - idx - 1;

    regs.erase(reg.getName());
    regs.try_emplace(lowPart, lowIndex, lowCount, lowPart);
    regs.try_emplace(highPart, highIndex, highCount, highPart);
  }
}

/**
 * @brief Adds a qubit to a register in the register map
 * @details If the register map is empty, a new register is created with the
 * default name. If the qubit can be appended to the start or the end of an
 * existing register, it is appended. Otherwise a new register is created with
 * the default name and the qubit index appended.
 * @param regs A collection of all the registers
 * @param physicalQubitIndex The index of the qubit to be added
 * @param defaultRegName The default name of the register to be created
 */
void addQubitToQubitRegister(QuantumRegisterMap& regs, Qubit physicalQubitIndex,
                             const std::string& defaultRegName) {
  auto fusionPossible = false;
  for (auto& [name, reg] : regs) {
    auto& startIndex = reg.getStartIndex();
    auto& count = reg.getSize();
    // 1st case: can append to start of existing register
    if (startIndex == physicalQubitIndex + 1) {
      startIndex--;
      count++;
      fusionPossible = true;
      break;
    }
    // 2nd case: can append to end of existing register
    if (startIndex + count == physicalQubitIndex) {
      count++;
      fusionPossible = true;
      break;
    }
  }

  consolidateRegister(regs);

  if (regs.empty()) {
    regs.try_emplace(defaultRegName, physicalQubitIndex, 1, defaultRegName);
  } else if (!fusionPossible) {
    const auto newRegName =
        defaultRegName + "_" + std::to_string(physicalQubitIndex);
    regs.try_emplace(newRegName, physicalQubitIndex, 1, newRegName);
  }
}
} // namespace

/***
 * Public Methods
 ***/
std::size_t QuantumComputation::getNindividualOps() const {
  std::size_t nops = 0;
  for (const auto& op : ops) {
    if (const auto* const comp =
            dynamic_cast<const CompoundOperation*>(op.get());
        comp != nullptr) {
      nops += comp->size();
    } else {
      ++nops;
    }
  }

  return nops;
}

std::size_t QuantumComputation::getNsingleQubitOps() const {
  std::size_t nops = 0;
  for (const auto& op : ops) {
    if (!op->isUnitary()) {
      continue;
    }

    if (const auto* const comp =
            dynamic_cast<const CompoundOperation*>(op.get());
        comp != nullptr) {
      for (const auto& subop : *comp) {
        if (subop->isUnitary() && !subop->isControlled() &&
            subop->getNtargets() == 1U) {
          ++nops;
        }
      }
    } else {
      if (!op->isControlled() && op->getNtargets() == 1U) {
        ++nops;
      }
    }
  }
  return nops;
}

std::size_t QuantumComputation::getDepth() const {
  if (empty()) {
    return 0U;
  }

  std::vector<std::size_t> depths(getNqubits(), 0U);
  for (const auto& op : ops) {
    op->addDepthContribution(depths);
  }

  return *std::max_element(depths.begin(), depths.end());
}

void QuantumComputation::initializeIOMapping() {
  // try gathering (additional) output permutation information from
  // measurements, e.g., a measurement
  //      `measure q[i] -> c[j];`
  // implies that the j-th (logical) output is obtained from measuring the i-th
  // physical qubit.
  const bool outputPermutationFound = !outputPermutation.empty();

  // track whether the circuit contains measurements at the end of the circuit
  // if it does, then all qubits that are not measured shall be considered
  // garbage outputs
  bool outputPermutationFromMeasurements = false;
  std::set<Qubit> measuredQubits{};

  for (const auto& opIt : ops) {
    if (const auto* const op = dynamic_cast<NonUnitaryOperation*>(opIt.get());
        op != nullptr && op->getType() == Measure) {
      outputPermutationFromMeasurements = true;
      assert(op->getTargets().size() == op->getClassics().size());
      auto classicIt = op->getClassics().cbegin();
      for (const auto& q : op->getTargets()) {
        const auto qubitidx = q;
        // only the first measurement of a qubit is used to determine the output
        // permutation
        if (measuredQubits.contains(qubitidx)) {
          continue;
        }

        const auto bitidx = *classicIt;
        if (outputPermutationFound) {
          // output permutation was already set before -> permute existing
          // values
          if (const auto current = outputPermutation.at(qubitidx);
              static_cast<std::size_t>(current) != bitidx) {
            for (auto& p : outputPermutation) {
              if (static_cast<std::size_t>(p.second) == bitidx) {
                p.second = current;
                break;
              }
            }
            outputPermutation.at(qubitidx) = static_cast<Qubit>(bitidx);
          }
        } else {
          // directly set permutation if none was set beforehand
          outputPermutation[qubitidx] = static_cast<Qubit>(bitidx);
        }
        measuredQubits.emplace(qubitidx);
        ++classicIt;
      }
    }
  }

  // clear any qubits that were not measured from the output permutation
  // these will be marked garbage further down below
  if (outputPermutationFromMeasurements) {
    auto it = outputPermutation.begin();
    while (it != outputPermutation.end()) {
      if (!measuredQubits.contains(it->first)) {
        it = outputPermutation.erase(it);
      } else {
        ++it;
      }
    }
  }

  garbage.assign(nqubits + nancillae, false);
  for (const auto& [physicalIn, logicalIn] : initialLayout) {
    // if the qubit is not an output, mark it as garbage
    const bool isOutput = std::ranges::any_of(
        outputPermutation,
        [&logicIn = logicalIn](const auto& p) { return p.second == logicIn; });
    if (!isOutput) {
      setLogicalQubitGarbage(logicalIn);
    }

    // if the qubit is an ancillary and idle, mark it as garbage
    if (const bool isIdle = isIdleQubit(physicalIn);
        logicalQubitIsAncillary(logicalIn) && isIdle) {
      setLogicalQubitGarbage(logicalIn);
    }
  }
}

const QuantumRegister&
QuantumComputation::addQubitRegister(std::size_t nq,
                                     const std::string& regName) {
  if (quantumRegisters.contains(regName)) {
    throw std::runtime_error("[addQubitRegister] Register " + regName +
                             " already exists");
  }

  if (nq == 0) {
    throw std::runtime_error(
        "[addQubitRegister] New register size must be larger than 0");
  }

  if (nancillae != 0) {
    throw std::runtime_error(
        "[addQubitRegister] Cannot add qubit register after ancillary "
        "qubits have been added");
  }

  quantumRegisters.try_emplace(regName, static_cast<Qubit>(nqubits), nq,
                               regName);
  for (std::size_t i = 0; i < nq; ++i) {
    auto j = static_cast<Qubit>(nqubits + i);
    initialLayout.emplace(j, j);
    outputPermutation.emplace(j, j);
  }
  nqubits += nq;
  ancillary.resize(nqubits + nancillae);
  garbage.resize(nqubits + nancillae);
  return quantumRegisters.at(regName);
}

const ClassicalRegister&
QuantumComputation::addClassicalRegister(std::size_t nc,
                                         const std::string& regName) {
  if (classicalRegisters.contains(regName)) {
    throw std::runtime_error("[addClassicalRegister] Register " + regName +
                             " already exists");
  }
  if (nc == 0) {
    throw std::runtime_error(
        "[addClassicalRegister] New register size must be larger than 0");
  }

  const auto [it, success] =
      classicalRegisters.try_emplace(regName, nclassics, nc, regName);
  assert(success);
  nclassics += nc;
  return it->second;
}

const QuantumRegister&
QuantumComputation::addAncillaryRegister(std::size_t nq,
                                         const std::string& regName) {
  if (ancillaRegisters.contains(regName)) {
    throw std::runtime_error("[addAncillaryRegister] Register " + regName +
                             " already exists");
  }

  if (nq == 0) {
    throw std::runtime_error(
        "[addAncillaryRegister] New register size must be larger than 0");
  }

  const auto totalqubits = static_cast<Qubit>(nqubits + nancillae);
  ancillaRegisters.try_emplace(regName, totalqubits, nq, regName);
  ancillary.resize(totalqubits + nq);
  garbage.resize(totalqubits + nq);
  for (std::size_t i = 0; i < nq; ++i) {
    auto j = static_cast<Qubit>(totalqubits + i);
    initialLayout.emplace(j, j);
    outputPermutation.emplace(j, j);
    ancillary[j] = true;
  }
  nancillae += nq;
  return ancillaRegisters.at(regName);
}

std::pair<Qubit, std::optional<Qubit>>
QuantumComputation::removeQubit(const Qubit logicalQubitIndex) {
  // Find index of the physical qubit i is assigned to
  const auto physicalQubitIndex = getPhysicalQubitIndex(logicalQubitIndex);

  // get register and register-index of the corresponding qubit
  auto& reg = getQubitRegister(physicalQubitIndex);
  const auto& idx = reg.getLocalIndex(physicalQubitIndex);

  if (physicalQubitIsAncillary(physicalQubitIndex)) {
    removeQubitfromQubitRegister(ancillaRegisters, reg, idx);
    // reduce ancilla count
    nancillae--;
  } else {
    removeQubitfromQubitRegister(quantumRegisters, reg, idx);
    // reduce qubit count
    if (ancillary.at(logicalQubitIndex)) {
      // if the qubit is ancillary, it is not counted as a qubit
      nancillae--;
    } else {
      nqubits--;
    }
  }

  // adjust initial layout permutation
  initialLayout.erase(physicalQubitIndex);

  // remove potential output permutation entry
  std::optional<Qubit> outputQubitIndex{};
  if (const auto it = outputPermutation.find(physicalQubitIndex);
      it != outputPermutation.end()) {
    outputQubitIndex = it->second;
    // erasing entry
    outputPermutation.erase(physicalQubitIndex);
  }

  // update ancillary and garbage tracking
  const auto totalQubits = nqubits + nancillae;
  for (std::size_t i = logicalQubitIndex; i < totalQubits; ++i) {
    ancillary[i] = ancillary[i + 1];
    garbage[i] = garbage[i + 1];
  }
  // unset last entry
  ancillary[totalQubits] = false;
  garbage[totalQubits] = false;

  return {physicalQubitIndex, outputQubitIndex};
}

// adds j-th physical qubit as ancilla to the end of reg or creates the register
// if necessary
void QuantumComputation::addAncillaryQubit(
    Qubit physicalQubitIndex, std::optional<Qubit> outputQubitIndex) {
  if (initialLayout.count(physicalQubitIndex) > 0 ||
      outputPermutation.count(physicalQubitIndex) > 0) {
    throw std::runtime_error(
        "[addAncillaryQubit] Attempting to insert physical "
        "qubit that is already assigned");
  }

  addQubitToQubitRegister(ancillaRegisters, physicalQubitIndex, "anc");

  // index of logical qubit
  const auto logicalQubitIndex = nqubits + nancillae;

  // resize ancillary and garbage tracking vectors
  ancillary.resize(logicalQubitIndex + 1U);
  garbage.resize(logicalQubitIndex + 1U);

  // increase ancillae count and mark as ancillary
  nancillae++;
  ancillary[logicalQubitIndex] = true;

  // adjust initial layout
  initialLayout.emplace(physicalQubitIndex,
                        static_cast<Qubit>(logicalQubitIndex));

  // adjust output permutation
  if (outputQubitIndex.has_value()) {
    outputPermutation.emplace(physicalQubitIndex, *outputQubitIndex);
  } else {
    // if a qubit is not relevant for the output, it is considered garbage
    garbage[logicalQubitIndex] = true;
  }
}

void QuantumComputation::addQubit(const Qubit logicalQubitIndex,
                                  const Qubit physicalQubitIndex,
                                  const std::optional<Qubit> outputQubitIndex) {
  if (initialLayout.count(physicalQubitIndex) > 0 ||
      outputPermutation.count(physicalQubitIndex) > 0) {
    throw std::runtime_error(
        "[addQubit] Attempting to insert physical qubit that is "
        "already assigned");
  }

  if (logicalQubitIndex > nqubits) {
    throw std::runtime_error(
        "[addQubit] There are currently only " + std::to_string(nqubits) +
        " qubits in the circuit. Adding " + std::to_string(logicalQubitIndex) +
        " is therefore not possible at the moment.");
    // TODO: this does not necessarily have to lead to an error. A new qubit
    // register could be created and all ancillaries shifted
  }

  addQubitToQubitRegister(quantumRegisters, physicalQubitIndex, "q");

  // increase qubit count
  nqubits++;
  // adjust initial layout
  initialLayout.emplace(physicalQubitIndex, logicalQubitIndex);
  if (outputQubitIndex.has_value()) {
    // adjust output permutation
    outputPermutation.emplace(physicalQubitIndex, *outputQubitIndex);
  }

  // update ancillary and garbage tracking
  const auto totalQubits = nqubits + nancillae;
  ancillary.resize(totalQubits);
  garbage.resize(totalQubits);
  for (auto i = totalQubits - 1; i > logicalQubitIndex; --i) {
    ancillary[i] = ancillary[i - 1];
    garbage[i] = garbage[i - 1];
  }
  // unset new entry
  ancillary[logicalQubitIndex] = false;
  garbage[logicalQubitIndex] = false;
}
QuantumComputation
QuantumComputation::instantiate(const VariableAssignment& assignment) const {
  QuantumComputation result(*this);
  result.instantiateInplace(assignment);
  return result;
}

void QuantumComputation::invert() {
  for (const auto& op : ops) {
    op->invert();
  }
  std::ranges::reverse(ops);

  if (initialLayout.size() == outputPermutation.size()) {
    std::swap(initialLayout, outputPermutation);
  } else {
    std::cerr << "Warning: Inverting a circuit with different initial layout "
                 "and output permutation sizes. This is not supported yet.\n"
                 "The circuit will be inverted, but the initial layout and "
                 "output permutation will not be swapped.\n";
  }
}

bool QuantumComputation::operator==(const QuantumComputation& rhs) const {
  if (nqubits != rhs.nqubits || nancillae != rhs.nancillae ||
      nclassics != rhs.nclassics || quantumRegisters != rhs.quantumRegisters ||
      classicalRegisters != rhs.classicalRegisters ||
      ancillaRegisters != rhs.ancillaRegisters ||
      initialLayout != rhs.initialLayout ||
      outputPermutation != rhs.outputPermutation ||
      ancillary != rhs.ancillary || garbage != rhs.garbage ||
      seed != rhs.seed || globalPhase != rhs.globalPhase ||
      occurringVariables != rhs.occurringVariables) {
    return false;
  }

  if (ops.size() != rhs.ops.size()) {
    return false;
  }

  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (*ops[i] != *rhs.ops[i]) {
      return false;
    }
  }

  return true;
}

std::ostream& QuantumComputation::print(std::ostream& os) const {
  os << name << "\n";
  const auto width =
      ops.empty() ? 1 : static_cast<int>(std::log10(ops.size()) + 1.);

  os << std::setw(width + 1) << "i:";
  for (const auto& [physical, logical] : initialLayout) {
    if (ancillary[logical]) {
      os << "\033[31m";
    }
    os << std::setw(4) << logical << "\033[0m";
  }
  os << "\n";

  size_t i = 0U;
  for (const auto& op : ops) {
    os << std::setw(width) << ++i << ":";
    op->print(os, initialLayout, static_cast<std::size_t>(width) + 1U,
              getNqubits());
    os << "\n";
  }

  os << std::setw(width + 1) << "o:";
  for (const auto& physicalQubit : initialLayout) {
    auto it = outputPermutation.find(physicalQubit.first);
    if (it == outputPermutation.end()) {
      os << "\033[31m" << std::setw(4) << "|" << "\033[0m";
    } else {
      os << std::setw(4) << it->second;
    }
  }
  os << "\n";
  return os;
}

std::ostream& QuantumComputation::printStatistics(std::ostream& os) const {
  os << "QC Statistics:";
  os << "\n\tn: " << static_cast<std::size_t>(nqubits);
  os << "\n\tanc: " << static_cast<std::size_t>(nancillae);
  os << "\n\tm: " << ops.size();
  os << "\n--------------\n";
  return os;
}

void QuantumComputation::dumpOpenQASM(std::ostream& of, bool openQASM3) const {
  // dump initial layout and output permutation

  // since it might happen that the physical qubit indices are not consecutive,
  // due to qubit removals, we need to adjust them accordingly.
  Permutation qubitToIndex{};

  Permutation inverseInitialLayout{};
  Qubit idx = 0;
  for (const auto& [physical, logical] : initialLayout) {
    inverseInitialLayout.emplace(logical, idx);
    qubitToIndex[physical] = idx;
    ++idx;
  }
  of << "// i";
  for (const auto& [logical, physical] : inverseInitialLayout) {
    of << " " << static_cast<std::size_t>(physical);
  }
  of << "\n";

  Permutation inverseOutputPermutation{};
  for (const auto& [physical, logical] : outputPermutation) {
    inverseOutputPermutation.emplace(logical, qubitToIndex[physical]);
  }
  of << "// o";
  for (const auto& [logical, physical] : inverseOutputPermutation) {
    of << " " << physical;
  }
  of << "\n";

  if (openQASM3) {
    of << "OPENQASM 3.0;\n";
    of << "include \"stdgates.inc\";\n";
  } else {
    of << "OPENQASM 2.0;\n";
    of << "include \"qelib1.inc\";\n";
  }

  // combine qregs and ancregs
  auto combinedRegs = quantumRegisters;
  for (const auto& reg : ancillaRegisters) {
    combinedRegs.emplace(reg);
  }
  printSortedRegisters(combinedRegs, openQASM3 ? "qubit" : "qreg", of,
                       openQASM3);

  printSortedRegisters(classicalRegisters, openQASM3 ? "bit" : "creg", of,
                       openQASM3);

  // build qubit index -> register map
  QubitIndexToRegisterMap qubitMap{};
  for (const auto& [_, reg] : combinedRegs) {
    const auto bound = reg.getStartIndex() + reg.getSize();
    for (Qubit i = reg.getStartIndex(); i < bound; ++i) {
      qubitMap.try_emplace(i, reg, reg.toString(i));
    }
  }
  // build classical index -> register map
  BitIndexToRegisterMap bitMap{};
  for (const auto& [_, reg] : classicalRegisters) {
    const auto bound = reg.getStartIndex() + reg.getSize();
    for (Bit i = reg.getStartIndex(); i < bound; ++i) {
      bitMap.try_emplace(i, reg, reg.toString(i));
    }
  }

  for (const auto& op : ops) {
    op->dumpOpenQASM(of, qubitMap, bitMap, 0, openQASM3);
  }
}

std::string QuantumComputation::toQASM(const bool qasm3) const {
  std::stringstream ss;
  dumpOpenQASM(ss, qasm3);
  return ss.str();
}
std::unique_ptr<Operation> QuantumComputation::asOperation() {
  if (ops.empty()) {
    return {};
  }
  if (ops.size() == 1) {
    auto op = std::move(ops.front());
    ops.clear();
    return op;
  }
  return asCompoundOperation();
}
void QuantumComputation::reset() {
  ops.clear();
  nqubits = 0;
  nclassics = 0;
  nancillae = 0;
  quantumRegisters.clear();
  classicalRegisters.clear();
  ancillaRegisters.clear();
  initialLayout.clear();
  outputPermutation.clear();
}

void QuantumComputation::dump(const std::string& filename,
                              const Format format) const {
  auto of = std::ofstream(filename);
  if (!of.good()) {
    throw std::runtime_error("[dump] Error opening file: " + filename);
  }
  if (format == Format::OpenQASM3) {
    dumpOpenQASM(of, true);
  } else {
    dumpOpenQASM(of, false);
  }
}

bool QuantumComputation::isIdleQubit(const Qubit physicalQubit) const {
  return std::ranges::none_of(
      ops.cbegin(), ops.cend(),
      [&physicalQubit](const auto& op) { return op->actsOn(physicalQubit); });
}

void QuantumComputation::stripIdleQubits(bool force) {
  auto layoutCopy = initialLayout;
  for (auto physicalQubitIt = layoutCopy.rbegin();
       physicalQubitIt != layoutCopy.rend(); ++physicalQubitIt) {
    if (const auto physicalQubitIndex = physicalQubitIt->first;
        isIdleQubit(physicalQubitIndex)) {
      if (auto it = outputPermutation.find(physicalQubitIndex);
          it != outputPermutation.end() && !force) {
        continue;
      }

      const auto logicalQubitIndex = initialLayout.at(physicalQubitIndex);
      // check whether the logical qubit is used in the output permutation
      auto usedInOutputPermutation = false;
      for (const auto& [physical, logical] : outputPermutation) {
        if (logical == logicalQubitIndex) {
          usedInOutputPermutation = true;
          break;
        }
      }
      if (usedInOutputPermutation && !force) {
        // cannot strip a logical qubit that is used in the output permutation
        continue;
      }

      removeQubit(logicalQubitIndex);

      if (logicalQubitIndex < nqubits + nancillae) {
        for (auto& [physical, logical] : initialLayout) {
          if (logical > logicalQubitIndex) {
            --logical;
          }
        }

        for (auto& [physical, logical] : outputPermutation) {
          if (logical > logicalQubitIndex) {
            --logical;
          }
        }
      }
    }
  }
}

QuantumRegister&
QuantumComputation::getQubitRegister(const Qubit physicalQubitIndex) {
  for (auto& [_, reg] : quantumRegisters) {
    if (reg.contains(physicalQubitIndex)) {
      return reg;
    }
  }

  for (auto& [_, reg] : ancillaRegisters) {
    if (reg.contains(physicalQubitIndex)) {
      return reg;
    }
  }

  throw std::runtime_error("[getQubitRegister] Qubit index " +
                           std::to_string(physicalQubitIndex) +
                           " not found in any register");
}

Qubit QuantumComputation::getPhysicalQubitIndex(
    const Qubit logicalQubitIndex) const {
  for (const auto& [physical, logical] : initialLayout) {
    if (logical == logicalQubitIndex) {
      return physical;
    }
  }
  throw std::runtime_error("[getPhysicalQubitIndex] Logical qubit index " +
                           std::to_string(logicalQubitIndex) +
                           " not found in initial layout");
}

std::ostream&
QuantumComputation::printPermutation(const Permutation& permutation,
                                     std::ostream& os) {
  for (const auto& [physical, logical] : permutation) {
    os << "\t" << physical << ": " << logical << "\n";
  }
  return os;
}

Qubit QuantumComputation::getHighestLogicalQubitIndex() const {
  return initialLayout.maxValue();
}

Qubit QuantumComputation::getHighestPhysicalQubitIndex() const {
  return initialLayout.maxKey();
}

bool QuantumComputation::physicalQubitIsAncillary(
    const Qubit physicalQubitIndex) const {
  return std::ranges::any_of(ancillaRegisters,
                             [&physicalQubitIndex](const auto& reg) {
                               return reg.second.contains(physicalQubitIndex);
                             });
}

void QuantumComputation::setLogicalQubitAncillary(
    const Qubit logicalQubitIndex) {
  if (logicalQubitIsAncillary(logicalQubitIndex)) {
    return;
  }

  nqubits--;
  nancillae++;
  ancillary[logicalQubitIndex] = true;
}

void QuantumComputation::setLogicalQubitsAncillary(
    const Qubit minLogicalQubitIndex, const Qubit maxLogicalQubitIndex) {
  for (Qubit i = minLogicalQubitIndex; i <= maxLogicalQubitIndex; i++) {
    setLogicalQubitAncillary(i);
  }
}

void QuantumComputation::setLogicalQubitGarbage(const Qubit logicalQubitIndex) {
  garbage[logicalQubitIndex] = true;
  // setting a logical qubit garbage also means removing it from the output
  // permutation if it was present before
  for (auto it = outputPermutation.begin(); it != outputPermutation.end();
       ++it) {
    if (it->second == logicalQubitIndex) {
      outputPermutation.erase(it);
      break;
    }
  }
}

void QuantumComputation::setLogicalQubitsGarbage(
    const Qubit minLogicalQubitIndex, const Qubit maxLogicalQubitIndex) {
  for (Qubit i = minLogicalQubitIndex; i <= maxLogicalQubitIndex; i++) {
    setLogicalQubitGarbage(i);
  }
}

[[nodiscard]] std::pair<bool, std::optional<Qubit>>
QuantumComputation::containsLogicalQubit(const Qubit logicalQubitIndex) const {
  if (const auto it =
          std::ranges::find_if(initialLayout,
                               [&logicalQubitIndex](const auto& mapping) {
                                 return mapping.second == logicalQubitIndex;
                               });
      it != initialLayout.cend()) {
    return {true, it->first};
  }
  return {false, std::nullopt};
}

bool QuantumComputation::isLastOperationOnQubit(
    const const_iterator& opIt, const const_iterator& end) const {
  if (opIt == end) {
    return true;
  }

  // determine which qubits the gate acts on
  std::vector<bool> actson(nqubits + nancillae);
  for (std::size_t i = 0; i < actson.size(); ++i) {
    if ((*opIt)->actsOn(static_cast<Qubit>(i))) {
      actson[i] = true;
    }
  }

  // iterate over remaining gates and check if any act on qubits overlapping
  // with the target gate
  auto atEnd = opIt;
  std::advance(atEnd, 1);
  while (atEnd != end) {
    for (std::size_t i = 0; i < actson.size(); ++i) {
      if (actson[i] && (*atEnd)->actsOn(static_cast<Qubit>(i))) {
        return false;
      }
    }
    ++atEnd;
  }
  return true;
}

const QuantumRegister&
QuantumComputation::unifyQuantumRegisters(const std::string& regName) {
  ancillaRegisters.clear();
  quantumRegisters.clear();
  nqubits += nancillae;
  nancillae = 0;
  quantumRegisters.try_emplace(regName, 0, nqubits, regName);
  return quantumRegisters.at(regName);
}

void QuantumComputation::appendMeasurementsAccordingToOutputPermutation(
    const std::string& registerName) {
  // ensure that the circuit contains enough classical registers
  if (classicalRegisters.empty()) {
    // in case there are no registers, create a new one
    addClassicalRegister(outputPermutation.size(), registerName);
  } else if (nclassics < outputPermutation.size()) {
    if (classicalRegisters.contains(registerName)) {
      throw std::runtime_error(
          "[appendMeasurementsAccordingToOutputPermutation] Register " +
          registerName + " already exists but is too small");
    }
    addClassicalRegister(outputPermutation.size() - nclassics, registerName);
  }
  barrier();
  // append measurements according to output permutation
  for (const auto& [qubit, clbit] : outputPermutation) {
    measure(qubit, clbit);
  }
}

void QuantumComputation::checkQubitRange(const Qubit qubit) const {
  if (const auto it = initialLayout.find(qubit);
      it == initialLayout.end() || it->second >= getNqubits()) {
    throw std::out_of_range("Qubit index out of range: " +
                            std::to_string(qubit));
  }
}
void QuantumComputation::checkQubitRange(const Qubit qubit,
                                         const Controls& controls) const {
  checkQubitRange(qubit);
  for (const auto& [ctrl, _] : controls) {
    checkQubitRange(ctrl);
  }
}

void QuantumComputation::checkQubitRange(const Qubit qubit0, const Qubit qubit1,
                                         const Controls& controls) const {
  checkQubitRange(qubit0, controls);
  checkQubitRange(qubit1);
}

void QuantumComputation::checkQubitRange(
    const std::vector<Qubit>& qubits) const {
  for (const auto& qubit : qubits) {
    checkQubitRange(qubit);
  }
}

void QuantumComputation::checkBitRange(const Bit bit) const {
  if (bit >= nclassics) {
    std::stringstream ss{};
    ss << "Classical bit index " << bit << " not found in any register";
    throw std::runtime_error(ss.str());
  }
}

void QuantumComputation::checkBitRange(const std::vector<Bit>& bits) const {
  for (const auto& bit : bits) {
    checkBitRange(bit);
  }
}

void QuantumComputation::checkClassicalRegister(
    const ClassicalRegister& creg) const {
  if (creg.getStartIndex() + creg.getSize() > nclassics) {
    std::stringstream ss{};
    ss << "Classical register starting at index " << creg.getStartIndex()
       << " with " << creg.getSize() << " bits is too large! The circuit has "
       << nclassics << " classical bits.";
    throw std::runtime_error(ss.str());
  }
}

void QuantumComputation::reverse() { std::ranges::reverse(ops); }

QuantumComputation::QuantumComputation(const std::size_t nq,
                                       const std::size_t nc,
                                       const std::size_t s)
    : seed(s) {
  if (nq > 0) {
    addQubitRegister(nq);
  }
  if (nc > 0) {
    addClassicalRegister(nc);
  }
  if (seed != 0) {
    mt.seed(seed);
  } else {
    // create and properly seed rng
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::ranges::generate(randomData, [&rd]() { return rd(); });
    std::seed_seq seeds(std::begin(randomData), std::end(randomData));
    mt.seed(seeds);
  }
}

QuantumComputation::QuantumComputation(const QuantumComputation& qc)
    : nqubits(qc.nqubits), nclassics(qc.nclassics), nancillae(qc.nancillae),
      name(qc.name), quantumRegisters(qc.quantumRegisters),
      classicalRegisters(qc.classicalRegisters),
      ancillaRegisters(qc.ancillaRegisters), ancillary(qc.ancillary),
      garbage(qc.garbage), mt(qc.mt), seed(qc.seed),
      globalPhase(qc.globalPhase), occurringVariables(qc.occurringVariables),
      initialLayout(qc.initialLayout), outputPermutation(qc.outputPermutation) {
  ops.reserve(qc.ops.size());
  for (const auto& op : qc.ops) {
    emplace_back(op->clone());
  }
}
QuantumComputation&
QuantumComputation::operator=(const QuantumComputation& qc) {
  if (this != &qc) {
    nqubits = qc.nqubits;
    nclassics = qc.nclassics;
    nancillae = qc.nancillae;
    name = qc.name;
    quantumRegisters = qc.quantumRegisters;
    classicalRegisters = qc.classicalRegisters;
    ancillaRegisters = qc.ancillaRegisters;
    mt = qc.mt;
    seed = qc.seed;
    globalPhase = qc.globalPhase;
    occurringVariables = qc.occurringVariables;
    initialLayout = qc.initialLayout;
    outputPermutation = qc.outputPermutation;
    ancillary = qc.ancillary;
    garbage = qc.garbage;

    ops.clear();
    ops.reserve(qc.ops.size());
    for (const auto& op : qc.ops) {
      emplace_back(op->clone());
    }
  }
  return *this;
}

void QuantumComputation::addVariable(const SymbolOrNumber& expr) {
  if (std::holds_alternative<Symbolic>(expr)) {
    const auto& sym = std::get<Symbolic>(expr);
    for (const auto& term : sym) {
      occurringVariables.insert(term.getVar());
    }
  }
}

bool QuantumComputation::isVariableFree() const {
  return std::ranges::all_of(
      ops, [](const auto& op) { return !op->isSymbolicOperation(); });
}

// Instantiates this computation
void QuantumComputation::instantiateInplace(
    const VariableAssignment& assignment) {
  for (auto& op : ops) {
    if (auto* symOp = dynamic_cast<SymbolicOperation*>(op.get());
        symOp != nullptr) {
      symOp->instantiate(assignment);
      // if the operation is fully instantiated, it can be replaced by the
      // corresponding standard operation
      if (symOp->isStandardOperation()) {
        op = std::make_unique<StandardOperation>(
            *dynamic_cast<StandardOperation*>(symOp));
      }
    }
  }
  // after an operation is instantiated, the respective parameters can be
  // removed from the circuit
  for (const auto& [var, _] : assignment) {
    occurringVariables.erase(var);
  }
}

void QuantumComputation::reorderOperations() {
  using DAG = std::vector<std::deque<std::unique_ptr<Operation>*>>;
  using DAGIterator = std::deque<std::unique_ptr<Operation>*>::iterator;
  using DAGIterators = std::vector<DAGIterator>;

  Qubit highestPhysicalQubit = 0;
  for (const auto& q : initialLayout) {
    highestPhysicalQubit = std::max(q.first, highestPhysicalQubit);
  }

  auto dag = DAG(highestPhysicalQubit + 1);

  for (auto& op : ops) {
    const auto usedQubits = op->getUsedQubits();
    for (const auto q : usedQubits) {
      dag.at(q).push_back(&op);
    }
  }

  // initialize iterators
  DAGIterators dagIterators{dag.size()};
  for (size_t q = 0; q < dag.size(); ++q) {
    if (dag.at(q).empty()) {
      // qubit is idle
      dagIterators.at(q) = dag.at(q).end();
    } else {
      // point to first operation
      dagIterators.at(q) = dag.at(q).begin();
    }
  }

  std::vector<std::unique_ptr<Operation>> newOps{};

  // iterate over DAG in depth-first fashion starting from the top-most qubit
  const auto msq = dag.size() - 1;
  bool done = false;
  while (!done) {
    // assume that everything is done
    done = true;

    // iterate over qubits in reverse order
    for (auto q = static_cast<std::make_signed_t<Qubit>>(msq); q >= 0; --q) {
      // nothing to be done for this qubit
      if (dagIterators.at(static_cast<std::size_t>(q)) ==
          dag.at(static_cast<std::size_t>(q)).end()) {
        continue;
      }
      done = false;

      // get the current operation on the qubit
      auto& it = dagIterators.at(static_cast<std::size_t>(q));
      auto& op = **it;

      // check whether the gate can be scheduled, i.e. whether all qubits it
      // acts on are at this operation
      bool executable = true;
      std::vector<bool> actsOn(dag.size());
      actsOn[static_cast<std::size_t>(q)] = true;
      for (std::size_t i = 0; i < dag.size(); ++i) {
        // actually check in reverse order
        const auto qb =
            static_cast<std::make_signed_t<Qubit>>(dag.size() - 1 - i);
        if (qb != q && op->actsOn(static_cast<Qubit>(qb))) {
          actsOn[static_cast<std::size_t>(qb)] = true;

          assert(dagIterators.at(static_cast<std::size_t>(qb)) !=
                 dag.at(static_cast<std::size_t>(qb)).end());
          // check whether operation is executable for the currently considered
          // qubit
          if (*dagIterators.at(static_cast<std::size_t>(qb)) != *it) {
            executable = false;
            break;
          }
        }
      }

      // continue, if this gate is not yet executable
      if (!executable) {
        continue;
      }

      // gate is executable, move it to the new vector
      newOps.emplace_back(std::move(op));

      // now increase all corresponding iterators
      for (std::size_t i = 0; i < dag.size(); ++i) {
        if (actsOn[i]) {
          ++(dagIterators.at(i));
        }
      }
    }
  }

  // clear all the operations from the quantum circuit
  ops.clear();
  // move all operations from the newly created vector to the original one
  std::ranges::move(newOps, std::back_inserter(ops));
}

namespace {
bool isDynamicCircuit(const std::unique_ptr<Operation>* op,
                      std::vector<bool>& measured) {
  assert(op != nullptr);
  const auto& it = *op;
  // whenever a classic-controlled or a reset operation are encountered
  // the circuit has to be dynamic.
  if (it->getType() == Reset || it->isClassicControlledOperation()) {
    return true;
  }

  if (it->isStandardOperation()) {
    // Whenever a qubit has already been measured, the circuit is dynamic
    const auto& usedQubits = it->getUsedQubits();
    return std::ranges::any_of(
        usedQubits, [&measured](const auto& q) { return measured[q]; });
  }

  if (it->getType() == Measure) {
    for (const auto& b : it->getTargets()) {
      measured[b] = true;
    }
    return false;
  }

  assert(it->isCompoundOperation());
  const auto& compOp = dynamic_cast<const CompoundOperation&>(*it);
  return std::ranges::any_of(compOp, [&measured](const auto& g) {
    return isDynamicCircuit(&g, measured);
  });
}
} // namespace

bool QuantumComputation::isDynamic() const {
  // marks whether a qubit in the DAG has been measured
  std::vector<bool> measured(getHighestPhysicalQubitIndex() + 1, false);
  return std::ranges::any_of(cbegin(), cend(), [&measured](const auto& op) {
    return ::qc::isDynamicCircuit(&op, measured);
  });
}

QuantumComputation
QuantumComputation::fromCompoundOperation(const CompoundOperation& op) {
  QuantumComputation qc{};
  Qubit maxQubitIndex = 0;
  Bit maxBitIndex = 0;
  for (const auto& g : op) {
    // clone the gate and add it to the circuit
    qc.emplace_back(g->clone());

    // update the maximum qubit index
    const auto& usedQubits = g->getUsedQubits();
    for (const auto& q : usedQubits) {
      maxQubitIndex = std::max(maxQubitIndex, q);
    }

    if (g->getType() == Measure) {
      // update the maximum classical bit index
      const auto& measureOp = dynamic_cast<const NonUnitaryOperation&>(*g);
      const auto& classics = measureOp.getClassics();
      for (const auto& c : classics) {
        maxBitIndex = std::max(maxBitIndex, c);
      }
    }
  }

  // The following also sets the initial layout and the output permutation
  qc.addQubitRegister(static_cast<size_t>(maxQubitIndex) + 1);
  qc.addClassicalRegister(static_cast<size_t>(maxBitIndex) + 1);

  return qc;
}

std::size_t QuantumComputation::getNmeasuredQubits() const noexcept {
  return getNqubits() - getNgarbageQubits();
}
std::size_t QuantumComputation::getNgarbageQubits() const {
  return static_cast<std::size_t>(
      std::count(getGarbage().cbegin(), getGarbage().cend(), true));
}

///---------------------------------------------------------------------------
///                            \n Operations \n
///---------------------------------------------------------------------------

void QuantumComputation::gphase(const fp angle) {
  globalPhase += angle;
  // normalize to [0, 2pi)
  while (globalPhase < 0) {
    globalPhase += 2 * PI;
  }
  while (globalPhase >= 2 * PI) {
    globalPhase -= 2 * PI;
  }
}

#define DEFINE_SINGLE_TARGET_OPERATION(op)                                     \
  void QuantumComputation::op(const Qubit target) {                            \
    mc##op(Controls{}, target);                                                \
  }                                                                            \
  void QuantumComputation::c##op(const Control& control, const Qubit target) { \
    mc##op(Controls{control}, target);                                         \
  }                                                                            \
  void QuantumComputation::mc##op(const Controls& controls,                    \
                                  const Qubit target) {                        \
    checkQubitRange(target, controls);                                         \
    emplace_back<StandardOperation>(controls, target, opTypeFromString(#op));  \
  }

DEFINE_SINGLE_TARGET_OPERATION(i)
DEFINE_SINGLE_TARGET_OPERATION(x)
DEFINE_SINGLE_TARGET_OPERATION(y)
DEFINE_SINGLE_TARGET_OPERATION(z)
DEFINE_SINGLE_TARGET_OPERATION(h)
DEFINE_SINGLE_TARGET_OPERATION(s)
DEFINE_SINGLE_TARGET_OPERATION(sdg)
DEFINE_SINGLE_TARGET_OPERATION(t)
DEFINE_SINGLE_TARGET_OPERATION(tdg)
DEFINE_SINGLE_TARGET_OPERATION(v)
DEFINE_SINGLE_TARGET_OPERATION(vdg)
DEFINE_SINGLE_TARGET_OPERATION(sx)
DEFINE_SINGLE_TARGET_OPERATION(sxdg)

#undef DEFINE_SINGLE_TARGET_OPERATION

#define DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(op, param)             \
  void QuantumComputation::op(const SymbolOrNumber&(param),                    \
                              const Qubit target) {                            \
    mc##op(param, Controls{}, target);                                         \
  }                                                                            \
  void QuantumComputation::c##op(const SymbolOrNumber&(param),                 \
                                 const Control& control, const Qubit target) { \
    mc##op(param, Controls{control}, target);                                  \
  }                                                                            \
  void QuantumComputation::mc##op(const SymbolOrNumber&(param),                \
                                  const Controls& controls,                    \
                                  const Qubit target) {                        \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param)) {                                   \
      emplace_back<StandardOperation>(controls, target, opTypeFromString(#op), \
                                      std::vector{std::get<fp>(param)});       \
    } else {                                                                   \
      addVariables(param);                                                     \
      emplace_back<SymbolicOperation>(controls, target, opTypeFromString(#op), \
                                      std::vector{param});                     \
    }                                                                          \
  }

DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rx, theta)
DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(ry, theta)
DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rz, theta)
DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(p, theta)

#undef DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION

#define DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)       \
  void QuantumComputation::op(const SymbolOrNumber&(param0),                   \
                              const SymbolOrNumber&(param1),                   \
                              const Qubit target) {                            \
    mc##op(param0, param1, Controls{}, target);                                \
  }                                                                            \
  void QuantumComputation::c##op(const SymbolOrNumber&(param0),                \
                                 const SymbolOrNumber&(param1),                \
                                 const Control& control, const Qubit target) { \
    mc##op(param0, param1, Controls{control}, target);                         \
  }                                                                            \
  void QuantumComputation::mc##op(                                             \
      const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),            \
      const Controls& controls, const Qubit target) {                          \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1)) {                                  \
      emplace_back<StandardOperation>(                                         \
          controls, target, opTypeFromString(#op),                             \
          std::vector{std::get<fp>(param0), std::get<fp>(param1)});            \
    } else {                                                                   \
      addVariables(param0, param1);                                            \
      emplace_back<SymbolicOperation>(controls, target, opTypeFromString(#op), \
                                      std::vector{param0, param1});            \
    }                                                                          \
  }

DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(u2, phi, lambda)

#undef DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION

#define DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(op, param0, param1,     \
                                                       param2)                 \
  void QuantumComputation::op(                                                 \
      const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),            \
      const SymbolOrNumber&(param2), const Qubit target) {                     \
    mc##op(param0, param1, param2, Controls{}, target);                        \
  }                                                                            \
  void QuantumComputation::c##op(const SymbolOrNumber&(param0),                \
                                 const SymbolOrNumber&(param1),                \
                                 const SymbolOrNumber&(param2),                \
                                 const Control& control, const Qubit target) { \
    mc##op(param0, param1, param2, Controls{control}, target);                 \
  }                                                                            \
  void QuantumComputation::mc##op(                                             \
      const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),            \
      const SymbolOrNumber&(param2), const Controls& controls,                 \
      const Qubit target) {                                                    \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1) &&                                  \
        std::holds_alternative<fp>(param2)) {                                  \
      emplace_back<StandardOperation>(controls, target, opTypeFromString(#op), \
                                      std::vector{std::get<fp>(param0),        \
                                                  std::get<fp>(param1),        \
                                                  std::get<fp>(param2)});      \
    } else {                                                                   \
      addVariables(param0, param1, param2);                                    \
      emplace_back<SymbolicOperation>(controls, target, opTypeFromString(#op), \
                                      std::vector{param0, param1, param2});    \
    }                                                                          \
  }

DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(u, theta, phi, lambda)

#undef DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION

#define DEFINE_TWO_TARGET_OPERATION(op)                                        \
  void QuantumComputation::op(const Qubit target0, const Qubit target1) {      \
    mc##op(Controls{}, target0, target1);                                      \
  }                                                                            \
  void QuantumComputation::c##op(const Control& control, const Qubit target0,  \
                                 const Qubit target1) {                        \
    mc##op(Controls{control}, target0, target1);                               \
  }                                                                            \
  void QuantumComputation::mc##op(const Controls& controls,                    \
                                  const Qubit target0, const Qubit target1) {  \
    checkQubitRange(target0, target1, controls);                               \
    emplace_back<StandardOperation>(controls, target0, target1,                \
                                    opTypeFromString(#op));                    \
  }

DEFINE_TWO_TARGET_OPERATION(swap) // NOLINT: bugprone-exception-escape
DEFINE_TWO_TARGET_OPERATION(dcx)
DEFINE_TWO_TARGET_OPERATION(ecr)
DEFINE_TWO_TARGET_OPERATION(iswap)
DEFINE_TWO_TARGET_OPERATION(iswapdg)
DEFINE_TWO_TARGET_OPERATION(peres)
DEFINE_TWO_TARGET_OPERATION(peresdg)
DEFINE_TWO_TARGET_OPERATION(move)

#undef DEFINE_TWO_TARGET_OPERATION

#define DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(op, param)                \
  void QuantumComputation::op(const SymbolOrNumber&(param),                    \
                              const Qubit target0, const Qubit target1) {      \
    mc##op(param, Controls{}, target0, target1);                               \
  }                                                                            \
  void QuantumComputation::c##op(const SymbolOrNumber&(param),                 \
                                 const Control& control, const Qubit target0,  \
                                 const Qubit target1) {                        \
    mc##op(param, Controls{control}, target0, target1);                        \
  }                                                                            \
  void QuantumComputation::mc##op(const SymbolOrNumber&(param),                \
                                  const Controls& controls,                    \
                                  const Qubit target0, const Qubit target1) {  \
    checkQubitRange(target0, target1, controls);                               \
    if (std::holds_alternative<fp>(param)) {                                   \
      emplace_back<StandardOperation>(controls, target0, target1,              \
                                      opTypeFromString(#op),                   \
                                      std::vector{std::get<fp>(param)});       \
    } else {                                                                   \
      addVariables(param);                                                     \
      emplace_back<SymbolicOperation>(controls, target0, target1,              \
                                      opTypeFromString(#op),                   \
                                      std::vector{param});                     \
    }                                                                          \
  }

DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rxx, theta)
DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(ryy, theta)
DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzz, theta)
DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzx, theta)

#undef DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION

#define DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)          \
  void QuantumComputation::op(const SymbolOrNumber&(param0),                   \
                              const SymbolOrNumber&(param1),                   \
                              const Qubit target0, const Qubit target1) {      \
    mc##op(param0, param1, Controls{}, target0, target1);                      \
  }                                                                            \
  void QuantumComputation::c##op(                                              \
      const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),            \
      const Control& control, const Qubit target0, const Qubit target1) {      \
    mc##op(param0, param1, Controls{control}, target0, target1);               \
  }                                                                            \
  void QuantumComputation::mc##op(                                             \
      const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),            \
      const Controls& controls, const Qubit target0, const Qubit target1) {    \
    checkQubitRange(target0, target1, controls);                               \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1)) {                                  \
      emplace_back<StandardOperation>(                                         \
          controls, target0, target1, opTypeFromString(#op),                   \
          std::vector{std::get<fp>(param0), std::get<fp>(param1)});            \
    } else {                                                                   \
      addVariables(param0, param1);                                            \
      emplace_back<SymbolicOperation>(controls, target0, target1,              \
                                      opTypeFromString(#op),                   \
                                      std::vector{param0, param1});            \
    }                                                                          \
  }

DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_minus_yy, theta, beta)
DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_plus_yy, theta, beta)

#undef DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION

void QuantumComputation::measure(const Qubit qubit, const std::size_t bit) {
  checkQubitRange(qubit);
  checkBitRange(bit);
  emplace_back<NonUnitaryOperation>(qubit, bit);
}

void QuantumComputation::measure(const Targets& qubits,
                                 const std::vector<Bit>& bits) {
  checkQubitRange(qubits);
  checkBitRange(bits);
  emplace_back<NonUnitaryOperation>(qubits, bits);
}

void QuantumComputation::measureAll(const bool addBits) {
  if (addBits) {
    addClassicalRegister(getNqubits(), "meas");
  }

  if (nclassics < getNqubits()) {
    std::stringstream ss{};
    ss << "The number of classical bits (" << nclassics
       << ") is smaller than the number of qubits (" << getNqubits() << ")!";
    throw std::runtime_error(ss.str());
  }

  barrier();
  Qubit start = 0U;
  if (addBits) {
    start = static_cast<Qubit>(classicalRegisters.at("meas").getStartIndex());
  }
  // measure i -> (start+i) in descending order
  // (this is an optimization for the simulator)
  for (std::size_t i = getNqubits(); i > 0; --i) {
    const auto q = static_cast<Qubit>(i - 1);
    measure(q, start + q);
  }
}

void QuantumComputation::reset(const Qubit target) {
  checkQubitRange(target);
  emplace_back<NonUnitaryOperation>(std::vector<Qubit>{target}, Reset);
}
void QuantumComputation::reset(const Targets& targets) {
  checkQubitRange(targets);
  emplace_back<NonUnitaryOperation>(targets, Reset);
}
void QuantumComputation::barrier() {
  std::vector<Qubit> targets(getNqubits());
  std::iota(targets.begin(), targets.end(), 0);
  emplace_back<StandardOperation>(targets, Barrier);
}
void QuantumComputation::barrier(const Qubit target) {
  checkQubitRange(target);
  emplace_back<StandardOperation>(target, Barrier);
}
void QuantumComputation::barrier(const Targets& targets) {
  checkQubitRange(targets);
  emplace_back<StandardOperation>(targets, Barrier);
}
void QuantumComputation::classicControlled(
    const OpType op, const Qubit target,
    const ClassicalRegister& controlRegister, const std::uint64_t expectedValue,
    const ComparisonKind cmp, const std::vector<fp>& params) {
  classicControlled(op, target, Controls{}, controlRegister, expectedValue, cmp,
                    params);
}
void QuantumComputation::classicControlled(
    const OpType op, const Qubit target, const Control control,
    const ClassicalRegister& controlRegister, const std::uint64_t expectedValue,
    const ComparisonKind cmp, const std::vector<fp>& params) {
  classicControlled(op, target, Controls{control}, controlRegister,
                    expectedValue, cmp, params);
}
void QuantumComputation::classicControlled(
    const OpType op, const Qubit target, const Controls& controls,
    const ClassicalRegister& controlRegister, const std::uint64_t expectedValue,
    const ComparisonKind cmp, const std::vector<fp>& params) {
  checkQubitRange(target, controls);
  checkClassicalRegister(controlRegister);
  std::unique_ptr<Operation> gate =
      std::make_unique<StandardOperation>(controls, target, op, params);
  emplace_back<ClassicControlledOperation>(std::move(gate), controlRegister,
                                           expectedValue, cmp);
}
void QuantumComputation::classicControlled(const OpType op, const Qubit target,
                                           const Bit cBit,
                                           const std::uint64_t expectedValue,
                                           const ComparisonKind cmp,
                                           const std::vector<fp>& params) {
  classicControlled(op, target, Controls{}, cBit, expectedValue, cmp, params);
}
void QuantumComputation::classicControlled(const OpType op, const Qubit target,
                                           const Control control,
                                           const Bit cBit,
                                           const std::uint64_t expectedValue,
                                           const ComparisonKind cmp,
                                           const std::vector<fp>& params) {
  classicControlled(op, target, Controls{control}, cBit, expectedValue, cmp,
                    params);
}
void QuantumComputation::classicControlled(const OpType op, const Qubit target,
                                           const Controls& controls,
                                           const Bit cBit,
                                           const std::uint64_t expectedValue,
                                           const ComparisonKind cmp,
                                           const std::vector<fp>& params) {
  checkQubitRange(target, controls);
  checkClassicalRegister({1, cBit});
  std::unique_ptr<Operation> gate =
      std::make_unique<StandardOperation>(controls, target, op, params);
  emplace_back<ClassicControlledOperation>(std::move(gate), cBit, expectedValue,
                                           cmp);
}
} // namespace qc
