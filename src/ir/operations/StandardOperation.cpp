/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/StandardOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qc {
/***
 * Protected Methods
 ***/
OpType StandardOperation::parseU3(fp& theta, fp& phi, fp& lambda) {
  if (std::abs(theta) < PARAMETER_TOLERANCE &&
      std::abs(phi) < PARAMETER_TOLERANCE) {
    parameter = {lambda};
    return parseU1(parameter[0]);
  }

  if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
    parameter = {phi, lambda};
    return parseU2(parameter[0], parameter[1]);
  }

  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    lambda = 0.L;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      checkInteger(theta);
      checkFractionPi(theta);
      parameter = {theta};
      return RY;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
      checkInteger(theta);
      checkFractionPi(theta);
      parameter = {theta};
      return RX;
    }

    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
      if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
        parameter.clear();
        return Y;
      }
    }
  }

  if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
    lambda = -PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
      parameter = {-theta};
      return RX;
    }
  }

  if (std::abs(lambda - PI) < PARAMETER_TOLERANCE) {
    lambda = PI;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      phi = 0.L;
      if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
        parameter.clear();
        return X;
      }
    }
  }

  // parse a real u3 gate
  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);
  checkInteger(theta);
  checkFractionPi(theta);

  return U;
}

OpType StandardOperation::parseU2(fp& phi, fp& lambda) {
  if (std::abs(phi) < PARAMETER_TOLERANCE) {
    phi = 0.L;
    if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return H;
    }
    if (std::abs(lambda) < PARAMETER_TOLERANCE) {
      parameter = {PI_2};
      return RY;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return V;
    }
  }

  if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
    lambda = -PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return Vdg;
    }
  }

  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);

  return U2;
}

OpType StandardOperation::parseU1(fp& lambda) {
  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return I;
  }
  const bool sign = std::signbit(lambda);

  if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return Z;
  }

  if (std::abs(std::abs(lambda) - PI_2) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return sign ? Sdg : S;
  }

  if (std::abs(std::abs(lambda) - PI_4) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return sign ? Tdg : T;
  }

  checkInteger(lambda);
  checkFractionPi(lambda);

  return P;
}

void StandardOperation::checkUgate() {
  if (parameter.empty()) {
    return;
  }
  if (type == P) {
    assert(parameter.size() == 1);
    type = parseU1(parameter.at(0));
  } else if (type == U2) {
    assert(parameter.size() == 2);
    type = parseU2(parameter.at(0), parameter.at(1));
  } else if (type == U) {
    assert(parameter.size() == 3);
    type = parseU3(parameter.at(0), parameter.at(1), parameter.at(2));
  }
}

void StandardOperation::setup() {
  checkUgate();
  name = toString(type);
}

/***
 * Constructors
 ***/
StandardOperation::StandardOperation(const Qubit target, const OpType g,
                                     std::vector<fp> params) {
  type = g;
  parameter = std::move(params);
  setup();
  targets.emplace_back(target);
}

StandardOperation::StandardOperation(const Targets& targ, const OpType g,
                                     std::vector<fp> params) {
  type = g;
  parameter = std::move(params);
  setup();
  targets = targ;
}

StandardOperation::StandardOperation(const Control control, const Qubit target,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(target, g, params) {
  StandardOperation::addControl(control);
}

StandardOperation::StandardOperation(const Control control, const Targets& targ,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(targ, g, params) {
  StandardOperation::addControl(control);
}

StandardOperation::StandardOperation(const Controls& c, const Qubit target,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(target, g, params) {
  addControls(c);
}

StandardOperation::StandardOperation(const Controls& c, const Targets& targ,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(targ, g, params) {
  addControls(c);
}

// MCF (cSWAP), Peres, parameterized two target Constructor
StandardOperation::StandardOperation(const Controls& c, const Qubit target0,
                                     const Qubit target1, const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(c, {target0, target1}, g, params) {}

bool StandardOperation::isGlobal(const size_t nQubits) const {
  return getUsedQubits().size() == nQubits;
}

/***
 * Public Methods
 ***/
bool StandardOperation::isClifford() const {
  switch (type) {
  case I:
    return true;
  case X:
  case Y:
  case Z:
    return (getControls().size() <= 1);
  case H:
  case S:
  case Sdg:
  case SX:
  case SXdg:
  case DCX:
  case SWAP:
  case iSWAP:
  case ECR:
    return !isControlled();
  default:
    return false;
  }
}

void StandardOperation::dumpOpenQASM(
    std::ostream& of, const QubitIndexToRegisterMap& qubitMap,
    [[maybe_unused]] const BitIndexToRegisterMap& bitMap, size_t indent,
    bool openQASM3) const {
  std::ostringstream op;
  op << std::setprecision(std::numeric_limits<fp>::digits10);

  op << std::string(indent * OUTPUT_INDENT_SIZE, ' ');

  if (openQASM3) {
    dumpOpenQASM3(of, op, qubitMap);
  } else {
    dumpOpenQASM2(of, op, qubitMap);
  }
}

void StandardOperation::dumpOpenQASM2(
    std::ostream& of, std::ostringstream& op,
    const QubitIndexToRegisterMap& qubitMap) const {
  if ((controls.size() > 1 && type != X) || controls.size() > 2) {
    std::cout << "[WARNING] Multiple controlled gates are not natively "
                 "supported by OpenQASM. "
              << "However, this library can parse .qasm files with multiple "
                 "controlled gates (e.g., cccx) correctly. "
              << "Thus, while not valid vanilla OpenQASM, the dumped file will "
                 "work with this library.\n";
  }

  // safe the numbers of controls as a prefix to the operation name
  op << std::string(controls.size(), 'c');

  const bool isSpecialGate = type == Peres || type == Peresdg;

  if (!isSpecialGate) {
    // apply X operations to negate the respective controls
    for (const auto& c : controls) {
      if (c.type == Control::Type::Neg) {
        of << "x " << qubitMap.at(c.qubit).second << ";\n";
      }
    }
  }

  dumpGateType(of, op, qubitMap);

  if (!isSpecialGate) {
    // apply X operations to negate the respective controls again
    for (const auto& c : controls) {
      if (c.type == Control::Type::Neg) {
        of << "x " << qubitMap.at(c.qubit).second << ";\n";
      }
    }
  }
}

void StandardOperation::dumpOpenQASM3(
    std::ostream& of, std::ostringstream& op,
    const QubitIndexToRegisterMap& qubitMap) const {
  dumpControls(op);

  dumpGateType(of, op, qubitMap);
}

void StandardOperation::dumpGateType(
    std::ostream& of, std::ostringstream& op,
    const QubitIndexToRegisterMap& qubitMap) const {
  // Dump the operation name and parameters.
  switch (type) {
  case GPhase:
    op << "gphase(" << parameter.at(0) << ")";
    break;
  case I:
    op << "id";
    break;
  case Barrier:
    assert(controls.empty());
    op << "barrier";
    break;
  case H:
    op << "h";
    break;
  case X:
    op << "x";
    break;
  case Y:
    op << "y";
    break;
  case Z:
    op << "z";
    break;
  case S:
    if (!controls.empty()) {
      op << "p(pi/2)";
    } else {
      op << "s";
    }
    break;
  case Sdg:
    if (!controls.empty()) {
      op << "p(-pi/2)";
    } else {
      op << "sdg";
    }
    break;
  case T:
    if (!controls.empty()) {
      op << "p(pi/4)";
    } else {
      op << "t";
    }
    break;
  case Tdg:
    if (!controls.empty()) {
      op << "p(-pi/4)";
    } else {
      op << "tdg";
    }
    break;
  case V:
    op << "U(pi/2,-pi/2,pi/2)";
    break;
  case Vdg:
    op << "U(pi/2,pi/2,-pi/2)";
    break;
  case U:
    op << "U(" << parameter[0] << "," << parameter[1] << "," << parameter[2]
       << ")";
    break;
  case U2:
    op << "U(pi/2," << parameter[0] << "," << parameter[1] << ")";
    break;
  case P:
    op << "p(" << parameter[0] << ")";
    break;
  case SX:
    op << "sx";
    break;
  case SXdg:
    op << "sxdg";
    break;
  case RX:
    op << "rx(" << parameter[0] << ")";
    break;
  case RY:
    op << "ry(" << parameter[0] << ")";
    break;
  case RZ:
    op << "rz(" << parameter[0] << ")";
    break;
  case DCX:
    op << "dcx";
    break;
  case ECR:
    op << "ecr";
    break;
  case RXX:
    op << "rxx(" << parameter[0] << ")";
    break;
  case RYY:
    op << "ryy(" << parameter[0] << ")";
    break;
  case RZZ:
    op << "rzz(" << parameter[0] << ")";
    break;
  case RZX:
    op << "rzx(" << parameter[0] << ")";
    break;
  case XXminusYY:
    op << "xx_minus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case XXplusYY:
    op << "xx_plus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case SWAP:
    op << "swap";
    break;
  case iSWAP:
    op << "iswap";
    break;
  case iSWAPdg:
    op << "iswapdg";
    break;
  case Move:
    op << "move";
    break;
  case Peres:
    of << op.str() << "cx";
    for (const auto& c : controls) {
      of << " " << qubitMap.at(c.qubit).second << ",";
    }
    of << " " << qubitMap.at(targets[1]).second << ", "
       << qubitMap.at(targets[0]).second << ";\n";

    of << op.str() << "x";
    for (const auto& c : controls) {
      of << " " << qubitMap.at(c.qubit).second << ",";
    }
    of << " " << qubitMap.at(targets[1]).second << ";\n";
    return;
  case Peresdg:
    of << op.str() << "x";
    for (const auto& c : controls) {
      of << " " << qubitMap.at(c.qubit).second << ",";
    }
    of << " " << qubitMap.at(targets[1]).second << ";\n";

    of << op.str() << "cx";
    for (const auto& c : controls) {
      of << " " << qubitMap.at(c.qubit).second << ",";
    }
    of << " " << qubitMap.at(targets[1]).second << ", "
       << qubitMap.at(targets[0]).second << ";\n";
    return;
  default:
    std::cerr << "gate type " << toString(type)
              << " could not be converted to OpenQASM\n.";
  }

  // apply the operation
  of << op.str();

  // First print control qubits.
  for (auto it = controls.begin(); it != controls.end();) {
    of << " " << qubitMap.at(it->qubit).second;
    // we only print a comma if there are more controls or targets.
    if (++it != controls.end() || !targets.empty()) {
      of << ",";
    }
  }
  // Print target qubits.
  if (!targets.empty() && type == Barrier &&
      isWholeQubitRegister(qubitMap, targets.front(), targets.back())) {
    of << " " << qubitMap.at(targets.front()).first.getName();
  } else {
    for (auto it = targets.begin(); it != targets.end();) {
      of << " " << qubitMap.at(*it).second;
      // only print comma if there are more targets
      if (++it != targets.end()) {
        of << ",";
      }
    }
  }
  of << ";\n";
}

auto StandardOperation::commutesAtQubit(const Operation& other,
                                        const Qubit& qubit) const -> bool {
  if (other.isCompoundOperation()) {
    return other.commutesAtQubit(*this, qubit);
  }
  // check whether both operations act on the given qubit
  if (!actsOn(qubit) || !other.actsOn(qubit)) {
    return true;
  }
  if (controls.contains(qubit)) {
    // if this is controlled on the given qubit
    if (const auto& controls2 = other.getControls();
        controls2.contains(qubit)) {
      // if other is controlled on the given qubit
      // q: ──■────■──
      //      |    |
      return true;
    }
    // here: qubit is a target of other
    return other.isDiagonalGate();
    // true, iff qubit is a target and other is a diagonal gate, e.g., rz
    //         ┌────┐
    // q: ──■──┤ RZ ├
    //      |  └────┘
  }
  // here: qubit is a target of this
  if (const auto& controls2 = other.getControls(); controls2.contains(qubit)) {
    return isDiagonalGate();
    // true, iff qubit is a target and this is a diagonal gate and other is
    // controlled, e.g.
    //    ┌────┐
    // q: ┤ RZ ├──■──
    //    └────┘  |
  }
  // here: qubit is a target of both operations
  if (isDiagonalGate() && other.isDiagonalGate()) {
    // if both operations are diagonal gates, e.g.
    //    ┌────┐┌────┐
    // q: ┤ RZ ├┤ RZ ├
    //    └────┘└────┘
    return true;
  }
  if (parameter.size() <= 1) {
    return type == other.getType() && targets == other.getTargets();
    // true, iff both operations are of the same type, e.g.
    //    ┌───┐┌───┐
    // q: ┤ E ├┤ E ├
    //    | C || C |
    //    ┤ R ├┤ R ├
    //    └───┘└───┘
    //      |    |
    //    ──■────┼──
    //           |
    //    ───────■──
  }
  // operations with more than one parameter might not be commutative when the
  // parameter are not the same, i.e. a general U3 gate
  // TODO: this check might introduce false negatives
  return type == other.getType() && targets == other.getTargets() &&
         parameter == other.getParameter();
}

void StandardOperation::invert() {
  switch (type) {
  // self-inverting gates
  case I:
  case X:
  case Y:
  case Z:
  case H:
  case SWAP:
  case ECR:
  case Barrier:
    break;
  // gates where we just update parameters
  case GPhase:
  case P:
  case RX:
  case RY:
  case RZ:
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
    parameter[0] = -parameter[0];
    break;
  case U2:
    std::swap(parameter[0], parameter[1]);
    parameter[0] = -parameter[0] + PI;
    parameter[1] = -parameter[1] - PI;
    break;
  case U:
    parameter[0] = -parameter[0];
    parameter[1] = -parameter[1];
    parameter[2] = -parameter[2];
    std::swap(parameter[1], parameter[2]);
    break;
  case XXminusYY:
  case XXplusYY:
    parameter[0] = -parameter[0];
    break;
  case DCX:
    std::swap(targets[0], targets[1]);
    break;
  // gates where we have specialized inverted operation types
  case S:
    type = Sdg;
    break;
  case Sdg:
    type = S;
    break;
  case T:
    type = Tdg;
    break;
  case Tdg:
    type = T;
    break;
  case V:
    type = Vdg;
    break;
  case Vdg:
    type = V;
    break;
  case SX:
    type = SXdg;
    break;
  case SXdg:
    type = SX;
    break;
  case Peres:
    type = Peresdg;
    break;
  case Peresdg:
    type = Peres;
    break;
  case iSWAP:
    type = iSWAPdg;
    break;
  case iSWAPdg:
    type = iSWAP;
    break;
  default:
    throw std::runtime_error("Inverting gate" + toString(type) +
                             " is not supported.");
  }
}

void StandardOperation::dumpControls(std::ostringstream& op) const {
  if (controls.empty()) {
    return;
  }

  // if operation is in stdgates.inc, we print a c prefix instead of ctrl @
  if (bool printBuiltin = std::ranges::none_of(
          controls,
          [](const Control& c) { return c.type == Control::Type::Neg; });
      printBuiltin) {
    const auto numControls = controls.size();
    switch (type) {
    case P:
    case RX:
    case Y:
    case RY:
    case Z:
    case RZ:
    case H:
    case SWAP:
      printBuiltin = numControls == 1;
      break;
    case X:
      printBuiltin = numControls == 1 || numControls == 2;
      break;
    default:
      printBuiltin = false;
    }
    if (printBuiltin) {
      op << std::string(numControls, 'c');
      return;
    }
  }

  Control::Type currentType = controls.begin()->type;
  int count = 0;

  for (const auto& control : controls) {
    if (control.type == currentType) {
      ++count;
    } else {
      op << (currentType == Control::Type::Neg ? "negctrl" : "ctrl");
      if (count > 1) {
        op << "(" << count << ")";
      }
      op << " @ ";
      currentType = control.type;
      count = 1;
    }
  }

  op << (currentType == Control::Type::Neg ? "negctrl" : "ctrl");
  if (count > 1) {
    op << "(" << count << ")";
  }
  op << " @ ";
}
} // namespace qc
