/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/StandardOperation.hpp"

#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
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
    lambda = 0.0;
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
      phi = 0.0;
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
    phi = 0.0;
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
    return (controls.size() <= 1);
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
  case RCCX:
  case Barrier:
    break;
  // gates where we just update parameters
  case GPhase:
  case P:
  case RX:
  case RY:
  case RZ:
  case R:
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

} // namespace qc
