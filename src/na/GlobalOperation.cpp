//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#include "na/GlobalOperation.hpp"

#include "operations/CompoundOperation.hpp"

namespace na {

GlobalOperation::GlobalOperation(
    const qc::OpType& ot, std::size_t nctrl, const std::size_t nq,
    std::vector<std::unique_ptr<qc::Operation>>&& operations)
    : GlobalOperation(ot, nctrl, nq) {
  std::set<qc::Qubit> qubits;
  for (auto& op : operations) {
    if (op->getType() != opsType || op->getNcontrols() != nControls ||
        op->getNtargets() != nTargets || op->getParameter() != parameter) {
      throw std::invalid_argument(
          "Not all operations are of the same type as the global operation.");
    }
    // check whether the intersection of qubits and op->getUsedQubits() is not
    // empty
    if (!qubits.empty()) {
      std::vector<qc::Qubit> intersection;
      std::set_intersection(
          qubits.cbegin(), qubits.cend(), op->getUsedQubits().cbegin(),
          op->getUsedQubits().cend(), std::back_inserter(intersection));
      if (!intersection.empty()) {
        throw std::invalid_argument(
            "Multiple operations act on the same qubit.");
      }
    }
    // insert used qubits into qubits
    qubits.merge(op->getUsedQubits());
  }
  ops = std::move(operations);
}

void na::GlobalOperation::invert() {
  switch (opsType) {
  // self-inverting gates
  case qc::I:
  case qc::X:
  case qc::Y:
  case qc::Z:
  case qc::H:
  case qc::SWAP:
  case qc::ECR:
  case qc::Barrier:
  // gates where we just update parameters
  case qc::GPhase:
  case qc::P:
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
  case qc::U2:
  case qc::U:
  case qc::XXminusYY:
  case qc::XXplusYY:
  case qc::DCX:
    break;
  // gates where we have specialized inverted operation types
  case qc::S:
    opsType = qc::Sdg;
    break;
  case qc::Sdg:
    opsType = qc::S;
    break;
  case qc::T:
    type = qc::Tdg;
    break;
  case qc::Tdg:
    opsType = qc::T;
    break;
  case qc::V:
    opsType = qc::Vdg;
    break;
  case qc::Vdg:
    opsType = qc::V;
    break;
  case qc::SX:
    opsType = qc::SXdg;
    break;
  case qc::SXdg:
    opsType = qc::SX;
    break;
  case qc::Peres:
    opsType = qc::Peresdg;
    break;
  case qc::Peresdg:
    opsType = qc::Peres;
    break;
  case qc::iSWAP:
    opsType = qc::iSWAPdg;
    break;
  case qc::iSWAPdg:
    opsType = qc::iSWAP;
    break;
  default:
    throw std::runtime_error("Inverting gate" + toString(type) +
                             " is not supported.");
  }
  qc::CompoundOperation::invert();
}

} // namespace na
