/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "circuit_optimizer/CircuitOptimizer.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qc {
namespace {
void addToDag(CircuitOptimizer::DAG& dag, std::unique_ptr<Operation>* op) {
  const auto usedQubits = (*op)->getUsedQubits();
  for (const auto q : usedQubits) {
    dag.at(q).push_back(op);
  }
}
} // namespace

void CircuitOptimizer::removeIdentities(QuantumComputation& qc) {
  // delete the identities from circuit
  removeOperation(qc, {I}, 0);
}

void CircuitOptimizer::removeOperation(
    QuantumComputation& qc, const std::unordered_set<OpType>& opTypes,
    const size_t opSize) {
  // opSize = 0 means that the operation can have any number of qubits
  auto it = qc.begin();
  while (it != qc.end()) {
    if (opTypes.find((*it)->getType()) != opTypes.end() &&
        (opSize == 0 || it->get()->getNqubits() == opSize)) {
      it = qc.erase(it);
    } else if ((*it)->isCompoundOperation()) {
      auto& compOp = dynamic_cast<CompoundOperation&>(**it);
      auto cit = compOp.cbegin();
      while (cit != compOp.cend()) {
        if (const auto* cop = cit->get();
            opTypes.find(cop->getType()) != opTypes.end() &&
            (opSize == 0 || cop->getNqubits() == opSize)) {
          cit = compOp.erase(cit);
        } else {
          ++cit;
        }
      }
      if (compOp.empty()) {
        it = qc.erase(it);
      } else {
        if (compOp.size() == 1) {
          // CompoundOperation has degraded to single Operation
          (*it) = std::move(*(compOp.begin()));
        }
        ++it;
      }
    } else {
      ++it;
    }
  }
}

void CircuitOptimizer::swapReconstruction(QuantumComputation& qc) {
  auto dag = DAG(qc.getHighestPhysicalQubitIndex() + 1);

  for (auto& it : qc) {
    if (!it->isStandardOperation()) {
      addToDag(dag, &it);
      continue;
    }

    // Operation is not a CNOT
    if (it->getType() != X || it->getNcontrols() != 1 ||
        it->getControls().begin()->type != Control::Type::Pos) {
      addToDag(dag, &it);
      continue;
    }

    const Qubit control = it->getControls().begin()->qubit;
    const Qubit target = it->getTargets().at(0);

    // first operation
    if (dag.at(control).empty() || dag.at(target).empty()) {
      addToDag(dag, &it);
      continue;
    }

    auto* opC = dag.at(control).back();
    auto* opT = dag.at(target).back();

    // previous operation is not a CNOT
    if ((*opC)->getType() != X || (*opC)->getNcontrols() != 1 ||
        (*opC)->getControls().begin()->type != Control::Type::Pos ||
        (*opT)->getType() != X || (*opT)->getNcontrols() != 1 ||
        (*opT)->getControls().begin()->type != Control::Type::Pos) {
      addToDag(dag, &it);
      continue;
    }

    const auto opControl = (*opC)->getControls().begin()->qubit;
    const auto opCtarget = (*opC)->getTargets().at(0);
    const auto opTcontrol = (*opT)->getControls().begin()->qubit;
    const auto opTtarget = (*opT)->getTargets().at(0);

    // operation at control and target qubit are not the same
    if (opControl != opTcontrol || opCtarget != opTtarget) {
      addToDag(dag, &it);
      continue;
    }

    if (control == opControl && target == opCtarget) {
      // elimination
      dag.at(control).pop_back();
      dag.at(target).pop_back();
      (*opC)->setGate(I);
      (*opC)->clearControls();
      it->setGate(I);
      it->clearControls();
    } else if (control == opCtarget && target == opControl) {
      dag.at(control).pop_back();
      dag.at(target).pop_back();

      // replace with SWAP + CNOT
      (*opC)->setGate(SWAP);
      if (target > control) {
        (*opC)->setTargets({control, target});
      } else {
        (*opC)->setTargets({target, control});
      }
      (*opC)->clearControls();
      addToDag(dag, opC);

      it->setTargets({control});
      it->setControls({Control{target}});
      addToDag(dag, &it);
    } else {
      addToDag(dag, &it);
    }
  }

  removeIdentities(qc);
}

CircuitOptimizer::DAG CircuitOptimizer::constructDAG(QuantumComputation& qc) {
  auto dag = DAG(qc.getHighestPhysicalQubitIndex() + 1);

  for (auto& op : qc) {
    addToDag(dag, &op);
  }
  return dag;
}

void CircuitOptimizer::singleQubitGateFusion(QuantumComputation& qc) {
  static const std::map<OpType, OpType> INVERSE_MAP = {
      {I, I},   {X, X},   {Y, Y},   {Z, Z},     {H, H},     {S, Sdg},
      {Sdg, S}, {T, Tdg}, {Tdg, T}, {SX, SXdg}, {SXdg, SX}, {Barrier, Barrier}};

  auto dag = DAG(qc.getHighestPhysicalQubitIndex() + 1);

  for (auto& it : qc) {
    // not a single-qubit operation
    if (!it->isStandardOperation() || !it->getControls().empty() ||
        it->getTargets().size() > 1) {
      addToDag(dag, &it);
      continue;
    }

    const auto target = it->getTargets().at(0);

    // first operation
    if (dag.at(target).empty()) {
      addToDag(dag, &it);
      continue;
    }

    auto dagQubit = dag.at(target);
    auto* op = dagQubit.back();

    // no single qubit op to fuse with operation
    if (!(*op)->isCompoundOperation() &&
        (!(*op)->getControls().empty() || (*op)->getTargets().size() > 1)) {
      addToDag(dag, &it);
      continue;
    }

    // compound operation
    if ((*op)->isCompoundOperation()) {
      auto* compop = dynamic_cast<CompoundOperation*>(op->get());

      // check if compound operation contains non-single-qubit gates
      std::size_t involvedQubits = 0;
      for (std::size_t q = 0; q < dag.size(); ++q) {
        if (compop->actsOn(static_cast<Qubit>(q))) {
          ++involvedQubits;
        }
      }
      if (involvedQubits > 1) {
        addToDag(dag, &it);
        continue;
      }

      // check if the compound operation is empty (e.g., -X-H-H-X-Z-)
      if (compop->empty()) {
        compop->emplace_back(it->clone());
        it->setGate(I);
        continue;
      }

      // check if inverse
      auto lastop = (--(compop->end()));
      auto inverseIt = INVERSE_MAP.find((*lastop)->getType());
      // check if current operation is the inverse of the previous operation
      if (inverseIt != INVERSE_MAP.end() &&
          it->getType() == inverseIt->second) {
        compop->pop_back();
        it->setGate(I);
      } else {
        compop->emplace_back<StandardOperation>(
            it->getTargets().at(0), it->getType(), it->getParameter());
        it->setGate(I);
      }

      continue;
    }

    // single qubit op

    // check if current operation is the inverse of the previous operation
    auto inverseIt = INVERSE_MAP.find((*op)->getType());
    if (inverseIt != INVERSE_MAP.end() && it->getType() == inverseIt->second) {
      (*op)->setGate(I);
      it->setGate(I);
    } else {
      auto compop = std::make_unique<CompoundOperation>();
      compop->emplace_back<StandardOperation>(
          (*op)->getTargets().at(0), (*op)->getType(), (*op)->getParameter());
      compop->emplace_back<StandardOperation>(
          it->getTargets().at(0), it->getType(), it->getParameter());
      it->setGate(I);
      (*op) = std::move(compop);
      dag.at(target).push_back(op);
    }
  }

  removeIdentities(qc);
}

namespace {
bool removeDiagonalGate(CircuitOptimizer::DAG& dag,
                        CircuitOptimizer::DAGReverseIterators& dagIterators,
                        Qubit idx, CircuitOptimizer::DAGReverseIterator& it,
                        Operation* op);

void removeDiagonalGatesBeforeMeasureRecursive(
    CircuitOptimizer::DAG& dag,
    CircuitOptimizer::DAGReverseIterators& dagIterators, Qubit idx,
    const Operation* until) {
  // qubit is finished -> consider next qubit
  if (dagIterators.at(idx) == dag.at(idx).rend()) {
    if (idx < static_cast<Qubit>(dag.size() - 1)) {
      removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, idx + 1,
                                                nullptr);
    }
    return;
  }
  // check if desired operation was reached
  if (until != nullptr) {
    if ((*dagIterators.at(idx))->get() == until) {
      return;
    }
  }

  auto& it = dagIterators.at(idx);
  while (it != dag.at(idx).rend()) {
    // check if desired operation was reached
    if (until != nullptr) {
      if ((*dagIterators.at(idx))->get() == until) {
        break;
      }
    }
    auto* op = (*it)->get();
    if (op->isStandardOperation()) {
      // try removing gate and upon success increase all corresponding iterators
      auto onlyDiagonalGates =
          removeDiagonalGate(dag, dagIterators, idx, it, op);
      if (onlyDiagonalGates) {
        for (const auto& control : op->getControls()) {
          ++(dagIterators.at(control.qubit));
        }
        for (const auto& target : op->getTargets()) {
          ++(dagIterators.at(target));
        }
      }

    } else if (op->isCompoundOperation()) {
      // iterate over all gates of compound operation and upon success increase
      // all corresponding iterators
      auto* compOp = dynamic_cast<CompoundOperation*>(op);
      bool onlyDiagonalGates = true;
      auto cit = compOp->rbegin();
      while (cit != compOp->rend()) {
        auto* cop = cit->get();
        onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
        if (!onlyDiagonalGates) {
          break;
        }
        ++cit;
      }
      if (onlyDiagonalGates) {
        for (size_t q = 0; q < dag.size(); ++q) {
          if (compOp->actsOn(static_cast<Qubit>(q))) {
            ++(dagIterators.at(q));
          }
        }
      }
    } else if (op->isClassicControlledOperation()) {
      // consider the operation that is classically controlled and proceed as
      // above
      auto* cop = dynamic_cast<ClassicControlledOperation*>(op)->getOperation();
      const bool onlyDiagonalGates =
          removeDiagonalGate(dag, dagIterators, idx, it, cop);
      if (onlyDiagonalGates) {
        for (const auto& control : cop->getControls()) {
          ++(dagIterators.at(control.qubit));
        }
        for (const auto& target : cop->getTargets()) {
          ++(dagIterators.at(target));
        }
      }
    } else if (op->isNonUnitaryOperation()) {
      // non-unitary operation is not diagonal
      it = dag.at(idx).rend();
    } else {
      throw std::runtime_error("Unexpected operation encountered");
    }
  }

  // qubit is finished -> consider next qubit
  if (dagIterators.at(idx) == dag.at(idx).rend() &&
      idx < static_cast<Qubit>(dag.size() - 1)) {
    removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, idx + 1,
                                              nullptr);
  }
}

bool removeDiagonalGate(CircuitOptimizer::DAG& dag,
                        CircuitOptimizer::DAGReverseIterators& dagIterators,
                        Qubit idx, CircuitOptimizer::DAGReverseIterator& it,
                        Operation* op) {
  // not a diagonal gate
  if (!op->isDiagonalGate()) {
    it = dag.at(idx).rend();
    return false;
  }

  if (op->getNcontrols() != 0) {
    // need to check all controls and targets
    bool onlyDiagonalGates = true;
    for (const auto& control : op->getControls()) {
      auto controlQubit = control.qubit;
      if (controlQubit == idx) {
        continue;
      }
      if (control.type == Control::Type::Neg) {
        dagIterators.at(controlQubit) = dag.at(controlQubit).rend();
        onlyDiagonalGates = false;
        break;
      }
      if (dagIterators.at(controlQubit) == dag.at(controlQubit).rend()) {
        onlyDiagonalGates = false;
        break;
      }
      // recursive call at control with this operation as goal
      removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, controlQubit,
                                                (*it)->get());
      // check if iteration of control qubit was successful
      if (*dagIterators.at(controlQubit) != *it) {
        onlyDiagonalGates = false;
        break;
      }
    }
    for (const auto& target : op->getTargets()) {
      if (target == idx) {
        continue;
      }
      if (dagIterators.at(target) == dag.at(target).rend()) {
        onlyDiagonalGates = false;
        break;
      }
      // recursive call at target with this operation as goal
      removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, target,
                                                (*it)->get());
      // check if iteration of target qubit was successful
      if (*dagIterators.at(target) != *it) {
        onlyDiagonalGates = false;
        break;
      }
    }
    if (!onlyDiagonalGates) {
      // end qubit
      dagIterators.at(idx) = dag.at(idx).rend();
    } else {
      // set operation to identity so that it can be collected by the
      // removeIdentities pass
      op->setGate(I);
    }
    return onlyDiagonalGates;
  }
  // set operation to identity so that it can be collected by the
  // removeIdentities pass
  op->setGate(I);
  return true;
}
} // namespace

void CircuitOptimizer::removeDiagonalGatesBeforeMeasure(
    QuantumComputation& qc) {
  auto dag = constructDAG(qc);

  // initialize iterators
  DAGReverseIterators dagIterators{dag.size()};
  for (size_t q = 0; q < dag.size(); ++q) {
    if (dag.at(q).empty() || dag.at(q).back()->get()->getType() != Measure) {
      // qubit is not measured and thus does not have to be considered
      dagIterators.at(q) = dag.at(q).rend();
    } else {
      // point to operation before measurement
      dagIterators.at(q) = ++(dag.at(q).rbegin());
    }
  }
  // iterate over DAG in depth-first fashion
  removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, 0, nullptr);

  // remove resulting identities from circuit
  removeIdentities(qc);
}

namespace {
bool removeFinalMeasurement(CircuitOptimizer::DAG& dag,
                            CircuitOptimizer::DAGReverseIterators& dagIterators,
                            Qubit idx,
                            const CircuitOptimizer::DAGReverseIterator& it,
                            Operation* op);

void removeFinalMeasurementsRecursive(
    CircuitOptimizer::DAG& dag,
    CircuitOptimizer::DAGReverseIterators& dagIterators, Qubit idx,
    const Operation* until) {
  if (dagIterators.at(idx) == dag.at(idx).rend()) { // we reached the end
    if (idx < static_cast<Qubit>(dag.size() - 1)) {
      removeFinalMeasurementsRecursive(dag, dagIterators, idx + 1, nullptr);
    }
    return;
  }
  // check if desired operation was reached
  if (until != nullptr) {
    if ((*dagIterators.at(idx))->get() == until) {
      return;
    }
  }
  auto& it = dagIterators.at(idx);
  while (it != dag.at(idx).rend()) {
    if (until != nullptr) {
      if ((*dagIterators.at(idx))->get() == until) {
        break;
      }
    }
    auto* op = (*it)->get();
    if (op->getType() == Measure || op->getType() == Barrier) {
      const bool onlyMeasurement =
          removeFinalMeasurement(dag, dagIterators, idx, it, op);
      if (onlyMeasurement) {
        for (const auto& target : op->getTargets()) {
          if (dagIterators.at(target) == dag.at(target).rend()) {
            break;
          }
          ++(dagIterators.at(target));
        }
      }
    } else if (op->isCompoundOperation() && op->isNonUnitaryOperation()) {
      // iterate over all gates of compound operation and upon success increase
      // all corresponding iterators
      auto* compOp = dynamic_cast<CompoundOperation*>(op);
      bool onlyMeasurement = true;
      auto cit = compOp->rbegin();
      while (cit != compOp->rend()) {
        auto* cop = cit->get();
        if (cop->getNtargets() > 0 && cop->getTargets()[0] != idx) {
          ++cit;
          continue;
        }
        onlyMeasurement =
            removeFinalMeasurement(dag, dagIterators, idx, it, cop);
        if (!onlyMeasurement) {
          break;
        }
        ++cit;
      }
      if (onlyMeasurement) {
        ++(dagIterators.at(idx));
      }
    } else {
      // not a measurement, we are done
      dagIterators.at(idx) = dag.at(idx).rend();
      break;
    }
  }
  if (dagIterators.at(idx) == dag.at(idx).rend() &&
      idx < static_cast<Qubit>(dag.size() - 1)) {
    removeFinalMeasurementsRecursive(dag, dagIterators, idx + 1, nullptr);
  }
}

bool removeFinalMeasurement(CircuitOptimizer::DAG& dag,
                            CircuitOptimizer::DAGReverseIterators& dagIterators,
                            const Qubit idx,
                            const CircuitOptimizer::DAGReverseIterator& it,
                            Operation* op) {
  if (op->getNtargets() != 0) {
    // need to check all targets
    bool onlyMeasurements = true;
    for (const auto& target : op->getTargets()) {
      if (target == idx) {
        continue;
      }
      if (dagIterators.at(target) == dag.at(target).rend()) {
        onlyMeasurements = false;
        break;
      }
      // recursive call at target with this operation as goal
      removeFinalMeasurementsRecursive(dag, dagIterators, target, (*it)->get());
      // check if iteration of target qubit was successful
      if (dagIterators.at(target) == dag.at(target).rend() ||
          *dagIterators.at(target) != *it) {
        onlyMeasurements = false;
        break;
      }
    }
    if (!onlyMeasurements) {
      // end qubit
      dagIterators.at(idx) = dag.at(idx).rend();
    } else {
      // set operation to identity so that it can be collected by the
      // removeIdentities pass
      op->setGate(I);
    }
    return onlyMeasurements;
  }
  return false;
}
} // namespace

void CircuitOptimizer::removeFinalMeasurements(QuantumComputation& qc) {
  auto dag = constructDAG(qc);
  DAGReverseIterators dagIterators{dag.size()};
  for (size_t q = 0; q < dag.size(); ++q) {
    dagIterators.at(q) = (dag.at(q).rbegin());
  }

  removeFinalMeasurementsRecursive(dag, dagIterators, 0, nullptr);

  removeIdentities(qc);
}

void CircuitOptimizer::decomposeSWAP(QuantumComputation& qc,
                                     bool isDirectedArchitecture) {
  // decompose SWAPS in three cnot and optionally in four H
  auto it = qc.begin();
  while (it != qc.end()) {
    if ((*it)->isStandardOperation()) {
      if ((*it)->getType() == SWAP) {
        const auto targets = (*it)->getTargets();
        it = qc.erase(it);
        it = qc.insert(it, std::make_unique<StandardOperation>(
                               Control{targets[0]}, targets[1], X));
        if (isDirectedArchitecture) {
          it =
              qc.insert(it, std::make_unique<StandardOperation>(targets[0], H));
          it =
              qc.insert(it, std::make_unique<StandardOperation>(targets[1], H));
          it = qc.insert(it, std::make_unique<StandardOperation>(
                                 Control{targets[0]}, targets[1], X));
          it =
              qc.insert(it, std::make_unique<StandardOperation>(targets[0], H));
          it =
              qc.insert(it, std::make_unique<StandardOperation>(targets[1], H));
        } else {
          it = qc.insert(it, std::make_unique<StandardOperation>(
                                 Control{targets[1]}, targets[0], X));
        }
        it = qc.insert(it, std::make_unique<StandardOperation>(
                               Control{targets[0]}, targets[1], X));
      } else {
        ++it;
      }
    } else if ((*it)->isCompoundOperation()) {
      auto& compOp = dynamic_cast<CompoundOperation&>(**it);
      auto cit = compOp.begin();
      while (cit != compOp.end()) {
        if ((*cit)->isStandardOperation() && (*cit)->getType() == SWAP) {
          const auto targets = (*cit)->getTargets();
          cit = compOp.erase(cit);
          cit = compOp.insert<StandardOperation>(cit, Control{targets[0]},
                                                 targets[1], X);
          if (isDirectedArchitecture) {
            cit = compOp.insert<StandardOperation>(cit, targets[0], H);
            cit = compOp.insert<StandardOperation>(cit, targets[1], H);
            cit = compOp.insert<StandardOperation>(cit, Control{targets[0]},
                                                   targets[1], X);
            cit = compOp.insert<StandardOperation>(cit, targets[0], H);
            cit = compOp.insert<StandardOperation>(cit, targets[1], H);
          } else {
            cit = compOp.insert<StandardOperation>(cit, Control{targets[1]},
                                                   targets[0], X);
          }
          cit = compOp.insert<StandardOperation>(cit, Control{targets[0]},
                                                 targets[1], X);
        } else {
          ++cit;
        }
      }
      ++it;
    } else {
      ++it;
    }
  }
}

namespace {
void changeTargets(Targets& targets,
                   const std::map<Qubit, Qubit>& replacementMap) {
  for (auto& target : targets) {
    auto newTargetIt = replacementMap.find(target);
    if (newTargetIt != replacementMap.end()) {
      target = newTargetIt->second;
    }
  }
}

void changeControls(Controls& controls,
                    const std::map<Qubit, Qubit>& replacementMap) {
  if (controls.empty() || replacementMap.empty()) {
    return;
  }

  // iterate over the replacement map and see if any control matches
  for (const auto& [from, to] : replacementMap) {
    auto controlIt = controls.find(from);
    if (controlIt != controls.end()) {
      const auto controlType = controlIt->type;
      controls.erase(controlIt);
      controls.insert(Control{to, controlType});
    }
  }
}
} // namespace

void CircuitOptimizer::eliminateResets(QuantumComputation& qc) {
  //      ┌───┐┌─┐     ┌───┐┌─┐            ┌───┐┌─┐ ░
  // q_0: ┤ H ├┤M├─|0>─┤ H ├┤M├       q_0: ┤ H ├┤M├─░─────────
  //      └───┘└╥┘     └───┘└╥┘   -->      └───┘└╥┘ ░ ┌───┐┌─┐
  // c: 2/══════╩════════════╩═       q_1: ──────╫──░─┤ H ├┤M├
  //            0            1                   ║  ░ └───┘└╥┘
  //                                  c: 2/══════╩══════════╩═
  //                                             0          1
  auto replacementMap = std::map<Qubit, Qubit>();
  auto it = qc.begin();
  while (it != qc.end()) {
    if ((*it)->getType() == Reset) {
      for (const auto& target : (*it)->getTargets()) {
        auto indexAddQubit = static_cast<Qubit>(qc.getNqubits());
        qc.addQubit(indexAddQubit, indexAddQubit, indexAddQubit);
        auto oldReset = replacementMap.find(target);
        if (oldReset != replacementMap.end()) {
          oldReset->second = indexAddQubit;
        } else {
          replacementMap.try_emplace(target, indexAddQubit);
        }
      }
      it = qc.erase(it);
    } else if (!replacementMap.empty()) {
      if ((*it)->isCompoundOperation()) {
        auto& compOp = dynamic_cast<CompoundOperation&>(**it);
        auto compOpIt = compOp.begin();
        while (compOpIt != compOp.end()) {
          if ((*compOpIt)->getType() == Reset) {
            for (const auto& compTarget : (*compOpIt)->getTargets()) {
              auto indexAddQubit = static_cast<Qubit>(qc.getNqubits());
              qc.addQubit(indexAddQubit, indexAddQubit, indexAddQubit);
              if (auto oldReset = replacementMap.find(compTarget);
                  oldReset != replacementMap.end()) {
                oldReset->second = indexAddQubit;
              } else {
                replacementMap.try_emplace(compTarget, indexAddQubit);
              }
            }
            compOpIt = compOp.erase(compOpIt);
          } else {
            if ((*compOpIt)->isStandardOperation() ||
                (*compOpIt)->isClassicControlledOperation()) {
              auto& targets = (*compOpIt)->getTargets();
              auto& controls = (*compOpIt)->getControls();
              changeTargets(targets, replacementMap);
              changeControls(controls, replacementMap);
            } else if ((*compOpIt)->isNonUnitaryOperation()) {
              auto& targets = (*compOpIt)->getTargets();
              changeTargets(targets, replacementMap);
            }
            ++compOpIt;
          }
        }
      }
      if ((*it)->isStandardOperation() ||
          (*it)->isClassicControlledOperation()) {
        auto& targets = (*it)->getTargets();
        auto& controls = (*it)->getControls();
        changeTargets(targets, replacementMap);
        changeControls(controls, replacementMap);
      } else if ((*it)->isNonUnitaryOperation()) {
        auto& targets = (*it)->getTargets();
        changeTargets(targets, replacementMap);
      }
      ++it;
    } else {
      ++it;
    }
  }
}

void CircuitOptimizer::deferMeasurements(QuantumComputation& qc) {
  //      ┌───┐┌─┐                         ┌───┐     ┌─┐
  // q_0: ┤ H ├┤M├───────             q_0: ┤ H ├──■──┤M├
  //      └───┘└╥┘ ┌───┐                   └───┘┌─┴─┐└╥┘
  // q_1: ──────╫──┤ X ├─     -->     q_1: ─────┤ X ├─╫─
  //            ║  └─╥─┘                        └───┘ ║
  //            ║ ┌──╨──┐             c: 2/═══════════╩═
  // c: 2/══════╩═╡ = 1 ╞                             0
  //            0 └─────┘
  std::unordered_map<Qubit, std::size_t> qubitsToAddMeasurements{};
  auto it = qc.begin();
  while (it != qc.end()) {
    if (const auto* measurement = dynamic_cast<NonUnitaryOperation*>(it->get());
        measurement != nullptr && measurement->getType() == Measure) {
      const auto targets = measurement->getTargets();
      const auto classics = measurement->getClassics();

      if (targets.size() != 1 && classics.size() != 1) {
        throw std::runtime_error(
            "Deferring measurements with more than 1 target is not yet "
            "supported. Try decomposing your measurements.");
      }

      // if this is the last operation, nothing has to be done
      if (*it == qc.back()) {
        break;
      }

      const auto measurementQubit = targets[0];
      const auto measurementBit = classics[0];

      // remember q->c for adding measurements later
      qubitsToAddMeasurements[measurementQubit] = measurementBit;

      // remove the measurement from the vector of operations
      it = qc.erase(it);

      // starting from the next operation after the measurement (if there is
      // any)
      auto opIt = it;
      auto currentInsertionPoint = it;

      // iterate over all subsequent operations
      while (opIt != qc.end()) {
        const auto* operation = opIt->get();
        if (operation->isUnitary()) {
          // if an operation does not act on the measured qubit, the insert
          // location for potential operations has to be updated
          if (!operation->actsOn(measurementQubit)) {
            ++currentInsertionPoint;
          }
          ++opIt;
          continue;
        }

        if (operation->getType() == Reset) {
          throw std::runtime_error(
              "Reset encountered in deferMeasurements routine. Please use the "
              "eliminateResets method before deferring measurements.");
        }

        if (const auto* measurement2 =
                dynamic_cast<NonUnitaryOperation*>(opIt->get());
            measurement2 != nullptr && operation->getType() == Measure) {
          const auto& targets2 = measurement2->getTargets();
          const auto& classics2 = measurement2->getClassics();

          // if this is the same measurement a breakpoint has been reached
          if (targets == targets2 && classics == classics2) {
            break;
          }

          ++currentInsertionPoint;
          ++opIt;
          continue;
        }

        if (const auto* classicOp =
                dynamic_cast<ClassicControlledOperation*>(opIt->get());
            classicOp != nullptr) {
          const auto& expectedValue = classicOp->getExpectedValue();

          Bit cBit = 0;
          if (const auto& controlRegister = classicOp->getControlRegister();
              controlRegister.has_value()) {
            assert(!classicOp->getControlBit().has_value());
            if (controlRegister->getSize() != 1) {
              throw std::runtime_error(
                  "Classic-controlled operations targeted at more than one bit "
                  "are currently not supported. Try decomposing the operation "
                  "into individual contributions.");
            }
            cBit = controlRegister->getStartIndex();
          }
          if (const auto& controlBit = classicOp->getControlBit();
              controlBit.has_value()) {
            assert(!classicOp->getControlRegister().has_value());
            cBit = controlBit.value();
          }

          // if this is not the classical bit that is measured, continue
          if (cBit != measurementBit) {
            if (!operation->actsOn(measurementQubit)) {
              ++currentInsertionPoint;
            }
            ++opIt;
            continue;
          }

          // get the underlying operation
          const auto* standardOp =
              dynamic_cast<StandardOperation*>(classicOp->getOperation());
          if (standardOp == nullptr) {
            std::stringstream ss{};
            ss << "Underlying operation of classic-controlled operation is "
                  "not a StandardOperation.\n";
            classicOp->print(ss, qc.getNqubits());
            throw std::runtime_error(ss.str());
          }

          // get all the necessary information for reconstructing the
          // operation
          const auto type = standardOp->getType();
          const auto targs = standardOp->getTargets();
          for (const auto& target : targs) {
            if (target == measurementQubit) {
              throw std::runtime_error(
                  "Implicit reset operation in circuit detected. Measuring a "
                  "qubit and then targeting the same qubit with a "
                  "classic-controlled operation is not allowed at the "
                  "moment.");
            }
          }

          // determine the appropriate control to add
          auto controls = standardOp->getControls();
          const auto controlQubit = measurementQubit;
          const auto controlType =
              (expectedValue == 1) ? Control::Type::Pos : Control::Type::Neg;
          controls.emplace(controlQubit, controlType);

          const auto parameters = standardOp->getParameter();

          // remove the classic-controlled operation
          // carefully handle iterator invalidation.
          // if the current insertion point is the same as the current
          // iterator the insertion point has to be updated to the new
          // operation as well.
          auto itInvalidated = (it >= opIt);
          const auto insertionPointInvalidated =
              (currentInsertionPoint >= opIt);

          opIt = qc.erase(opIt);

          if (itInvalidated) {
            it = opIt;
          }
          if (insertionPointInvalidated) {
            currentInsertionPoint = opIt;
          }

          itInvalidated = (it >= currentInsertionPoint);
          // insert the new operation (invalidated all pointer onwards)
          currentInsertionPoint = qc.insert(
              currentInsertionPoint, std::make_unique<StandardOperation>(
                                         controls, targs, type, parameters));

          if (itInvalidated) {
            it = currentInsertionPoint;
          }
          // advance just after the currently inserted operation
          ++currentInsertionPoint;
          // the inner loop also has to restart from here due to the
          // invalidation of the iterators
          opIt = currentInsertionPoint;
        }
      }
    }
    ++it;
  }
  if (qubitsToAddMeasurements.empty()) {
    return;
  }
  qc.outputPermutation.clear();
  for (const auto& [qubit, clbit] : qubitsToAddMeasurements) {
    qc.measure(qubit, clbit);
  }
  qc.initializeIOMapping();
}

namespace {
using Iterator = QuantumComputation::iterator;
void flattenCompoundOperation(QuantumComputation& qc, Iterator& it) {
  assert((*it)->isCompoundOperation());
  auto& op = dynamic_cast<CompoundOperation&>(**it);
  auto opIt = op.begin();
  std::int64_t movedOperations = 0;
  while (opIt != op.end()) {
    // move the operation from the compound operation in front of the compound
    // operation in the flattened container. `it` then points to the newly
    // inserted element
    it = qc.insert(it, std::move(*opIt));
    // advance the operation iterator to point past the now moved-from element
    // in the compound operation
    ++opIt;
    // advance the general iterator to again point to the compound operation
    ++it;
    // track the moved operations
    ++movedOperations;
  }
  // whenever all the operations have been processed, `it` points to the
  // compound operation and `opIt` to `op.end()`. The compound operation can now
  // be deleted safely
  it = qc.erase(it);
  // move the general iterator back to the position of the last moved operation
  std::advance(it, -movedOperations);
}
} // namespace

void CircuitOptimizer::flattenOperations(QuantumComputation& qc,
                                         bool customGatesOnly) {
  auto it = qc.begin();
  while (it != qc.end()) {
    if ((*it)->isCompoundOperation()) {
      auto& op = dynamic_cast<CompoundOperation&>(**it);
      if (!customGatesOnly || op.isCustomGate()) {
        flattenCompoundOperation(qc, it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }
}

void CircuitOptimizer::cancelCNOTs(QuantumComputation& qc) {
  auto dag = DAG(qc.getHighestPhysicalQubitIndex() + 1U);

  for (auto& it : qc) {
    if (!it->isStandardOperation()) {
      addToDag(dag, &it);
      continue;
    }

    // check whether the operation is a CNOT or SWAP gate
    const auto isCNOT = (it->getType() == X && it->getNcontrols() == 1U &&
                         it->getControls().begin()->type == Control::Type::Pos);
    const auto isSWAP = (it->getType() == SWAP && it->getNcontrols() == 0U);

    if (!isCNOT && !isSWAP) {
      addToDag(dag, &it);
      continue;
    }

    const Qubit q0 = it->getTargets().at(0);
    const Qubit q1 =
        isSWAP ? it->getTargets().at(1) : it->getControls().begin()->qubit;

    // first operation
    if (dag.at(q0).empty() || dag.at(q1).empty()) {
      addToDag(dag, &it);
      continue;
    }

    auto* op0 = dag.at(q0).back()->get();
    auto* op1 = dag.at(q1).back()->get();

    // check whether it's the same operation at both qubits
    if (op0 != op1) {
      addToDag(dag, &it);
      continue;
    }

    // check whether the operation is a CNOT or SWAP gate
    const auto prevOpIsCNOT =
        (op0->getType() == X && op0->getNcontrols() == 1U &&
         op0->getControls().begin()->type == Control::Type::Pos);
    const auto prevOpIsSWAP =
        (op0->getType() == SWAP && op0->getNcontrols() == 0U);

    if (!prevOpIsCNOT && !prevOpIsSWAP) {
      addToDag(dag, &it);
      continue;
    }

    const Qubit prevQ0 = op0->getTargets().at(0);
    const Qubit prevQ1 = prevOpIsSWAP ? op0->getTargets().at(1)
                                      : op0->getControls().begin()->qubit;

    if (isCNOT && prevOpIsCNOT) {
      // two identical CNOT gates cancel each other
      if (q0 == prevQ0 && q1 == prevQ1) {
        dag.at(q0).pop_back();
        dag.at(q1).pop_back();
        op0->setGate(I);
        op0->clearControls();
        it->setGate(I);
        it->clearControls();
      } else {
        // two CNOTs with alternating controls and targets
        // check whether there is a third one which would make this a SWAP gate

        auto prevPrevOp0It = ++(dag.at(q0).rbegin());
        auto prevPrevOp1It = ++(dag.at(q1).rbegin());
        // check whether there is another operation
        if (prevPrevOp0It == dag.at(q0).rend() ||
            prevPrevOp1It == dag.at(q1).rend()) {
          addToDag(dag, &it);
          continue;
        }

        auto* prevPrevOp0 = (*prevPrevOp0It)->get();
        auto* prevPrevOp1 = (*prevPrevOp1It)->get();

        if (prevPrevOp0 != prevPrevOp1) {
          addToDag(dag, &it);
          continue;
        }

        // check whether the operation is a CNOT
        const auto prevPrevOpIsCNOT =
            (prevPrevOp0->getType() == X && prevPrevOp0->getNcontrols() == 1U &&
             prevPrevOp0->getControls().begin()->type == Control::Type::Pos);

        if (!prevPrevOpIsCNOT) {
          addToDag(dag, &it);
          continue;
        }

        const Qubit prevPrevQ0 = prevPrevOp0->getTargets().at(0);
        const Qubit prevPrevQ1 = prevPrevOp0->getControls().begin()->qubit;

        if (q0 == prevPrevQ0 && q1 == prevPrevQ1) {
          // SWAP gate identified
          prevPrevOp0->setGate(SWAP);
          prevPrevOp0->clearControls();
          if (prevQ0 > prevQ1) {
            prevPrevOp0->setTargets({prevQ1, prevQ0});
          } else {
            prevPrevOp0->setTargets({prevQ0, prevQ1});
          }
          op0->setGate(I);
          op0->clearControls();
          it->setGate(I);
          it->clearControls();
          dag.at(q0).pop_back();
          dag.at(q1).pop_back();
        } else {
          addToDag(dag, &it);
          continue;
        }
      }
      continue;
    }

    if (isSWAP && prevOpIsSWAP) {
      // two identical SWAP gates cancel each other
      if (std::set{q0, q1} == std::set{prevQ0, prevQ1}) {
        dag.at(q0).pop_back();
        dag.at(q1).pop_back();
        op0->setGate(I);
        op0->clearControls();
        it->setGate(I);
        it->clearControls();
      } else {
        addToDag(dag, &it);
      }
      continue;
    }

    if (isCNOT && prevOpIsSWAP) {
      // SWAP followed by a CNOT is equivalent to two CNOTs
      op0->setGate(X);
      op0->setTargets({q0});
      op0->setControls({Control{q1}});
      it->setTargets({q1});
      it->setControls({Control{q0}});
      addToDag(dag, &it);
      continue;
    }

    if (isSWAP && prevOpIsCNOT) {
      // CNOT followed by a SWAP is equivalent to two CNOTs
      op0->setTargets({prevQ1});
      op0->setControls({Control{prevQ0}});
      it->setGate(X);
      it->setTargets({prevQ0});
      it->setControls({Control{prevQ1}});
      addToDag(dag, &it);
      continue;
    }
  }

  removeIdentities(qc);
}

namespace {
void replaceMCXWithMCZ(
    Iterator begin, const std::function<Iterator()>& end,
    const std::function<Iterator(Iterator, std::unique_ptr<Operation>&&)>&
        insert,
    const std::function<Iterator(Iterator)>& erase) {
  for (auto it = begin; it != end(); ++it) {
    auto& op = *it;
    if (op->getType() == X && op->getNcontrols() > 0) {
      const auto& controls = op->getControls();
      assert(op->getNtargets() == 1U);
      const auto target = op->getTargets()[0];

      // -c-    ---c---
      //  |  =     |
      // -X-    -H-Z-H-
      it = insert(it, std::make_unique<StandardOperation>(target, H));
      it = insert(it, std::make_unique<StandardOperation>(controls, target, Z));
      it = insert(it, std::make_unique<StandardOperation>(target, H));
      // advance to the original operation and delete it
      std::advance(it, 3);
      it = erase(it);
      --it;
    } else if (op->isCompoundOperation()) {
      auto* compOp = dynamic_cast<CompoundOperation*>(op.get());
      replaceMCXWithMCZ(
          compOp->begin(), [&compOp] { return compOp->end(); },
          [&compOp](auto iter, auto&& operation) {
            return compOp->insert(iter,
                                  std::forward<decltype(operation)>(operation));
          },
          [&compOp](auto iter) { return compOp->erase(iter); });
    }
  }
}
} // namespace

void CircuitOptimizer::replaceMCXWithMCZ(QuantumComputation& qc) {
  ::qc::replaceMCXWithMCZ(
      qc.begin(), [&qc] { return qc.end(); },
      [&qc](auto it, auto&& op) {
        return qc.insert(it, std::forward<decltype(op)>(op));
      },
      [&qc](auto it) { return qc.erase(it); });
}

namespace {
using ConstReverseIterator = QuantumComputation::const_reverse_iterator;
void backpropagateOutputPermutation(
    const ConstReverseIterator& rbegin, const ConstReverseIterator& rend,
    Permutation& permutation, std::unordered_set<Qubit>& missingLogicalQubits) {
  for (auto it = rbegin; it != rend; ++it) {
    if ((*it)->isCompoundOperation()) {
      auto& op = dynamic_cast<CompoundOperation&>(**it);
      backpropagateOutputPermutation(op.crbegin(), op.crend(), permutation,
                                     missingLogicalQubits);
      continue;
    }

    if ((*it)->getType() == SWAP && !(*it)->isControlled() &&
        (*it)->getTargets().size() == 2U) {
      const auto& targets = (*it)->getTargets();
      // four cases
      // 1. both targets are in the permutation
      // 2. only the first target is in the permutation
      // 3. only the second target is in the permutation
      // 4. neither target is in the permutation

      const auto it0 = permutation.find(targets[0]);
      const auto it1 = permutation.find(targets[1]);

      if (it0 != permutation.end() && it1 != permutation.end()) {
        // case 1: swap the entries
        std::swap(it0->second, it1->second);
        continue;
      }

      if (it0 != permutation.end()) {
        // case 2: swap the value assign the other target from the list of
        // missing logical qubits. Give preference to choosing the same logical
        // qubit as the missing physical qubit
        permutation[targets[1]] = it0->second;

        if (missingLogicalQubits.find(targets[0]) !=
            missingLogicalQubits.end()) {
          missingLogicalQubits.erase(targets[0]);
          it0->second = targets[0];
        } else {
          it0->second = *missingLogicalQubits.begin();
          missingLogicalQubits.erase(missingLogicalQubits.begin());
        }
        continue;
      }

      if (it1 != permutation.end()) {
        // case 3: swap the value assign the other target from the list of
        // missing logical qubits. Give preference to choosing the same logical
        // qubit as the missing physical qubit
        permutation[targets[0]] = it1->second;

        if (missingLogicalQubits.find(targets[1]) !=
            missingLogicalQubits.end()) {
          missingLogicalQubits.erase(targets[1]);
          it1->second = targets[1];
        } else {
          it1->second = *missingLogicalQubits.begin();
          missingLogicalQubits.erase(missingLogicalQubits.begin());
        }
        continue;
      }

      // case 4: nothing to do
    }
  }
}
} // namespace

void CircuitOptimizer::backpropagateOutputPermutation(QuantumComputation& qc) {
  auto permutation = qc.outputPermutation;

  // Collect all logical qubits missing from the output permutation
  std::unordered_set<Qubit> logicalQubits{};
  for (const auto& [physical, logical] : permutation) {
    logicalQubits.insert(logical);
  }
  std::unordered_set<Qubit> missingLogicalQubits{};
  for (Qubit i = 0; i < qc.getNqubits(); ++i) {
    if (logicalQubits.find(i) == logicalQubits.end()) {
      missingLogicalQubits.emplace(i);
    }
  }

  ::qc::backpropagateOutputPermutation(qc.crbegin(), qc.crend(), permutation,
                                       missingLogicalQubits);

  // `permutation` now holds a potentially incomplete initial layout
  // check whether the initial layout is complete and return if it is
  if (permutation.size() == qc.getNqubits()) {
    qc.initialLayout = permutation;
    return;
  }

  // Otherwise, fill the initial layout with the missing logical qubits.
  // Give preference to choosing the same logical qubit as the missing physical
  // qubit (i.e., an identity mapping) to avoid unnecessary permutations.
  for (Qubit i = 0; i < qc.getNqubits(); ++i) {
    if (permutation.find(i) == permutation.end()) {
      if (missingLogicalQubits.find(i) != missingLogicalQubits.end()) {
        permutation.emplace(i, i);
        missingLogicalQubits.erase(i);
      } else {
        permutation.emplace(i, *missingLogicalQubits.begin());
        missingLogicalQubits.erase(missingLogicalQubits.begin());
      }
    }
  }
  assert(missingLogicalQubits.empty());
  qc.initialLayout = permutation;
}

/**
 * @brief Disjoint Set Union data structure for qubits
 *
 * This data structure is used to maintain a relationship between qubits and
 * blocks they belong to. The blocks are formed by operations that act on the
 * same qubits.
 */
struct DSU {
  std::unordered_map<Qubit, Qubit> parent;
  std::unordered_map<Qubit, std::vector<Qubit>> bitBlocks;
  std::unordered_map<Qubit, std::unique_ptr<Operation>*> currentBlockInCircuit;
  std::unordered_map<Qubit, std::unique_ptr<CompoundOperation>>
      currentBlockOperations;
  std::size_t maxBlockSize = 0;
  // list of qubits that are currently in a block
  std::unordered_set<Qubit> currentBlockQubits;
  std::unordered_set<Qubit> alreadyfinalizedQubits;

  /**
   * @brief Check if a block is empty.
   * @param index Qubit to check
   * @return
   */
  [[nodiscard]] bool blockEmpty(const Qubit index) {
    return currentBlockInCircuit[findBlock(index)] == nullptr;
  }

  /**
   * @brief Find the block that a qubit belongs to.
   * @param index Qubit to find the block for
   * @return The block that the qubit belongs to
   */
  Qubit findBlock(const Qubit index) {
    if (parent.find(index) == parent.end()) {
      parent[index] = index;
      bitBlocks[index] = {index};
      currentBlockInCircuit[index] = nullptr;
      currentBlockOperations[index] = std::make_unique<CompoundOperation>();
    }
    if (parent[index] == index) {
      return index;
    }
    parent[index] = findBlock(parent[index]);
    return parent[index];
  }

  /**
   * @brief Merge two blocks together.
   * @details The smaller block is merged into the larger block.
   * @param block1 first block
   * @param block2 second block
   */
  void unionBlock(const Qubit block1, const Qubit block2) {
    auto parent1 = findBlock(block1);
    auto parent2 = findBlock(block2);
    if (parent1 == parent2) {
      return;
    }
    assert(currentBlockOperations[parent1] != nullptr);
    assert(currentBlockOperations[parent2] != nullptr);
    if (currentBlockOperations[parent1]->size() <
        currentBlockOperations[parent2]->size()) {
      std::swap(parent1, parent2);
    }
    parent[parent2] = parent1;
    currentBlockOperations[parent1]->merge(*currentBlockOperations[parent2]);
    bitBlocks[parent1].insert(bitBlocks[parent1].end(),
                              bitBlocks[parent2].begin(),
                              bitBlocks[parent2].end());
    if (currentBlockInCircuit[parent2] != nullptr) {
      (*currentBlockInCircuit[parent2]) =
          std::make_unique<StandardOperation>(0, I);
    }
    currentBlockInCircuit[parent2] = nullptr;
    currentBlockOperations[parent2] = std::make_unique<CompoundOperation>();
    bitBlocks[parent2].clear();
  }

  /**
   * @brief Finalize a block.
   * @details This replaces the original operation in the circuit with all the
   * operations in the block. If the block is empty, nothing is done. If the
   * block only contains a single operation, the operation is replaced with the
   * single operation. Otherwise, the block is replaced with a compound
   * operation.
   * @param index the qubit that the block belongs to
   */
  void finalizeBlock(const Qubit index) {
    const auto block = findBlock(index);
    if (currentBlockInCircuit[block] == nullptr) {
      return;
    }
    auto& compoundOp = currentBlockOperations[block];
    if (compoundOp->isConvertibleToSingleOperation()) {
      *currentBlockInCircuit[block] = compoundOp->collapseToSingleOperation();
    } else {
      *currentBlockInCircuit[block] = std::move(compoundOp);
    }
    // need to make a copy here because otherwise the updates in the loop might
    // invalidate the iterator
    const auto blockBits = bitBlocks[block];
    for (auto i : blockBits) {
      parent[i] = i;
      bitBlocks[i] = {i};
      currentBlockInCircuit[i] = nullptr;
      currentBlockOperations[i] = std::make_unique<CompoundOperation>();
    }
  }
};

void CircuitOptimizer::collectCliffordBlocks(QuantumComputation& qc,
                                             const std::size_t maxBlockSize) {
  if (qc.size() <= 1) {
    return;
  }

  qc.reorderOperations();
  deferMeasurements(qc);

  DSU dsu;
  dsu.maxBlockSize = maxBlockSize;

  DSU NoneCliffordDSU;
  NoneCliffordDSU.maxBlockSize = 1;
  for (auto opIt = qc.begin(); opIt != qc.end(); ++opIt) {
    auto& op = *opIt;
    bool canProcess = true;
    bool makesTooBig = false;

    if (!op->isUnitary()) {
      canProcess = false;
    }
    const auto usedQubits = op->getUsedQubits();
  }

  removeIdentities(qc);
}

void CircuitOptimizer::collectBlocks(QuantumComputation& qc,
                                     const std::size_t maxBlockSize,
                                     bool onlyCollectCliffords) {
  if (qc.size() <= 1) {
    return;
  }

  // ensure canonical ordering and that measurements are as far back as possible
  qc.reorderOperations();
  deferMeasurements(qc);

  // create an empty disjoint set union data structure
  DSU dsu{};
  DSU nonCliffordDSU{};
  for (auto opIt = qc.begin(); opIt != qc.end(); ++opIt) {

    auto& op = *opIt;

    bool canProcess = true;
    bool makesTooBig = false;

    if (!op->isUnitary()) {
      canProcess = false;
    }
    const auto usedQubits = op->getUsedQubits();
    // print operation and on what qubit it acts
    std::cout << "Processing operation: " << op->getName() << " on qubits: ";
    for (const auto& q : usedQubits) {
      std::cout << q << " ";
    }
    std::cout << std::endl;

    if (onlyCollectCliffords && !op->isClifford()) {
      // add used qubits to the non_Clifford DSU
      std::unordered_set<Qubit> non_Clifford_qubits{};
      for (const auto& q : usedQubits) {
        nonCliffordDSU.findBlock(q);
        non_Clifford_qubits.emplace(q);
        dsu.alreadyfinalizedQubits.emplace(q);
      }

      // make new DSU add the gate where the non Clifford operation is operating
      // on and finalize this block right away
      //  if the operation acts on more qubits than the maximum block size, all
      //  current blocks need to be finalized.
      for (const auto& q : non_Clifford_qubits) {
        nonCliffordDSU.finalizeBlock(q);
      }
    }

    if (canProcess) {
      // check if grouping the operation with the current block would make the
      // block too big
      std::unordered_set<Qubit> blockQubits;
      for (const auto& q : usedQubits) {
        blockQubits.emplace(dsu.findBlock(q));
      }
      std::size_t totalSize = 0;
      for (const auto& q : blockQubits) {
        totalSize += dsu.bitBlocks[q].size();
      }
      if (totalSize > maxBlockSize) {
        makesTooBig = true;
      }
    } else {
      // resolve cases where an operation cannot be processed
      for (const auto& q : usedQubits) {
        dsu.finalizeBlock(q);
      }
    }

    if (makesTooBig) {
      // if the operation acts on more qubits than the maximum block size, all
      // current blocks need to be finalized.
      if (usedQubits.size() > maxBlockSize) {
        // get all of the relevant blocks and check for the best way to combine
        // them together.
        std::unordered_map<Qubit, std::size_t> blocksAndSizes{};
        for (const auto& q : usedQubits) {
          const auto block = dsu.findBlock(q);
          if (dsu.blockEmpty(block) ||
              blocksAndSizes.find(block) != blocksAndSizes.end()) {
            continue;
          }
          blocksAndSizes[block] = dsu.bitBlocks[block].size();
        }
        // sort blocks in descending order
        std::vector<std::pair<Qubit, std::size_t>> sortedBlocks(
            blocksAndSizes.begin(), blocksAndSizes.end());
        std::sort(
            sortedBlocks.begin(), sortedBlocks.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        for (auto it = sortedBlocks.begin(); it != sortedBlocks.end(); ++it) {
          auto& [block, size] = *it;
          // maximally large block -> nothing to do
          if (size == maxBlockSize) {
            dsu.finalizeBlock(block);
            continue;
          }

          // fill up with as many blocks as possible
          auto nextIt = it + 1;
          while (nextIt != sortedBlocks.end() && size < maxBlockSize) {
            auto& [nextBlock, nextSize] = *nextIt;
            if (size + nextSize <= maxBlockSize) {
              dsu.unionBlock(block, nextBlock);
              size += nextSize;
              nextIt = sortedBlocks.erase(nextIt);
            } else {
              ++nextIt;
            }
          }
          dsu.finalizeBlock(block);
        }
      } else {
        // otherwise, finalize blocks that would free up enough space.
        // prioritize blocks that would free up the most space.
        std::unordered_map<Qubit, std::size_t> savings{};
        std::size_t totalSize = 0U;
        for (const auto& q : usedQubits) {
          const auto block = dsu.findBlock(q);
          if (savings.find(block) != savings.end()) {
            savings[block] -= 1;
          } else {
            savings[block] = dsu.bitBlocks[block].size() - 1;
            totalSize += dsu.bitBlocks[block].size();
          }
        }
        // sort savings in descending order
        std::vector<std::pair<Qubit, std::size_t>> sortedSavings(
            savings.begin(), savings.end());
        std::sort(
            sortedSavings.begin(), sortedSavings.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        auto savingsNeed = static_cast<std::int64_t>(totalSize - maxBlockSize);
        for (const auto& [index, saving] : sortedSavings) {
          if (savingsNeed > 0) {
            savingsNeed -= static_cast<std::int64_t>(saving);
            dsu.finalizeBlock(index);
          }
        }
      }
    }

    if (canProcess) {
      if (usedQubits.size() > maxBlockSize) {
        continue;
      }
      std::int64_t prev = -1;
      for (const auto& q : usedQubits) {
        if (prev != -1) {
          dsu.unionBlock(static_cast<Qubit>(prev), q);
        }
        prev = q;
      }

      if (onlyCollectCliffords && !op->isClifford()) {
        // check if next operation which acts on other qubits within the same
        // block fits in the block
        if (opIt + 1 != qc.end() && (*std::next(opIt))->isUnitary()) {
          const auto& nextOp = *(opIt + 1);
          const auto nextUsedQubits = nextOp->getUsedQubits();
          std::unordered_set<Qubit> nextBlockQubits;
          for (const auto& q : nextUsedQubits) {
            nextBlockQubits.emplace(dsu.findBlock(q));
          }
          std::size_t nextTotalSize = 0;
          for (const auto& q : nextBlockQubits) {
            nextTotalSize += dsu.bitBlocks[q].size();
          }
          if (nextTotalSize <= maxBlockSize) {
            // if the next operation fits in the block, do not finalize the
            // add operation to the block and then finalize it
            // current block yet
            for (const auto& q : usedQubits) {
              dsu.finalizeBlock(q);
            }
            continue;
          }
        }

        for (auto q : op->getUsedQubits()) {
          dsu.finalizeBlock(q);
        }
        continue;
      }

      const auto block = dsu.findBlock(static_cast<Qubit>(prev));
      const auto empty = dsu.blockEmpty(block);
      if (empty) {
        dsu.currentBlockInCircuit[block] = &(*opIt);
      }
      dsu.currentBlockOperations[block]->emplace_back(std::move(op));
      // if this is not the first operation in a block, remove it from the
      // circuit
      if (!empty) {
        opIt = qc.erase(opIt);
        // this can only ever be called on at least the second operation in a
        // circuit, so it is safe to decrement the iterator here.
        --opIt;
      }
    }
  }

  // finalize remaining blocks and remove identities
  for (const auto& [q, index] : dsu.parent) {
    if (q == index) {
      dsu.finalizeBlock(q);
    }
  }
  removeIdentities(qc);
}

namespace {
void elidePermutations(Iterator begin, const std::function<Iterator()>& end,
                       const std::function<Iterator(Iterator)>& erase,
                       Permutation& permutation) {
  for (auto it = begin; it != end();) {
    auto& op = *it;
    if (auto* compOp = dynamic_cast<CompoundOperation*>(op.get())) {
      elidePermutations(
          compOp->begin(), [&compOp]() { return compOp->end(); },
          [&compOp](auto iter) { return compOp->erase(iter); }, permutation);
      if (compOp->empty()) {
        it = erase(it);
        continue;
      }
      if (compOp->isConvertibleToSingleOperation()) {
        *it = compOp->collapseToSingleOperation();
      } else {
        // also update the tracked controls in the compound operation
        compOp->getControls() = permutation.apply(compOp->getControls());
      }
      ++it;
      continue;
    }

    if (op->getType() == SWAP && !op->isControlled()) {
      const auto& targets = op->getTargets();
      assert(targets.size() == 2U);
      assert(permutation.find(targets[0]) != permutation.end());
      assert(permutation.find(targets[1]) != permutation.end());
      auto& target0 = permutation[targets[0]];
      auto& target1 = permutation[targets[1]];
      std::swap(target0, target1);
      it = erase(it);
      continue;
    }

    op->apply(permutation);
    ++it;
  }
}
} // namespace

void CircuitOptimizer::elidePermutations(QuantumComputation& qc) {
  if (qc.empty()) {
    return;
  }

  auto permutation = qc.initialLayout;
  ::qc::elidePermutations(
      qc.begin(), [&qc]() { return qc.end(); },
      [&qc](auto it) { return qc.erase(it); }, permutation);

  // adjust the initial layout
  Permutation initialLayout{};
  for (auto& [physical, logical] : qc.initialLayout) {
    initialLayout[logical] = logical;
  }
  qc.initialLayout = initialLayout;

  // adjust the output permutation
  Permutation outputPermutation{};
  for (auto& [physical, logical] : qc.outputPermutation) {
    assert(permutation.find(physical) != permutation.end());
    outputPermutation[permutation[physical]] = logical;
  }
  qc.outputPermutation = outputPermutation;
}

} // namespace qc
