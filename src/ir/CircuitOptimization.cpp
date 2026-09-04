/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace qc {
namespace {
using DAG = std::vector<std::deque<std::unique_ptr<Operation>*>>;
using DAGReverseIterator =
    std::deque<std::unique_ptr<Operation>*>::reverse_iterator;
using DAGReverseIterators = std::vector<DAGReverseIterator>;

void addToDag(DAG& dag, std::unique_ptr<Operation>* op) {
  const auto usedQubits = (*op)->getUsedQubits();
  for (const auto q : usedQubits) {
    dag.at(q).push_back(op);
  }
}

DAG constructDAG(QuantumComputation& qc) {
  auto dag = DAG(qc.getHighestPhysicalQubitIndex() + 1);
  for (auto& op : qc) {
    addToDag(dag, &op);
  }
  return dag;
}

void removeIdentities(QuantumComputation& qc) {
  auto it = qc.begin();
  while (it != qc.end()) {
    if ((*it)->getType() == I) {
      it = qc.erase(it);
    } else if ((*it)->isCompoundOperation()) {
      auto& compOp = dynamic_cast<CompoundOperation&>(**it);
      auto cit = compOp.cbegin();
      while (cit != compOp.cend()) {
        if ((*cit)->getType() == I) {
          cit = compOp.erase(cit);
        } else {
          ++cit;
        }
      }
      if (compOp.empty()) {
        it = qc.erase(it);
      } else {
        if (compOp.size() == 1) {
          (*it) = std::move(*(compOp.begin()));
        }
        ++it;
      }
    } else {
      ++it;
    }
  }
}

bool removeFinalMeasurement(DAG& dag, DAGReverseIterators& dagIterators,
                            Qubit idx, const DAGReverseIterator& it,
                            Operation* op);

void removeFinalMeasurementsRecursive(DAG& dag,
                                      DAGReverseIterators& dagIterators,
                                      Qubit idx, const Operation* until) {
  if (dagIterators.at(idx) == dag.at(idx).rend()) {
    if (idx < static_cast<Qubit>(dag.size() - 1)) {
      removeFinalMeasurementsRecursive(dag, dagIterators, idx + 1, nullptr);
    }
    return;
  }
  if (until != nullptr && (*dagIterators.at(idx))->get() == until) {
    return;
  }

  auto& it = dagIterators.at(idx);
  while (it != dag.at(idx).rend()) {
    if (until != nullptr && (*dagIterators.at(idx))->get() == until) {
      break;
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
          ++dagIterators.at(target);
        }
      }
    } else if (op->isCompoundOperation() && op->isNonUnitaryOperation()) {
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
        ++dagIterators.at(idx);
      }
    } else {
      dagIterators.at(idx) = dag.at(idx).rend();
      break;
    }
  }
  if (dagIterators.at(idx) == dag.at(idx).rend() &&
      idx < static_cast<Qubit>(dag.size() - 1)) {
    removeFinalMeasurementsRecursive(dag, dagIterators, idx + 1, nullptr);
  }
}

bool removeFinalMeasurement(DAG& dag, DAGReverseIterators& dagIterators,
                            const Qubit idx, const DAGReverseIterator& it,
                            Operation* op) {
  if (op->getNtargets() == 0) {
    return false;
  }

  bool onlyMeasurements = true;
  for (const auto& target : op->getTargets()) {
    if (target == idx) {
      continue;
    }
    if (dagIterators.at(target) == dag.at(target).rend()) {
      onlyMeasurements = false;
      break;
    }
    removeFinalMeasurementsRecursive(dag, dagIterators, target, (*it)->get());
    if (dagIterators.at(target) == dag.at(target).rend() ||
        *dagIterators.at(target) != *it) {
      onlyMeasurements = false;
      break;
    }
  }
  if (!onlyMeasurements) {
    dagIterators.at(idx) = dag.at(idx).rend();
  } else {
    op->setGate(I);
  }
  return onlyMeasurements;
}

using Iterator = QuantumComputation::iterator;

void flattenCompoundOperation(QuantumComputation& qc, Iterator& it) {
  assert((*it)->isCompoundOperation());
  auto& op = dynamic_cast<CompoundOperation&>(**it);
  auto opIt = op.begin();
  int64_t movedOperations = 0;
  while (opIt != op.end()) {
    it = qc.insert(it, std::move(*opIt));
    ++opIt;
    ++it;
    ++movedOperations;
  }
  it = qc.erase(it);
  std::advance(it, -movedOperations);
}
} // namespace

void QuantumComputation::removeFinalMeasurements() {
  auto dag = constructDAG(*this);
  DAGReverseIterators dagIterators{dag.size()};
  for (size_t q = 0; q < dag.size(); ++q) {
    dagIterators.at(q) = dag.at(q).rbegin();
  }

  removeFinalMeasurementsRecursive(dag, dagIterators, 0, nullptr);
  removeIdentities(*this);
}

void QuantumComputation::flattenOperations(const bool customGatesOnly) {
  auto it = begin();
  while (it != end()) {
    if ((*it)->isCompoundOperation()) {
      auto& op = dynamic_cast<CompoundOperation&>(**it);
      if (!customGatesOnly || op.isCustomGate()) {
        flattenCompoundOperation(*this, it);
      } else {
        ++it;
      }
    } else {
      ++it;
    }
  }
}

} // namespace qc
