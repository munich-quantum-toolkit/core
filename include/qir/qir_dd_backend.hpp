#pragma once

#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"

namespace mqt {
class QIR_DD_Backend {
private:
  std::unordered_map<qc::Qubit, qc::Qubit> qRegister;
  dd::vEdge qState;

  QIR_DD_Backend() = default;
};
} // namespace mqt
