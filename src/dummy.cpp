#include "dummy.h"

#include <ir/QuantumComputation.hpp>

namespace mqt {
auto dummyCircuit() -> qc::QuantumComputation {
  auto qc = qc::QuantumComputation(2);
  qc.h(0);
  qc.cx(0, 1);
  qc.measureAll();
  return qc;
}
} // namespace mqt
