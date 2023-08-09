#pragma once

#include <QuantumComputation.hpp>

namespace qc {
class WState : public QuantumComputation {
public:
  explicit WState(std::size_t nq);
};
} // namespace qc
