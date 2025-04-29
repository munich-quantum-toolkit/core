/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/RandomCliffordCircuit.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/CompoundOperation.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

namespace qc {
namespace {
auto append1QClifford(QuantumComputation& circ, const std::uint16_t idx)
    -> void {
  const auto id = static_cast<std::uint8_t>(idx % 24);
  auto qc = QuantumComputation(circ.getNqubits());
  // Hadamard
  if ((id / 12 % 2) != 0) {
    qc.h(0);
  }

  // Rotation
  if (id / 4 % 3 == 1) {
    qc.h(0);
    qc.s(0);
  } else if (id / 4 % 3 == 2) {
    qc.sdg(0);
    qc.h(0);
  }

  // Pauli
  if (id % 4 == 1) {
    qc.z(0);
  } else if (id % 4 == 2) {
    qc.x(0);
  } else if (id % 4 == 3) {
    qc.y(0);
  }
  circ.emplace_back<CompoundOperation>(qc.asCompoundOperation());
}

auto append2QClifford(QuantumComputation& circ, const std::uint16_t idx,
                      const Qubit control, const Qubit target) -> void {
  auto id = static_cast<std::uint16_t>(idx % 11520);
  const auto pauliIdx = static_cast<std::uint8_t>(id % 16);
  id /= 16;

  auto qc = QuantumComputation(circ.getNqubits());
  if (id < 36) {
    // single-qubit Cliffords
    if ((id / 9 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 18 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 3 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 3 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }
  } else if (id < 360) {
    // Cliffords with a single CNOT
    id -= 36;

    if ((id / 81 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 162 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 3 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 3 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }

    qc.cx(control, target);

    if (id / 9 % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id / 9 % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 27 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 27 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }
  } else if (id < 684) {
    // Cliffords with two CNOTs
    id -= 360;

    if ((id / 81 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 162 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 3 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 3 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }

    qc.cx(control, target);
    qc.cx(target, control);

    if (id / 9 % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id / 9 % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
  } else {
    // Cliffords with a SWAP
    id -= 684;

    if ((id / 9 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 18 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }

    qc.cx(control, target);
    qc.cx(target, control);
    qc.cx(control, target);
  }

  // random Pauli on control qubit
  if (pauliIdx % 4 == 1) {
    qc.z(control);
  } else if (pauliIdx % 4 == 2) {
    qc.x(control);
  } else if (pauliIdx % 4 == 3) {
    qc.y(control);
  }

  // random Pauli on target qubit
  if (pauliIdx / 4 == 1) {
    qc.z(target);
  } else if (pauliIdx / 4 == 2) {
    qc.x(target);
  } else if (pauliIdx / 4 == 3) {
    qc.y(target);
  }

  circ.emplace_back<CompoundOperation>(qc.asCompoundOperation());
}
} // namespace

auto createRandomCliffordCircuit(const Qubit nq, const std::size_t depth,
                                 const std::size_t seed) -> QuantumComputation {
  auto qc = QuantumComputation(nq, nq, seed);
  qc.setName("random_clifford_" + std::to_string(nq) + "_" +
             std::to_string(depth) + "_" + std::to_string(seed));

  std::uniform_int_distribution<std::uint16_t> distribution(0, 11520);
  auto cliffordGenerator = [&]() { return distribution(qc.getGenerator()); };

  for (std::size_t l = 0; l < depth; ++l) {
    if (nq == 1) {
      append1QClifford(qc, cliffordGenerator());
    } else if (nq == 2) {
      append2QClifford(qc, cliffordGenerator(), 0, 1);
    } else {
      if (l % 2 != 0) {
        for (Qubit i = 1; i < nq - 1; i += 2) {
          append2QClifford(qc, cliffordGenerator(), i, i + 1);
        }
      } else {
        for (Qubit i = 0; i < nq - 1; i += 2) {
          append2QClifford(qc, cliffordGenerator(), i, i + 1);
        }
      }
    }
  }
  return qc;
}

} // namespace qc
