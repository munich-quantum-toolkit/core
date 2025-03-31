/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "algorithms/BernsteinVazirani.hpp"

#include "ir/Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>

namespace qc {

namespace {
[[nodiscard]] auto getMostSignificantBit(const BVBitString& s) -> std::size_t {
  std::size_t msb = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s.test(i)) {
      msb = i;
    }
  }
  return msb;
}

[[nodiscard]] auto generateBitstring(const std::size_t nq, std::mt19937_64& mt)
    -> BVBitString {
  BVBitString s;
  auto distribution = std::bernoulli_distribution();
  for (std::size_t i = 0; i < nq; ++i) {
    if (distribution(mt)) {
      s.set(i);
    }
  }
  return s;
}

[[nodiscard]] auto getExpected(const BVBitString& s, const Qubit nq)
    -> std::string {
  auto expected = s.to_string();
  std::reverse(expected.begin(), expected.end());
  while (expected.length() > nq) {
    expected.pop_back();
  }
  std::reverse(expected.begin(), expected.end());
  return expected;
}

auto constructBernsteinVaziraniCircuit(QuantumComputation& qc,
                                       const BVBitString& s, const Qubit nq) {
  qc.setName("bv_" + getExpected(s, nq));
  qc.addQubitRegister(1, "flag");
  qc.addQubitRegister(nq, "q");
  qc.addClassicalRegister(nq, "c");

  // prepare flag qubit
  qc.x(0);

  // set up initial layout
  qc.initialLayout[0] = static_cast<Qubit>(nq);
  for (Qubit i = 1; i <= nq; ++i) {
    qc.initialLayout[i] = i - 1;
  }
  qc.setLogicalQubitGarbage(nq);
  qc.outputPermutation.erase(0);

  // initial Hadamard transformation
  for (Qubit i = 1; i <= nq; ++i) {
    qc.h(i);
  }

  // apply controlled-Z gates according to secret bitstring
  for (Qubit i = 1; i <= nq; ++i) {
    if (s.test(i - 1)) {
      qc.cz(i, 0);
    }
  }

  // final Hadamard transformation
  for (Qubit i = 1; i <= nq; ++i) {
    qc.h(i);
  }

  // measure results
  for (Qubit i = 1; i <= nq; i++) {
    qc.measure(i, i - 1);
    qc.outputPermutation[i] = i - 1;
  }
}
} // namespace

auto createBernsteinVazirani(const BVBitString& hiddenString)
    -> QuantumComputation {
  const auto msb = static_cast<Qubit>(getMostSignificantBit(hiddenString));
  return createBernsteinVazirani(hiddenString, msb + 1);
}

auto createBernsteinVazirani(const Qubit nq, const std::size_t seed)
    -> QuantumComputation {
  auto qc = QuantumComputation(0, 0, seed);
  const auto hiddenString = generateBitstring(nq, qc.getGenerator());
  constructBernsteinVaziraniCircuit(qc, hiddenString, nq);
  return qc;
}

auto createBernsteinVazirani(const BVBitString& hiddenString, const Qubit nq)
    -> QuantumComputation {
  auto qc = QuantumComputation(0, 0);
  constructBernsteinVaziraniCircuit(qc, hiddenString, nq);
  return qc;
}

namespace {
auto constructIterativeBernsteinVaziraniCircuit(QuantumComputation& qc,
                                                const BVBitString& s,
                                                const Qubit nq) {
  qc.setName("iterative_bv_" + getExpected(s, nq));
  qc.addQubitRegister(1, "flag");
  qc.addQubitRegister(1, "q");
  qc.addClassicalRegister(nq, "c");

  // prepare flag qubit
  qc.x(0);

  // set up initial layout
  qc.initialLayout[0] = 1;
  qc.initialLayout[1] = 0;
  qc.setLogicalQubitGarbage(1);
  qc.outputPermutation.erase(0);
  qc.outputPermutation[1] = 0;

  for (std::size_t i = 0; i < nq; ++i) {
    // initial Hadamard
    qc.h(1);

    // apply controlled-Z gate according to secret bitstring
    if (s.test(i)) {
      qc.cz(1, 0);
    }

    // final Hadamard
    qc.h(1);

    // measure result
    qc.measure(1, i);

    // reset qubit if not finished
    if (i < nq - 1) {
      qc.reset(1);
    }
  }
  return qc;
}
} // namespace

auto createIterativeBernsteinVazirani(const BVBitString& hiddenString)
    -> QuantumComputation {
  const auto msb = static_cast<Qubit>(getMostSignificantBit(hiddenString));
  return createIterativeBernsteinVazirani(hiddenString, msb + 1);
}

auto createIterativeBernsteinVazirani(const Qubit nq, const std::size_t seed)
    -> QuantumComputation {
  auto qc = QuantumComputation(0, 0, seed);
  const auto hiddenString = generateBitstring(nq, qc.getGenerator());
  constructIterativeBernsteinVaziraniCircuit(qc, hiddenString, nq);
  return qc;
}

auto createIterativeBernsteinVazirani(const BVBitString& hiddenString,
                                      const Qubit nq) -> QuantumComputation {
  auto qc = QuantumComputation(0, 0);
  constructIterativeBernsteinVaziraniCircuit(qc, hiddenString, nq);
  return qc;
}
} // namespace qc
