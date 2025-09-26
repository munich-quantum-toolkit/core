/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/runtime/Runtime.hpp"

#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "ir/Definitions.hpp"
#include "qir/runtime/QIR.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qir {

auto Runtime::generateRandomSeed() -> uint64_t {
  std::array<std::random_device::result_type, std::mt19937_64::state_size>
      randomData{};
  std::random_device rd;
  std::ranges::generate(randomData, std::ref(rd));
  std::seed_seq seeds(randomData.begin(), randomData.end());
  std::mt19937_64 rng(seeds);
  return rng();
}
Runtime& Runtime::getInstance() {
  static Runtime instance;
  return instance;
}
auto Runtime::reset() -> void {
  addressMode = AddressMode::UNKNOWN;
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  numQubitsInQState = 0;
  dd->decRef(qState);
  dd->garbageCollect();
  qState = dd::vEdge::one();
  mt.seed(generateRandomSeed());
  qRegister.clear();
  rRegister.clear();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ZERO_ADDRESS),
                    ResultStruct{.refcount = 0, .r = false});
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ONE_ADDRESS),
                    ResultStruct{.refcount = 0, .r = true});
  // NOLINTEND(performance-no-int-to-ptr)
}

Runtime::Runtime() : Runtime(generateRandomSeed()) {}

Runtime::Runtime(const uint64_t randomSeed)
    : addressMode(AddressMode::UNKNOWN),
      currentMaxQubitAddress(MIN_DYN_QUBIT_ADDRESS), currentMaxQubitId(0),
      currentMaxResultAddress(MIN_DYN_RESULT_ADDRESS), numQubitsInQState(0),
      dd(std::make_unique<dd::Package>()), qState(dd::vEdge::one()),
      mt(randomSeed) {
  qRegister = std::unordered_map<const Qubit*, qc::Qubit>();
  rRegister = std::unordered_map<Result*, ResultStruct>();
  // NOLINTBEGIN(performance-no-int-to-ptr)
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ZERO_ADDRESS),
                    ResultStruct{.refcount = 0, .r = false});
  rRegister.emplace(reinterpret_cast<Result*>(RESULT_ONE_ADDRESS),
                    ResultStruct{.refcount = 0, .r = true});
  // NOLINTEND(performance-no-int-to-ptr)
}

auto Runtime::enlargeState(const std::uint64_t maxQubit) -> void {
  if (maxQubit >= numQubitsInQState) {
    const auto d = maxQubit - numQubitsInQState + 1;
    qubitPermutation.resize(numQubitsInQState + d);
    std::iota(qubitPermutation.begin() +
                  static_cast<std::vector<qc::Qubit>::difference_type>(
                      numQubitsInQState),
              qubitPermutation.end(), numQubitsInQState);
    numQubitsInQState += d;

    // resize the DD package only if necessary
    if (dd->qubits() < numQubitsInQState) {
      dd->resize(numQubitsInQState);
    }

    // if the state is terminal, we need to create a new node
    if (qState.isTerminal()) {
      qState = dd::makeZeroState(d, *dd);
      return;
    }

    // enlarge state
    for (auto q = qState.p->v; q < numQubitsInQState; ++q) {
      auto old = qState;
      qState = dd->makeDDNode(q + 1U, std::array{qState, dd::vEdge::zero()});
      dd->incRef(qState);
      dd->decRef(old);
    }
  }
}

auto Runtime::swap(Qubit* qubit1, Qubit* qubit2) -> void {
  const auto target1 = translateAddresses(std::array{qubit1})[0];
  const auto target2 = translateAddresses(std::array{qubit2})[0];
  std::swap(qubitPermutation[target1], qubitPermutation[target2]);
}

auto Runtime::qAlloc() -> Qubit* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  qRegister.emplace(qubit, currentMaxQubitId++);
  return qubit;
}

auto Runtime::qFree(Qubit* qubit) -> void {
  reset<1>({{qubit}});
  qRegister.erase(qubit);
}

auto Runtime::rAlloc() -> Result* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{.refcount = 1, .r = false});
  return result;
}

auto Runtime::rFree(Result* result) -> void { rRegister.erase(result); }

auto Runtime::deref(Result* result) -> ResultStruct& {
  auto it = rRegister.find(result);
  if (it == rRegister.end()) {
    if (addressMode != AddressMode::UNKNOWN) {
      addressMode = AddressMode::STATIC;
    }
    if (addressMode == AddressMode::DYNAMIC) {
      std::stringstream ss;
      ss << __FILE__ << ":" << __LINE__
         << ": Result not allocated (not found): " << result;
      throw std::out_of_range(ss.str());
    }
    it = rRegister.emplace(result, ResultStruct{.refcount = 0, .r = false})
             .first;
  }
  return it->second;
}

auto Runtime::equal(Result* result1, Result* result2) -> bool {
  return deref(result1).r == deref(result2).r;
}

} // namespace qir
