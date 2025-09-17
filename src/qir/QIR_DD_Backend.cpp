/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/QIR_DD_Backend.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qir/QIR.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qir {

auto QIR_DD_Backend::generateRandomSeed() -> uint64_t {
  std::array<std::random_device::result_type, std::mt19937_64::state_size>
      randomData{};
  std::random_device rd;
  std::ranges::generate(randomData, std::ref(rd));
  std::seed_seq seeds(randomData.begin(), randomData.end());
  std::mt19937_64 rng(seeds);
  return rng();
}
QIR_DD_Backend& QIR_DD_Backend::getInstance() {
  static QIR_DD_Backend instance;
  return instance;
}
auto QIR_DD_Backend::resetBackend() -> void {
  addressMode = AddressMode::UNKNOWN;
  currentMaxQubitAddress = MIN_DYN_QUBIT_ADDRESS;
  currentMaxQubitId = 0;
  currentMaxResultAddress = MIN_DYN_RESULT_ADDRESS;
  numQubitsInQState = 0;
  dd->decRef(qState);
  qState = dd::vEdge::one();
  dd->incRef(qState);
  dd->garbageCollect();
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

QIR_DD_Backend::QIR_DD_Backend() : QIR_DD_Backend(generateRandomSeed()) {}

QIR_DD_Backend::QIR_DD_Backend(const uint64_t randomSeed)
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

template <size_t SIZE>
auto QIR_DD_Backend::translateAddresses(std::array<Qubit*, SIZE> qubits)
    -> std::array<qc::Qubit, SIZE> {
  // extract addresses from opaque qubit pointers
  std::array<qc::Qubit, SIZE> qubitIds{};
  if (addressMode != AddressMode::STATIC) {
    // addressMode == AddressMode::DYNAMIC or AddressMode::UNKNOWN
    try {
      Utils::transform(
          [&](const auto q) {
            try {
              return qRegister.at(q);
            } catch (const std::out_of_range&) {
              std::stringstream ss;
              ss << __FILE__ << ":" << __LINE__
                 << ": Qubit not allocated (not found): " << q;
              throw std::out_of_range(ss.str());
            }
          },
          qubits, qubitIds);
    } catch (std::out_of_range&) {
      if (addressMode == AddressMode::DYNAMIC) {
        throw; // rethrow
      }
      // addressMode == AddressMode::UNKNOWN
      addressMode = AddressMode::STATIC;
    }
  }
  // addressMode might have changed to STATIC
  if (addressMode == AddressMode::STATIC) {
    Utils::transform(
        [](const auto q) {
          return static_cast<qc::Qubit>(reinterpret_cast<uintptr_t>(q));
        },
        qubits, qubitIds);
  }
  const auto maxQubit = *std::max_element(qubitIds.cbegin(), qubitIds.cend());
  enlargeState(maxQubit);
  return qubitIds;
}

template <typename... Args>
auto QIR_DD_Backend::createOperation(qc::OpType op, Args&... args)
    -> qc::StandardOperation {
  const auto& params = Utils::packOfType<qc::fp>(args...);
  const auto& qubits = Utils::packOfType<Qubit*>(args...);
  static_assert(
      std::tuple_size_v<std::remove_reference_t<decltype(params)>> +
              std::tuple_size_v<std::remove_reference_t<decltype(qubits)>> ==
          sizeof...(Args),
      "Number of parameters and qubits must match the number of "
      "arguments. Parameters must come first followed by the qubits.");

  auto addresses = translateAddresses(qubits);
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    addresses[i] = qubitPermutation[addresses[i]];
  }
  // store parameters into vector (without copying)
  const std::vector<qc::fp> paramVec(params.data(),
                                     params.data() + params.size());
  // split addresses into control and target
  uint8_t t = 0;
  if (isSingleQubitGate(op)) {
    t = 1;
  } else if (isTwoQubitGate(op)) {
    t = 2;
  } else {
    std::stringstream ss;
    ss << __FILE__ << ":" << __LINE__
       << ": Operation type is not known: " << toString(op);
    throw std::invalid_argument(ss.str());
  }
  if (qubits.size() > t) { // create controlled operation
    const auto& controls =
        qc::Controls(addresses.cbegin(), addresses.cend() - t);
    const auto& targets = qc::Targets(addresses.data() + (qubits.size() - t),
                                      addresses.data() + qubits.size());
    return {controls, targets, op, paramVec};
  }
  if (qubits.size() == t) { // create uncontrolled operation
    const auto targets = qc::Targets(addresses.data(), addresses.data() + t);
    return {targets, op, paramVec};
  }
  std::stringstream ss;
  ss << __FILE__ << ":" << __LINE__
     << ": Operation requires more qubits than given (" << toString(op)
     << "): " << qubits.size();
  throw std::invalid_argument(ss.str());
}

auto QIR_DD_Backend::enlargeState(const std::uint64_t maxQubit) -> void {
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
      dd->decRef(qState);
      qState = dd->makeDDNode(1U, std::array{qState, dd::vEdge::zero()});
      dd->incRef(qState);
      return;
    }

    // enlarge state
    for (auto q = qState.p->v; q < numQubitsInQState; ++q) {
      dd->decRef(qState);
      qState = dd->makeDDNode(q + 1U, std::array{qState, dd::vEdge::zero()});
      dd->incRef(qState);
    }
  }
}

template <typename... Args>
auto QIR_DD_Backend::apply(const qc::OpType op, Args&&... args) -> void {
  const qc::StandardOperation& operation =
      createOperation(op, std::forward<Args>(args)...);

  qState = applyUnitaryOperation(operation, qState, *dd);
}

template <typename... Args> auto QIR_DD_Backend::measure(Args... args) -> void {
  const auto& qubits = Utils::packOfType<Qubit*>(args...);
  const auto& results = Utils::packOfType<Result*>(args...);
  static_assert(
      std::tuple_size_v<std::remove_reference_t<decltype(qubits)>> ==
          std::tuple_size_v<std::remove_reference_t<decltype(results)>>,
      "Number of qubits and results must match. First, all qubits followed "
      "then by all results.");
  static_assert(
      std::tuple_size_v<std::remove_reference_t<decltype(qubits)>> +
              std::tuple_size_v<std::remove_reference_t<decltype(results)>> ==
          sizeof...(Args),
      "Number of qubits and results must match the number of arguments. First, "
      "all qubits followed then by all results.");
  auto targets = translateAddresses(qubits);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    targets[i] = qubitPermutation[targets[i]];
  }
  // measure qubits
  Utils::apply2(
      [&](const auto q, auto& r) {
        const auto& result =
            dd->measureOneCollapsing(qState, static_cast<dd::Qubit>(q), mt);
        deref(r).r = result == '1';
      },
      targets, results);
}

template <size_t SIZE>
auto QIR_DD_Backend::reset(std::array<Qubit*, SIZE> qubits) -> void {
  auto targets = translateAddresses(qubits);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    targets[i] = qubitPermutation[targets[i]];
  }
  const qc::NonUnitaryOperation resetOp({targets.data(), targets.data() + SIZE},
                                        qc::Reset);
  qState = applyReset(resetOp, qState, *dd, mt);
}

auto QIR_DD_Backend::swap(Qubit* qubit1, Qubit* qubit2) -> void {
  const auto target1 = translateAddresses(std::array{qubit1})[0];
  const auto target2 = translateAddresses(std::array{qubit2})[0];
  const auto tmp = qubitPermutation[target1];
  qubitPermutation[target1] = qubitPermutation[target2];
  qubitPermutation[target2] = tmp;
}

auto QIR_DD_Backend::qAlloc() -> Qubit* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* qubit = reinterpret_cast<Qubit*>(currentMaxQubitAddress++);
  qRegister.emplace(qubit, currentMaxQubitId++);
  return qubit;
}

auto QIR_DD_Backend::qFree(Qubit* qubit) -> void {
  reset<1>({{qubit}});
  qRegister.erase(qubit);
}

auto QIR_DD_Backend::rAlloc() -> Result* {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  auto* result = reinterpret_cast<Result*>(currentMaxResultAddress++);
  rRegister.emplace(result, ResultStruct{.refcount = 1, .r = false});
  return result;
}

auto QIR_DD_Backend::rFree(Result* result) -> void { rRegister.erase(result); }

auto QIR_DD_Backend::deref(Result* result) -> ResultStruct& {
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

auto QIR_DD_Backend::equal(Result* result1, Result* result2) -> bool {
  return deref(result1).r == deref(result2).r;
}

} // namespace qir
