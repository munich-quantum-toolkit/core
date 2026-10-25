/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Utils/Verification.h"

#include "dd/Complex.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/ResourceControl.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/DDWorkerBudget.h"
#include "mqt/Dialect/QCO/Utils/VerificationInterface.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <future>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mlir::qco {
namespace {
using Verdict = VerificationVerdict;
using Method = VerificationMethod;
using Relation = VerificationRelation;
using Phase = VerificationPhase;
using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;

// Roots keep intermediate products alive across safe-point garbage collection.
template <class Node> class Root {
public:
  Root(dd::Package& package, dd::Edge<Node> edge)
      : package_(package), edge_(edge) {
    package_.incRef(edge_);
  }
  ~Root() {
    try {
      package_.decRef(edge_);
    } catch (const std::invalid_argument&) {
      // An unregistered root would violate this class's ownership invariant.
      std::terminate();
    }
  }
  Root(const Root&) = delete;
  Root& operator=(const Root&) = delete;
  [[nodiscard]] dd::Edge<Node> get() const { return edge_; }
  void set(dd::Edge<Node> edge) {
    package_.incRef(edge);
    package_.decRef(edge_);
    edge_ = edge;
  }

private:
  dd::Package& package_;
  dd::Edge<Node> edge_;
};
using MatrixRoot = Root<dd::mNode>;
using VectorRoot = Root<dd::vNode>;

struct Prepared {
  const DDResolvedExecution& execution;
  const VerificationInterface& interface;
  std::vector<dd::Qubit> inputMap;
  std::vector<std::pair<dd::Qubit, dd::Qubit>> outputSwaps;
  std::vector<DDOutputBit> outputs;
};
} // namespace

static Prepared prepare(const DDResolvedExecution& execution,
                        const VerificationInterface& interface, size_t width) {
  Prepared result{
      .execution = execution,
      .interface = interface,
      .inputMap = {},
      .outputSwaps = {},
      .outputs = interface.outputs,
  };
  result.inputMap.resize(width);
  std::iota(result.inputMap.begin(), result.inputMap.end(), dd::Qubit{0});
  for (size_t i = 0; i < interface.numQubits; ++i) {
    result.inputMap[interface.inputWires[i]] = static_cast<dd::Qubit>(i);
  }
  // physical[i] is the pre-permutation wire that must end up at output i.
  std::vector<dd::Qubit> physical(width);
  std::iota(physical.begin(), physical.end(), dd::Qubit{0});
  for (size_t i = 0; i < interface.numQubits; ++i) {
    physical[i] = result.inputMap[interface.outputWires[i]];
  }
  std::vector<dd::Qubit> current(width);
  std::iota(current.begin(), current.end(), dd::Qubit{0});
  for (size_t i = 0; i < width; ++i) {
    const auto found = std::find(current.begin() + static_cast<ptrdiff_t>(i),
                                 current.end(), physical[i]);
    const auto j = static_cast<size_t>(std::distance(current.begin(), found));
    if (i != j) {
      result.outputSwaps.emplace_back(static_cast<dd::Qubit>(i),
                                      static_cast<dd::Qubit>(j));
      std::swap(current[i], current[j]);
    }
  }
  std::vector<dd::Qubit> inverseOutput(width);
  for (size_t i = 0; i < interface.numQubits; ++i) {
    inverseOutput[interface.outputWires[i]] = static_cast<dd::Qubit>(i);
  }
  for (auto& bit : result.outputs) {
    if (bit.wire) {
      bit.wire = inverseOutput[*bit.wire];
    }
  }
  return result;
}

static size_t operationCount(const Prepared& program) {
  return program.execution.gates.size() + program.outputSwaps.size();
}

static dd::MatrixDD operationDD(dd::Package& package, const Prepared& program,
                                size_t index) {
  if (index < program.execution.gates.size()) {
    return makeResolvedGateDD(package, program.execution.gates[index],
                              program.inputMap);
  }
  const auto [first, second] =
      program.outputSwaps[index - program.execution.gates.size()];
  static const dd::TwoQubitGateMatrix SWAP{
      {{1., 0., 0., 0.}, {0., 0., 1., 0.}, {0., 1., 0., 0.}, {0., 0., 0., 1.}},
  };
  return package.makeTwoQubitGateDD(SWAP, first, second);
}

static dd::MatrixDD scaled(dd::Package& package, dd::MatrixDD edge,
                           Complex factor) {
  edge.w = package.cn.lookup(static_cast<Complex>(edge.w) * factor);
  return edge;
}

static dd::VectorDD scaled(dd::Package& package, dd::VectorDD edge,
                           Complex factor) {
  edge.w = package.cn.lookup(static_cast<Complex>(edge.w) * factor);
  return edge;
}

namespace {
// Weighted DAG norms retain the root weight, unlike an identity-shape test.
class MatrixNorms {
public:
  explicit MatrixNorms(dd::ResourceControl& resources)
      : resources_(resources), cache_(&resources) {}
  std::pair<double, long double> operator()(dd::MatrixDD edge) {
    resources_.checkpoint();
    const auto weight = static_cast<Complex>(edge.w);
    if (!std::isfinite(weight.real()) || !std::isfinite(weight.imag())) {
      throw std::runtime_error("non-finite matrix weight");
    }
    if (weight == Complex{}) {
      return {0, 0};
    }
    std::pair<double, long double> unweighted{1, 1};
    if (!edge.isTerminal()) {
      const auto found = cache_.find(edge.p);
      if (found != cache_.end()) {
        unweighted = found->second;
      } else {
        unweighted = {0, 0};
        for (const auto& child : edge.p->e) {
          const auto [maximum, norm] = (*this)(child);
          unweighted.first = std::max(unweighted.first, maximum);
          unweighted.second += norm / 2;
        }
        cache_.emplace(edge.p, unweighted);
      }
    }
    const auto magnitude = std::abs(weight);
    const auto maximum = magnitude * unweighted.first;
    const auto norm =
        static_cast<long double>(magnitude) * magnitude * unweighted.second;
    if (!std::isfinite(maximum) || !std::isfinite(norm)) {
      throw std::runtime_error("non-finite matrix norm");
    }
    return {maximum, norm};
  }

private:
  dd::ResourceControl& resources_;
  std::pmr::unordered_map<dd::mNode*, std::pair<double, long double>> cache_;
};
} // namespace

template <class Node>
static double entryMaximum(dd::Edge<Node> edge, dd::ResourceControl& resources,
                           std::pmr::unordered_map<Node*, double>& cache) {
  resources.checkpoint();
  const auto magnitude = std::abs(static_cast<Complex>(edge.w));
  if (!std::isfinite(magnitude)) {
    throw std::runtime_error("non-finite entry weight");
  }
  if (magnitude == 0 || edge.isTerminal()) {
    return magnitude;
  }
  double maximum = 0;
  if (const auto found = cache.find(edge.p); found != cache.end()) {
    maximum = found->second;
  } else {
    for (const auto& child : edge.p->e) {
      maximum = std::max(maximum, entryMaximum(child, resources, cache));
    }
    cache.emplace(edge.p, maximum);
  }
  const auto weighted = magnitude * maximum;
  if (!std::isfinite(weighted)) {
    throw std::runtime_error("non-finite entry maximum");
  }
  return weighted;
}

template <class Node>
static double entryMaximum(dd::Edge<Node> edge,
                           dd::ResourceControl& resources) {
  std::pmr::unordered_map<Node*, double> cache(&resources);
  return entryMaximum(edge, resources, cache);
}

static dd::MatrixDD projector(dd::Package& package, dd::Qubit wire,
                              bool value) {
  const dd::GateMatrix matrix = value ? dd::GateMatrix{{0., 0., 0., 1.}}
                                      : dd::GateMatrix{{1., 0., 0., 0.}};
  return package.makeGateDD(matrix, wire);
}

static void construct(dd::Package& package, MatrixRoot& root,
                      const Prepared& program,
                      const std::vector<bool>& zeroInputs) {
  for (size_t i = 0; i < zeroInputs.size(); ++i) {
    if (zeroInputs[i]) {
      root.set(package.multiply(
          projector(package, static_cast<dd::Qubit>(i), false), root.get()));
    }
  }
  for (size_t i = 0; i < operationCount(program); ++i) {
    package.checkResources();
    const auto gate = operationDD(package, program, i);
    root.set(package.multiply(gate, root.get()));
    package.garbageCollect();
  }
}

static bool cleanOutputs(dd::Package& package, dd::ResourceControl& resources,
                         dd::MatrixDD root, const Prepared& program,
                         double tolerance) {
  for (const auto wire : program.interface.cleanOutputs) {
    MatrixRoot leakage(
        package,
        package.multiply(projector(package, static_cast<dd::Qubit>(wire), true),
                         root));
    if (entryMaximum(leakage.get(), resources) > tolerance) {
      return false;
    }
  }
  return true;
}

static VerificationCheckerResult
approximateResult(Method method, double distance,
                  const VerificationOptions& options) {
  VerificationCheckerResult result{.method = method, .distance = distance};
  if (!std::isfinite(distance)) {
    result.reason = "non-finite numerical result";
  } else if (distance < options.tolerance - options.numericalTolerance) {
    result.verdict = Verdict::Equivalent;
    result.reason = "distance below the acceptance interval";
  } else if (distance > options.tolerance + options.numericalTolerance) {
    result.verdict = Verdict::NotEquivalent;
    result.reason = "distance above the acceptance interval";
  } else {
    result.reason = "distance lies in the numerical uncertainty interval";
  }
  return result;
}

static VerificationCheckerResult
compare(dd::Package& package, dd::ResourceControl& resources, dd::MatrixDD left,
        dd::MatrixDD right, size_t width, size_t freeInputs,
        const VerificationOptions& options, Method method) {
  VerificationCheckerResult result{.method = method};
  Complex phase = 1;
  if (options.phase == Phase::Ignore) {
    MatrixRoot adjoint(package, package.conjugateTranspose(left));
    MatrixRoot overlap(package, package.multiply(adjoint.get(), right));
    const auto trace =
        static_cast<Complex>(package.trace(overlap.get(), width));
    if (!std::isfinite(std::abs(trace))) {
      result.reason = "non-finite phase overlap";
      return result;
    }
    if (std::abs(trace) != 0) {
      phase = std::conj(trace) / std::abs(trace);
    }
    result.phase = phase;
  }
  MatrixRoot residual(package,
                      package.add(left, scaled(package, right, -phase)));
  const auto [maximum, fullNorm] = MatrixNorms(resources)(residual.get());
  const auto squared =
      std::ldexp(fullNorm, static_cast<int>(width - freeInputs));
  if (!std::isfinite(squared)) {
    result.reason = "non-finite residual norm";
    return result;
  }
  const auto normalized = std::sqrt(static_cast<double>(squared));
  if (options.relation != Relation::Exact) {
    // Absorb only a few ulps at the orthogonal-isometry boundary.
    const auto rangeGuard =
        std::min(options.numericalTolerance,
                 64 * std::numeric_limits<double>::epsilon());
    if (options.relation == Relation::ProcessDistance &&
        squared > 2 + rangeGuard) {
      result.reason = "phase-aligned residual exceeds the isometry bound";
      return result;
    }
    const auto phaseAlignedSquared = std::min(squared, 2.L);
    const auto distance =
        options.relation == Relation::AmplitudeDistance
            ? normalized
            : std::sqrt(static_cast<double>(phaseAlignedSquared *
                                            (1 - phaseAlignedSquared / 4)));
    auto approximate = approximateResult(method, distance, options);
    approximate.phase = result.phase;
    return approximate;
  }
  result.distance = maximum;
  if (method == Method::Construction) {
    if (maximum <= options.tolerance) {
      result.verdict = Verdict::Equivalent;
      result.reason = "maximum entry residual meets the contract";
    } else if (options.phase == Phase::Preserve ||
               normalized > std::exp2(static_cast<double>(width) / 2) *
                                options.tolerance) {
      result.verdict = Verdict::NotEquivalent;
      result.reason =
          "entry residual exceeds the contract for every allowed phase";
    } else {
      result.reason = "the overlap phase is insufficient to decide the "
                      "maximum-entry relation";
    }
  } else {
    // Multiplication by a unitary preserves Frobenius norm, not max entry norm.
    const auto frobenius =
        normalized * std::exp2(static_cast<double>(freeInputs) / 2);
    result.distance = frobenius;
    if (frobenius <= options.tolerance) {
      result.verdict = Verdict::Equivalent;
      result.reason = "miter Frobenius bound proves the entry tolerance";
    } else if (frobenius >
               std::exp2(static_cast<double>(width)) * options.tolerance) {
      result.verdict = Verdict::NotEquivalent;
      result.reason = "miter Frobenius bound refutes the entry tolerance";
    } else {
      result.reason = "miter bound needs a construction comparison";
    }
  }
  return result;
}

static void outcomeEffect(dd::Package& package, MatrixRoot& effect,
                          dd::MatrixDD unitary, const Prepared& program,
                          size_t outcome) {
  effect.set(unitary);
  for (size_t i = 0; i < program.outputs.size(); ++i) {
    const auto value = ((outcome >> i) & 1U) != 0;
    const auto& bit = program.outputs[i];
    if (bit.constant) {
      if (*bit.constant != value) {
        effect.set(dd::MatrixDD::zero());
        return;
      }
    } else {
      effect.set(
          package.multiply(projector(package, *bit.wire, value), effect.get()));
    }
  }
  MatrixRoot adjoint(package, package.conjugateTranspose(unitary));
  effect.set(package.multiply(adjoint.get(), effect.get()));
}

static VerificationCheckerResult
construction(dd::ResourceControl& resources, const Prepared& left,
             const Prepared& right, const std::vector<bool>& zeroInputs,
             size_t freeInputs, const VerificationOptions& options) {
  if (options.relation == Relation::TerminalProbabilities &&
      (left.outputs.size() >= std::numeric_limits<size_t>::digits ||
       (size_t{1} << left.outputs.size()) > options.limits.maxOutcomes)) {
    return {
        .method = Method::Construction,
        .reason = "terminal outcome limit exceeded",
    };
  }
  dd::Package package(zeroInputs.size(), {}, &resources);
  MatrixRoot lhs(package, dd::Package::makeIdent());
  MatrixRoot rhs(package, dd::Package::makeIdent());
  construct(package, lhs, left, zeroInputs);
  construct(package, rhs, right, zeroInputs);
  if (options.relation == Relation::TerminalProbabilities) {
    VerificationCheckerResult result{.method = Method::Construction};
    const auto outcomes = size_t{1} << left.outputs.size();
    double maximum = 0;
    for (size_t outcome = 0; outcome < outcomes; ++outcome) {
      MatrixRoot a(package, dd::MatrixDD::zero());
      MatrixRoot b(package, dd::MatrixDD::zero());
      outcomeEffect(package, a, lhs.get(), left, outcome);
      outcomeEffect(package, b, rhs.get(), right, outcome);
      MatrixRoot difference(
          package, package.add(a.get(), scaled(package, b.get(), -1.)));
      maximum = std::max(maximum, entryMaximum(difference.get(), resources));
      if (maximum > options.tolerance) {
        result.verdict = Verdict::NotEquivalent;
        result.reason = "terminal outcome effects differ";
        result.distance = maximum;
        return result;
      }
      package.garbageCollect();
    }
    result.verdict = Verdict::Equivalent;
    result.reason = "all selected terminal outcome effects agree";
    result.distance = maximum;
    return result;
  }
  if (!cleanOutputs(package, resources, lhs.get(), left,
                    options.numericalTolerance) ||
      !cleanOutputs(package, resources, rhs.get(), right,
                    options.numericalTolerance)) {
    return {
        .method = Method::Construction,
        .verdict = Verdict::Unsupported,
        .reason =
            "coherent comparison cannot discard a nonzero workspace output",
    };
  }
  return compare(package, resources, lhs.get(), rhs.get(), zeroInputs.size(),
                 freeInputs, options, Method::Construction);
}

static VerificationCheckerResult
alternating(dd::ResourceControl& resources, const Prepared& left,
            const Prepared& right, size_t width,
            const VerificationOptions& options, bool serialAuto) {
  dd::Package package(width, {}, &resources);
  MatrixRoot miter(package, dd::Package::makeIdent());
  const auto count = std::max(operationCount(left), operationCount(right));
  for (size_t i = 0; i < count; ++i) {
    if (serialAuto && i == 1024) {
      return {
          .method = Method::Alternating,
          .reason =
              "serial alternating slice ended; construction remains eligible",
      };
    }
    package.checkResources();
    if (i < operationCount(left)) {
      miter.set(package.multiply(operationDD(package, left, i), miter.get()));
    }
    if (i < operationCount(right)) {
      const auto gate = operationDD(package, right, i);
      const auto adjoint = package.conjugateTranspose(gate);
      miter.set(package.multiply(miter.get(), adjoint));
    }
    package.garbageCollect();
  }
  return compare(package, resources, miter.get(), dd::Package::makeIdent(),
                 width, width, options, Method::Alternating);
}

static void simulate(dd::Package& package, VectorRoot& state,
                     const Prepared& program) {
  for (size_t i = 0; i < operationCount(program); ++i) {
    package.checkResources();
    state.set(package.multiply(operationDD(package, program, i), state.get()));
    package.garbageCollect();
  }
}

static VerificationCheckerResult
simulation(dd::ResourceControl& resources, const Prepared& left,
           const Prepared& right, const std::vector<bool>& zeroInputs,
           size_t freeInputs, const VerificationOptions& options) {
  dd::Package package(zeroInputs.size(), {}, &resources);
  std::mt19937_64 random(options.seed);
  VerificationCheckerResult result{.method = Method::Simulation};
  const auto count = freeInputs == 0 ? size_t{1} : options.maxSimulationSamples;
  for (size_t sample = 0; sample < count; ++sample) {
    std::vector<dd::BasisStates> basis(zeroInputs.size(),
                                       dd::BasisStates::zero);
    size_t superpositions = 0;
    for (size_t i = 0; i < basis.size(); ++i) {
      if (!zeroInputs[i]) {
        basis[i] =
            static_cast<dd::BasisStates>(sample < 2 ? sample : random() % 6);
        superpositions += static_cast<unsigned>(basis[i]) >= 2 ? 1 : 0;
      }
    }
    const auto input = dd::makeBasisState(basis.size(), basis, package);
    VectorRoot lhs(package, input);
    VectorRoot rhs(package, input);
    package.decRef(input);
    simulate(package, lhs, left);
    simulate(package, rhs, right);
    Complex phase = 1;
    if (options.phase == Phase::Ignore) {
      const auto overlap =
          static_cast<Complex>(package.innerProduct(lhs.get(), rhs.get()));
      if (!std::isfinite(std::abs(overlap))) {
        result.reason = "non-finite state overlap";
        return result;
      }
      if (std::abs(overlap) != 0) {
        phase = std::conj(overlap) / std::abs(overlap);
      }
    }
    VectorRoot difference(
        package, package.add(lhs.get(), scaled(package, rhs.get(), -phase)));
    const auto squaredNorm =
        package.innerProduct(difference.get(), difference.get()).r;
    if (!std::isfinite(squaredNorm) || squaredNorm < 0) {
      result.reason = "invalid state residual norm";
      return result;
    }
    const auto norm = std::sqrt(squaredNorm);
    ++result.samples;
    if (freeInputs == 0) {
      for (const auto* program : {&left, &right}) {
        const auto state = program == &left ? lhs.get() : rhs.get();
        for (const auto wire : program->interface.cleanOutputs) {
          VectorRoot leakage(
              package, package.multiply(projector(package, wire, true), state));
          if (entryMaximum(leakage.get(), resources) >
              options.numericalTolerance) {
            result.verdict = Verdict::Unsupported;
            result.reason = "fixed input leaves a nonzero workspace output";
            return result;
          }
        }
      }
      const auto maximum = entryMaximum(difference.get(), resources);
      result.distance = maximum;
      if (maximum <= options.tolerance) {
        result.verdict = Verdict::Equivalent;
        result.reason =
            "the sole initialized input meets the entry-residual contract";
        return result;
      }
      if (options.phase == Phase::Preserve) {
        result.verdict = Verdict::NotEquivalent;
        result.reason =
            "the sole initialized input violates the entry-residual contract";
        return result;
      }
    }
    // Every entry of A-B <= t implies ||(A-B)psi||2 <= sqrt(M)t||psi||1.
    const auto bound =
        options.tolerance *
        std::exp2(static_cast<double>(zeroInputs.size() + superpositions) / 2);
    if (norm > bound + options.numericalTolerance) {
      result.verdict = Verdict::NotEquivalent;
      result.reason = "sample exceeds the all-input entry-residual bound";
      for (const auto state : basis) {
        result.counterexample.push_back(static_cast<unsigned>(state));
      }
      result.phase = phase;
      result.distance = norm;
      return result;
    }
    package.garbageCollect();
  }
  result.reason =
      "finite basis/independent Pauli-product samples found no counterexample";
  return result;
}

static bool decisive(Verdict verdict) {
  return verdict == Verdict::Equivalent || verdict == Verdict::NotEquivalent;
}

static FailureOr<VerificationResult>
verifyImpl(func::FuncOp left, func::FuncOp right,
           const VerificationContract& contract,
           const DDArgumentBindings& leftBindings,
           const DDArgumentBindings& rightBindings,
           const VerificationOptions& options, std::stop_token stop) {
  if (!left || !right || !std::isfinite(options.tolerance) ||
      options.tolerance < 0 || !std::isfinite(options.numericalTolerance) ||
      options.numericalTolerance < 0 || options.limits.workers == 0 ||
      options.limits.timeout.count() < 0 ||
      (options.relation == Relation::ProcessDistance &&
       options.phase != Phase::Ignore) ||
      (options.relation == Relation::AmplitudeDistance &&
       options.phase != Phase::Preserve) ||
      (options.relation != Relation::TerminalProbabilities &&
       !contract.observedOutputs.empty())) {
    return failure();
  }
  auto leftModule = left->getParentOfType<ModuleOp>();
  auto rightModule = right->getParentOfType<ModuleOp>();
  if (!leftModule || !rightModule || failed(mlir::verify(leftModule)) ||
      failed(mlir::verify(rightModule)) ||
      failed(verifyLinearity(leftModule)) ||
      failed(verifyLinearity(rightModule))) {
    return failure();
  }
  VerificationResult result;
  result.contract = contract;
  result.tolerance = options.tolerance;
  result.numericalTolerance = options.numericalTolerance;
  const auto recordBindings = [](func::FuncOp func,
                                 const DDArgumentBindings& bindings) {
    std::vector<std::string> values;
    for (size_t i = 0; i < func.getNumArguments(); ++i) {
      if (auto attr = bindings.lookup(func.getArgument(i))) {
        std::string text = "arg" + std::to_string(i) + "=";
        llvm::raw_string_ostream stream(text);
        attr.print(stream);
        values.push_back(std::move(text));
      }
    }
    return values;
  };
  result.leftBindings = recordBindings(left, leftBindings);
  result.rightBindings = recordBindings(right, rightBindings);
  result.relation = options.relation;
  result.phase = options.phase;
  result.ddTolerance = dd::RealNumber::eps;
  const auto now = Clock::now();
  const auto remainingDuration =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          Clock::time_point::max() - now);
  const auto deadline = options.limits.timeout.count() > 0 &&
                                options.limits.timeout < remainingDuration
                            ? now + options.limits.timeout
                            : Clock::time_point::max();
  std::stop_source cancellation;
  const std::stop_callback externalCancellation(
      std::move(stop), [&] { cancellation.request_stop(); });
  const auto token = cancellation.get_token();
  const DDTraceOptions traceOptions{
      .maxSteps = options.limits.maxSteps,
      .maxCallDepth = options.limits.maxCallDepth,
      .maxGates = options.limits.maxGates,
      .maxBytes = options.limits.maxExecutionBytes,
      .maxQubits = options.limits.maxQubits,
      .deadline = deadline,
      .stop = token,
  };
  const auto lhs = resolveExecution(left, leftBindings, traceOptions);
  auto remaining = traceOptions;
  remaining.maxSteps -= std::min(remaining.maxSteps, lhs.steps);
  remaining.maxGates -= std::min(remaining.maxGates, lhs.gates.size());
  remaining.maxBytes -= std::min(remaining.maxBytes, lhs.storageBytes);
  const auto rhs = lhs.status == DDTraceStatus::Complete
                       ? resolveExecution(right, rightBindings, remaining)
                       : DDResolvedExecution{};
  if (lhs.status != DDTraceStatus::Complete ||
      rhs.status != DDTraceStatus::Complete) {
    const auto& failedTrace = lhs.status != DDTraceStatus::Complete ? lhs : rhs;
    result.verdict = failedTrace.status == DDTraceStatus::Unsupported
                         ? Verdict::Unsupported
                         : Verdict::Inconclusive;
    result.reason = failedTrace.message;
    return result;
  }
  const auto leftInterface = resolveVerificationInterface(left, lhs, contract);
  if (failed(leftInterface)) {
    result.verdict = Verdict::Unsupported;
    result.reason = "left program has no supported verification interface";
    return result;
  }
  const auto rightInterface = resolveVerificationInterface(
      right, rhs, contract, leftInterface->numInputs);
  if (failed(rightInterface)) {
    result.verdict = Verdict::Unsupported;
    result.reason =
        "right program does not implement the shared source interface";
    return result;
  }
  result.numInputs = leftInterface->numInputs;
  result.numQubits =
      std::max(leftInterface->numQubits, rightInterface->numQubits);
  result.resolvedGates = lhs.gates.size() + rhs.gates.size();
  result.outputLabels = leftInterface->outputLabels;
  if (result.numQubits > options.limits.maxQubits) {
    result.reason = "layout width exceeds the qubit limit";
    return result;
  }
  std::vector<bool> zeroInputs(result.numQubits, true);
  for (size_t i = 0; i < result.numInputs; ++i) {
    if (leftInterface->zeroInputs[i] != rightInterface->zeroInputs[i]) {
      result.verdict = Verdict::Unsupported;
      result.reason = "programs disagree on the free quantum input domain";
      return result;
    }
    zeroInputs[i] = leftInterface->zeroInputs[i];
    result.numFreeInputs += !zeroInputs[i] ? 1 : 0;
  }
  const auto a = prepare(lhs, *leftInterface, result.numQubits);
  const auto b = prepare(rhs, *rightInterface, result.numQubits);
  if (leftInterface->outputLabels != rightInterface->outputLabels) {
    result.verdict = Verdict::NotEquivalent;
    result.reason = "classical output identities differ";
    return result;
  }
  if (options.relation != Relation::TerminalProbabilities) {
    for (size_t i = 0; i < a.outputs.size(); ++i) {
      if (a.outputs[i].wire != b.outputs[i].wire ||
          a.outputs[i].constant != b.outputs[i].constant) {
        result.verdict = Verdict::NotEquivalent;
        result.reason =
            "terminal readout sources differ under the shared output map";
        return result;
      }
    }
  } else if (a.outputs.empty()) {
    result.verdict = Verdict::Unsupported;
    result.reason =
        "terminal observational checking requires an observed classical output";
    return result;
  }
  const bool canAlternate = result.numFreeInputs == result.numQubits &&
                            options.relation != Relation::TerminalProbabilities;
  // Finite samples cannot prove the clean-workspace premise for free inputs.
  const bool canSimulate =
      options.relation == Relation::Exact &&
      (result.numFreeInputs == 0 || (leftInterface->cleanOutputs.empty() &&
                                     rightInterface->cleanOutputs.empty()));
  std::vector<Method> methods;
  if (options.method == Method::Auto) {
    if (canSimulate && options.maxSimulationSamples > 0) {
      methods.push_back(Method::Simulation);
    }
    if (canAlternate) {
      methods.push_back(Method::Alternating);
    }
    methods.push_back(Method::Construction);
  } else if ((options.method == Method::Alternating && !canAlternate) ||
             (options.method == Method::Simulation && !canSimulate)) {
    result.verdict = Verdict::Unsupported;
    result.reason =
        "requested checker is ineligible for this relation or input domain";
    return result;
  } else {
    methods.push_back(options.method);
  }
  const auto grant = acquireDDWorkers(
      std::min(options.limits.workers, methods.size()), token, deadline);
  if (grant == 0) {
    result.reason = "cancelled or timed out while waiting for a worker";
    return result;
  }
  const auto releaseWorkers =
      llvm::make_scope_exit([&] { releaseDDWorkers(grant); });
  dd::ResourceControl resources({
      .stop = token,
      .deadline = deadline,
      .maxBytes = options.limits.maxDDBytes,
  });
  result.checkers.resize(methods.size());
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (;;) {
      const auto index = next.fetch_add(1);
      if (index >= methods.size()) {
        return;
      }
      auto& check = result.checkers[index];
      const auto method = methods[index];
      check.method = method;
      const auto start = Clock::now();
      try {
        resources.checkpoint();
        switch (method) {
        case Method::Construction:
          check = construction(resources, a, b, zeroInputs,
                               result.numFreeInputs, options);
          break;
        case Method::Alternating:
          check = alternating(resources, a, b, result.numQubits, options,
                              grant == 1 && options.method == Method::Auto);
          break;
        case Method::Simulation:
          check = simulation(resources, a, b, zeroInputs, result.numFreeInputs,
                             options);
          break;
        case Method::Auto:
          break;
        }
      } catch (const dd::ResourceLimitExceeded& error) {
        check.reason = error.what();
      } catch (const std::bad_alloc&) {
        check.reason = "host allocation failed";
      } catch (const std::exception& error) {
        check.reason = error.what();
      }
      check.seconds =
          std::chrono::duration<double>(Clock::now() - start).count();
      if (decisive(check.verdict)) {
        cancellation.request_stop();
      }
    }
  };
  std::vector<std::future<void>> workers;
  try {
    for (size_t i = 1; i < grant; ++i) {
      workers.push_back(std::async(std::launch::async, worker));
    }
  } catch (...) {
    cancellation.request_stop();
    for (auto& future : workers) {
      future.get();
    }
    throw;
  }
  worker();
  for (auto& future : workers) {
    future.get();
  }
  result.peakDDBytes = resources.peakBytes();
  for (const auto& check : result.checkers) {
    if (decisive(check.verdict)) {
      if (decisive(result.verdict) && result.verdict != check.verdict) {
        result.verdict = Verdict::Inconclusive;
        result.reason =
            "completed checkers produced conflicting numerical evidence";
        result.distance.reset();
        return result;
      }
      result.verdict = check.verdict;
      result.reason = check.reason;
      result.distance = check.distance;
    }
  }
  if (!decisive(result.verdict)) {
    const auto unsupported =
        std::ranges::find_if(result.checkers, [](const auto& check) {
          return check.verdict == Verdict::Unsupported;
        });
    if (unsupported != result.checkers.end()) {
      result.verdict = Verdict::Unsupported;
      result.reason = unsupported->reason;
    } else {
      result.reason = "no checker completed a decisive comparison";
      if (result.checkers.size() == 1) {
        result.reason = result.checkers.front().reason;
      }
    }
  }
  return result;
}
FailureOr<VerificationResult> verify(func::FuncOp left, func::FuncOp right,
                                     const VerificationContract& contract,
                                     const DDArgumentBindings& leftBindings,
                                     const DDArgumentBindings& rightBindings,
                                     const VerificationOptions& options,
                                     std::stop_token stop) {
  try {
    return verifyImpl(left, right, contract, leftBindings, rightBindings,
                      options, std::move(stop));
  } catch (const std::bad_alloc&) {
    return VerificationResult{
        .reason = "host allocation failed before checking completed",
    };
  } catch (const std::system_error& error) {
    return VerificationResult{.reason = error.what()};
  }
}
} // namespace mlir::qco
