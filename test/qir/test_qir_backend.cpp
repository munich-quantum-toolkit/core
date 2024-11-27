#include "dd/Node.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/StandardOperation.hpp"
#include "qir/qir.h"
#include "qir/qir_dd_backend.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <sstream>
#include <streambuf>

namespace mqt {

class QIRDDBackendTest : public ::testing::Test {
protected:
  std::stringstream buffer;
  std::streambuf* old = nullptr;
  // void SetUp() override { old = std::cout.rdbuf(buffer.rdbuf()); }
  // void TearDown() override { std::cout.rdbuf(old); }
};

TEST(DDPackageTest, BellPair) {
  dd::Package<> dd(0);
  dd::vEdge rootEdge = dd::vEdge::one();
  dd.incRef(rootEdge);
  const auto seed = QIR_DD_Backend::generateRandomSeed();
  std::cout << "seed = " << seed << "\n";
  // e.g.: 1160354067710695038
  std::mt19937_64 mt(seed);

  dd.resize(1);
  auto tmp = dd.kronecker(dd.makeZeroState(1), rootEdge, 0);
  dd.incRef(tmp);
  dd.decRef(rootEdge);
  rootEdge = tmp;
  rootEdge.printVector<>();

  const qc::StandardOperation h(0, qc::H);
  tmp = dd.multiply(getDD(&h, dd), rootEdge);
  dd.incRef(tmp);
  dd.decRef(rootEdge);
  rootEdge = tmp;
  rootEdge.printVector<>();

  dd.resize(2);
  tmp = dd.kronecker(dd.makeZeroState(1), rootEdge, 1);
  dd.incRef(tmp);
  dd.decRef(rootEdge);
  rootEdge = tmp;
  rootEdge.printVector<>();

  const qc::StandardOperation cx(0, 1, qc::X);
  tmp = dd.multiply(getDD(&cx, dd), rootEdge);
  dd.incRef(tmp);
  dd.decRef(rootEdge);
  rootEdge = tmp;
  rootEdge.printVector<>();

  const auto m1 = dd.measureOneCollapsing(rootEdge, 0, true, mt);
  const auto m2 = dd.measureOneCollapsing(rootEdge, 1, true, mt);
  EXPECT_EQ(m1, m2);
}

TEST_F(QIRDDBackendTest, BellPairStatic) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  auto* r0 = reinterpret_cast<Result*>(0UL);
  auto* r1 = reinterpret_cast<Result*>(1UL);
  __quantum__rt__initialize(nullptr);
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__mz__body(q0, r0);
  __quantum__qis__mz__body(q1, r1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_THAT(buffer.str(), testing::AnyOf("r0: 0\nr1: 0\n", "r0: 1\nr1: 1\n"));
}

TEST_F(QIRDDBackendTest, BellPairDynamic) {
  __quantum__rt__initialize(nullptr);
  auto* q0 = __quantum__rt__qubit_allocate();
  auto* q1 = __quantum__rt__qubit_allocate();
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  auto* r0 = __quantum__qis__m__body(q0);
  auto* r1 = __quantum__qis__m__body(q1);
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
  const auto m1 = __quantum__rt__read_result(r0);
  const auto m2 = __quantum__rt__read_result(r1);
  EXPECT_EQ(m1, m2);
  __quantum__rt__result_record_output(r0, "r0");
  __quantum__rt__result_record_output(r1, "r1");
  EXPECT_THAT(buffer.str(), testing::AnyOf("r0: 0\nr1: 0\n", "r0: 1\nr1: 1\n"));
  __quantum__rt__result_update_reference_count(r0, -1);
  __quantum__rt__result_update_reference_count(r1, -1);
}

} // namespace mqt
