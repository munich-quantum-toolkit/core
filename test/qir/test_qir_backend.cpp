#include "qir/qir.h"

#include <gtest/gtest.h>

namespace mqt {

class QIR_DD_BackendTest : public ::testing::Test {
protected:
  std::stringstream buffer;
  std::streambuf* old = nullptr;
  void SetUp() override { old = std::cout.rdbuf(buffer.rdbuf()); }
  void TearDown() override { std::cout.rdbuf(old); }
};

TEST_F(QIR_DD_BackendTest, BellPairStatic) {
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
  // __quantum__rt__result_record_output(r0, "r0");
  // __quantum__rt__result_record_output(r1, "r1");
  // EXPECT_TRUE(buffer.str() == "r0: 0\nr1: 0\n" ||
  //             buffer.str() == "r0: 1\nr1: 1\n");
}

TEST_F(QIR_DD_BackendTest, BellPairDynamic) {
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
  // __quantum__rt__result_record_output(r0, "r0");
  // __quantum__rt__result_record_output(r1, "r1");
  // EXPECT_TRUE(buffer.str() == "r0: 0\nr1: 0\n" ||
  //             buffer.str() == "r0: 1\nr1: 1\n");
  __quantum__rt__result_update_reference_count(r0, -1);
  __quantum__rt__result_update_reference_count(r1, -1);
}

} // namespace mqt
