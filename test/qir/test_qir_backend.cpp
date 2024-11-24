#include "qir/qir.h"

#include <gtest/gtest.h>

namespace mqt {

TEST(QIR_DD_Backend, BellPairStatic) {
  auto* q0 = reinterpret_cast<Qubit*>(0UL);
  auto* q1 = reinterpret_cast<Qubit*>(1UL);
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__m__body(q0, __quantum__rt__result_get_zero());
  __quantum__qis__m__body(q1, __quantum__rt__result_get_zero());
}

TEST(QIR_DD_Backend, BellPairDynamic) {
  auto q0 = __quantum__rt__qubit_allocate();
  auto q1 = __quantum__rt__qubit_allocate();
  __quantum__qis__h__body(q0);
  __quantum__qis__cx__body(q0, q1);
  __quantum__qis__m__body(q0, __quantum__rt__result_get_zero());
  __quantum__qis__m__body(q1, __quantum__rt__result_get_zero());
  __quantum__rt__qubit_release(q0);
  __quantum__rt__qubit_release(q1);
}

} // namespace mqt
