#include "dummy.h"

#include <gtest/gtest.h>
#include <iostream>

namespace mqt {

TEST(Dummy, dummy) {
  const auto qc = dummyCircuit();
  std::cout << qc.toQASM();
}

} // namespace mqt
