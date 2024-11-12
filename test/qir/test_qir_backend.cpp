#include "qir/qir.h"

#include <gtest/gtest.h>

namespace mqt {

TEST(Dummy, dummy) {
  const auto qc = dummyCircuit();
  std::cout << qc.toQASM();
}

} // namespace mqt
