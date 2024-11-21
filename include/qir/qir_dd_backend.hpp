#pragma once

#include "dd/Package.hpp"
#include "ir/QuantumComputation.hpp"

namespace mqt {
/**
 * @note This class is implemented following the design pattern Singleton in
 * order to access an instance of this class from the C function without having
 * a handle to it.
 */
class QIR_DD_Backend {
private:
  enum class AddressMode : uint8_t { UNKNOWN, DYNAMIC, STATIC };

  AddressMode addressMode;
  std::unordered_map<qc::Qubit, qc::Qubit> qRegister;
  dd::Package<> dd;
  dd::vEdge qState{};

  QIR_DD_Backend();

  auto determineAddressMode() -> void;

public:
  static QIR_DD_Backend& getInstance() {
    static QIR_DD_Backend instance;
    return instance;
  }

  QIR_DD_Backend(const QIR_DD_Backend&) = delete;
  QIR_DD_Backend& operator=(const QIR_DD_Backend&) = delete;

  template <typename... Args> auto apply(qc::OpType op, Args... qubits) -> void;
};
} // namespace mqt
