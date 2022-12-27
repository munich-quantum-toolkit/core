/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "ecc/Q18Surface.hpp"

void Q18Surface::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = qcOriginal->getNqubits();
    const auto clAncStart = qcOriginal->getNcbits();

    std::map<std::size_t, std::size_t> xCheckMasks;
    std::map<std::size_t, std::size_t> zCheckMasks;
    for (std::size_t j = 0; j < ancillaWidth; j++) {
        xCheckMasks[xChecks.at(j)] = 1 << j;
        zCheckMasks[zChecks.at(j)] = 1 << j;
    }

    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 36>   qubits        = {};
        std::array<dd::Control, 36> controlQubits = {};
        for (std::size_t j = 0; j < qubits.size(); j++) {
            qubits.at(j) = static_cast<dd::Qubit>(i + j * nQubits);
        }
        for (std::size_t j = 0; j < controlQubits.size(); j++) {
            controlQubits.at(j) = dd::Control{static_cast<dd::Qubit>(qubits.at(j)), dd::Control::Type::pos};
        }

        if (gatesWritten) {
            for (dd::Qubit const ai: ancillaIndices) {
                qcMapped->reset(qubits.at(ai));
            }
        }

        //initialize ancillas: Z-check
        for (const auto& pair: qubitCorrectionX) {
            for (auto ancilla: pair.second) {
                qcMapped->x(qubits[ancilla], controlQubits[pair.first]);
            }
        }

        //initialize ancillas: X-check

        for (std::size_t const xc: zChecks) {
            qcMapped->h(qubits.at(xc));
        }
        for (const auto& pair: qubitCorrectionZ) {
            for (auto ancilla: pair.second) {
                qcMapped->x(qubits[pair.first], controlQubits[ancilla]);
            }
        }
        for (std::size_t const xc: zChecks) {
            qcMapped->h(qubits.at(xc));
        }

        //map ancillas to classical bit result
        for (std::size_t j = 0; j < xChecks.size(); j++) {
            qcMapped->measure(qubits[xChecks.at(j)], clAncStart + j);
        }
        for (std::size_t j = 0; j < zChecks.size(); j++) {
            qcMapped->measure(qubits[zChecks.at(j)], clAncStart + ancillaWidth + j);
        }

        //logic: classical control
        auto controlRegister = std::make_pair(static_cast<dd::Qubit>(clAncStart), ancillaWidth);
        for (const auto& pair: qubitCorrectionX) {
            std::size_t mask = 0;
            for (std::size_t value: pair.second) {
                mask |= xCheckMasks[value];
            }
            classicalControl(controlRegister, mask, qc::X, qubits[pair.first]);
        }

        controlRegister = std::make_pair(static_cast<dd::Qubit>(clAncStart + ancillaWidth), ancillaWidth);
        for (const auto& pair: qubitCorrectionZ) {
            std::size_t mask = 0;
            for (std::size_t value: pair.second) {
                mask |= zCheckMasks[value];
            }
            classicalControl(controlRegister, mask, qc::Z, qubits[pair.first]);
        }

        gatesWritten = true;
    }
}

void Q18Surface::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto                                nQubits               = qcOriginal->getNqubits();
    static constexpr std::array<dd::Qubit, 4> physicalAncillaQubits = {8, 13, 15, 20};
    for (dd::Qubit i = 0; i < nQubits; i++) {
        for (dd::Qubit qubit: physicalAncillaQubits) {
            qcMapped->x(static_cast<dd::Qubit>(i + xInformation * nQubits), dd::Control{static_cast<dd::Qubit>(i + qubit * nQubits), dd::Control::Type::pos});
        }
        qcMapped->measure(static_cast<dd::Qubit>(i + xInformation * nQubits), i);
        qcMapped->reset(static_cast<dd::Qubit>(i));
        qcMapped->x(static_cast<dd::Qubit>(i), dd::Control{static_cast<dd::Qubit>(i + xInformation * nQubits), dd::Control::Type::pos});
    }
    isDecoded = true;
}

void Q18Surface::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal->getNqubits();

    //no control gate decomposition is supported
    if (gate.isControlled() && gate.getType() != qc::Measure) {
        //multi-qubit gates are currently not supported
        gateNotAvailableError(gate);
    } else {
        static constexpr std::array<std::pair<int, int>, 6> swapQubitIndices = {std::make_pair(1, 29), std::make_pair(3, 17), std::make_pair(6, 34), std::make_pair(8, 22), std::make_pair(13, 27), std::make_pair(18, 32)};

        switch (gate.getType()) {
            case qc::I:
                break;
            case qc::X:
                for (auto i: gate.getTargets()) {
                    for (auto j: logicalX) {
                        qcMapped->x(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                }
                break;
            case qc::H:
                //apply H gate to every data qubit
                //swap circuit along '/' axis
                for (auto i: gate.getTargets()) {
                    for (const auto j: dataQubits) {
                        qcMapped->h(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                    for (auto pair: swapQubitIndices) {
                        qcMapped->swap(static_cast<dd::Qubit>(i + pair.first * nQubits), static_cast<dd::Qubit>(i + pair.second * nQubits));
                    }
                    //qubits 5, 10, 15, 20, 25, 30 are along axis
                }
                break;
            case qc::Y:
                //Y = Z X
                for (auto i: gate.getTargets()) {
                    for (auto j: logicalZ) {
                        qcMapped->z(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                    for (auto j: logicalX) {
                        qcMapped->x(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                }
                break;
            case qc::Z:
                for (auto i: gate.getTargets()) {
                    for (auto j: logicalZ) {
                        qcMapped->z(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                }
                break;
            case qc::Measure:
                if (!isDecoded) {
                    measureAndCorrect();
                    writeDecoding();
                }
                if (const auto* measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                    const auto& classics = measureGate->getClassics();
                    const auto& targets  = measureGate->getTargets();
                    for (std::size_t j = 0; j < classics.size(); j++) {
                        qcMapped->measure(targets.at(j), classics.at(j));
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                break;
            default:
                gateNotAvailableError(gate);
        }
    }
}
