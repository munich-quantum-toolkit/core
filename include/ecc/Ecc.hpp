/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "QuantumComputation.hpp"

class Ecc {
public:
    enum class ID {
        Id,
        Q3Shor,
        Q9Shor,
        Q5Laflamme,
        Q7Steane,
        Q9Surface,
        Q18Surface,
        QxCustom
    };
    struct Info {
        ID          id;
        std::size_t nRedundantQubits;
        std::size_t nCorrectingBits; //in total (not per qubit); usually number of clbits needed for correcting one qubit
        std::string name;
    };

    Ecc(Info ecc, std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFrequency):
        qcOriginal(std::move(qc)), measureFrequency(measureFrequency), ecc(std::move(ecc)) {
        qcMapped = std::make_shared<qc::QuantumComputation>();
    }
    virtual ~Ecc() = default;

    std::shared_ptr<qc::QuantumComputation> apply();

    virtual std::string getName() {
        return ecc.name;
    }

    virtual std::size_t getNOutputQubits(std::size_t nInputQubits) {
        return nInputQubits * ecc.nRedundantQubits + ecc.nCorrectingBits;
    }

    [[nodiscard]] bool verifyExecution(bool simulateWithErrors) const;

protected:
    std::shared_ptr<qc::QuantumComputation> qcOriginal;
    std::shared_ptr<qc::QuantumComputation> qcMapped;
    std::size_t                             measureFrequency;
    bool                                    isDecoded    = true;
    bool                                    gatesWritten = false;
    Info                                    ecc;

    // Set parameters for verifying the eccs
    const size_t            shots                  = 50;
    constexpr static double tolerance              = 0.2;
    const size_t            seed                   = 1;
    const size_t            insertErrorAfterNGates = 1;

    virtual void initMappedCircuit();

    virtual void writeEncoding() {
        if (!isDecoded) {
            return;
        }
        isDecoded = false;
        measureAndCorrect();
    }

    virtual void measureAndCorrect() = 0;

    virtual void writeDecoding() = 0;

    virtual void mapGate(const qc::Operation& gate) = 0;

    void gateNotAvailableError(const qc::Operation& gate) const {
        throw qc::QFRException(std::string("Gate ") + gate.getName() + " not supported to encode in error code " + ecc.name + "!");
    }

    void writeToffoli(dd::Qubit target, dd::Qubit c1, bool p1, dd::Qubit c2, bool p2);

    void writeClassicalControl(dd::Qubit control, dd::QubitCount qubitCount, size_t value, qc::OpType opType, dd::Qubit target);

    //static, since some codes need to store those functions into function pointers
    static void writeXstatic(dd::Qubit target, dd::Control control, const std::shared_ptr<qc::QuantumComputation>& qcMapped);
    static void writeZstatic(dd::Qubit target, dd::Control control, const std::shared_ptr<qc::QuantumComputation>& qcMapped);
};
