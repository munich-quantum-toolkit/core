/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#ifndef QFR_Q7SteaneEcc_HPP
#define QFR_Q7SteaneEcc_HPP

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q7SteaneEcc: public Ecc {
public:
    Q7SteaneEcc(qc::QuantumComputation& qc, int measureFq):
        Ecc(
                {ID::Q7Steane, 7, 3, "Q7Steane"}, qc, measureFq) {}

protected:
    void initMappedCircuit() override;

    void writeEncoding() override;

    void measureAndCorrect() override;
    void measureAndCorrectSingle(bool xSyndrome);

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};

#endif //QFR_Q7SteaneEcc_HPP
