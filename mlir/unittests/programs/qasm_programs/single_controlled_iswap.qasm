OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
ctrl @ iswap q[0], q[1], q[2];
